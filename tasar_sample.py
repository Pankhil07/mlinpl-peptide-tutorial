#!/usr/bin/env python3
"""
Oracle-guided peptide sampling from a Hyformer checkpoint using TASAR.

Unlike `generate.py`, which samples sequences independently from the language
model, TASAR runs rounds of stochastic beam search and reweights the search
tree by an oracle score after each round. Because Hyformer carries both a
language-modelling and a prediction head, the oracle can be the model's own
property prediction — the checkpoint proposes sequences and scores them.

Example
-------
python tasar_sample.py \
    --model_config_path     configs/models/hyformer/config.json \
    --tokenizer_config_path configs/tokenizers/amino_acid/config.json \
    --dataset_config_path   configs/datasets/sequential_amp/config11.json \
    --ckpt_path             checkpoints/mic_ckpt.pt \
    --objective             selectivity \
    --n_rounds              20 \
    --out_csv               tasar_sequences.csv
"""

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.generators import TasarSampler
from hyformer.models.auto import AutoModel
from hyformer.utils.peptide_rules import is_peptide_ok, is_peptide_probable, is_valid_peptide
from hyformer.utils.tokenizers.auto import AutoTokenizer

console = logging.getLogger("tasar_sample")
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.set_float32_matmul_precision("high")

# Reward returned for sequences that are not valid peptides at all. Finite, so
# the advantage update stays well behaved, but low enough to be discouraged.
INVALID_REWARD = -10.0

# Panel used by the `selectivity` objective, in the order the 11-task MIC
# checkpoint was trained on.
MIC_LABELS = [
    "A. baumannii ATCC 19606",          # 00
    "E. coli ATCC 11775",               # 01
    "E. coli AIC221",                   # 02
    "E. coli AIC222 - CRE",             # 03
    "K. pneumoniae ATCC 13883",         # 04
    "P. aeruginosa PAO1",               # 05
    "P. aeruginosa PA14",               # 06
    "S. aureus ATCC 12600",             # 07
    "S. aureus ATCC BAA-1556 - MRSA",   # 08
    "E. faecalis ATCC 700802 - VRE",    # 09
    "E. faecium ATCC 700221 - VRE",     # 10
]
GRAM_NEGATIVE_IDXS = [0, 1, 2, 3, 4, 5, 6]
GRAM_POSITIVE_IDXS = [7, 8, 9, 10]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(args, dataset_config, device):
    tokenizer = AutoTokenizer.from_config(
        TokenizerConfig.from_config_filepath(args.tokenizer_config_path)
    )
    model = AutoModel.from_config(
        ModelConfig.from_config_filepath(args.model_config_path),
        prediction_task_type=dataset_config.prediction_task_type,
        num_prediction_tasks=dataset_config.num_prediction_tasks,
    )
    console.info(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # handle both wrapped checkpoints and bare state dicts
    # Keep the prediction head: it is the oracle.
    model.load_pretrained(state_dict=state, discard_prediction_head=False)
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def predict(model, input_ids, pred_token_id, pad_id):
    """Score one sequence with the prediction head. Returns a (n_tasks,) array."""
    pred_ids = input_ids.clone()
    pred_ids[:, 0] = pred_token_id  # swap the <lm> task token for <prediction>
    attention_mask = (pred_ids != pad_id).long()
    out = model(
        input_ids=pred_ids,
        attention_mask=attention_mask,
        task="prediction",
        return_loss=False,
    )
    return out["logits"].detach().cpu().numpy()[0]


def build_oracle(args, model, tokenizer, pred_token_id, pad_id, device):
    """Build the reward used by TASAR.

    Returns (oracle_fn, properties, reward_from_properties): the penalized
    reward TASAR optimizes, the raw prediction-head output, and the unpenalized
    objective, so the CSV can report all three.
    """

    def properties(seq_ids: torch.Tensor) -> np.ndarray:
        return predict(model, seq_ids.unsqueeze(0).to(device), pred_token_id, pad_id)

    def reward_from_properties(preds: np.ndarray) -> float:
        if args.objective == "selectivity":
            # Potency against Gram-negatives relative to Gram-positives. The
            # head predicts pMIC, so higher is more potent; subtracting the
            # Gram-positive arm rewards selective, not just broadly toxic,
            # peptides.
            preds = np.maximum(preds, 0.0)
            return float(preds[GRAM_NEGATIVE_IDXS].mean() - preds[GRAM_POSITIVE_IDXS].mean())
        value = float(preds[args.target_task])
        return -value if args.minimize else value

    def oracle_fn(seq_ids: torch.Tensor) -> float:
        seq = tokenizer.decode(seq_ids.tolist(), skip_special_tokens=True).strip().upper()
        if not is_valid_peptide(seq, args.min_len, args.max_len):
            return INVALID_REWARD

        preds = properties(seq_ids)
        if not np.all(np.isfinite(preds)):
            return INVALID_REWARD

        reward = reward_from_properties(preds)
        if args.rule_penalty > 0.0:
            if not is_peptide_ok(seq):
                reward -= args.rule_penalty
            if not is_peptide_probable(seq):
                reward -= args.rule_penalty
        return reward

    return oracle_fn, properties, reward_from_properties


def sample(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    console.info(f"Device: {device}")

    # Validate the objective against the head size before loading the weights,
    # so a mismatch reports the flag to fix rather than a state_dict error.
    dataset_config = DatasetConfig.from_config_filepath(args.dataset_config_path)

    if args.objective == "selectivity" and dataset_config.num_prediction_tasks != len(MIC_LABELS):
        raise SystemExit(
            f"--objective selectivity needs the {len(MIC_LABELS)}-task MIC checkpoint, but the "
            f"dataset config declares {dataset_config.num_prediction_tasks} task(s). "
            "Use --objective predict, or point --dataset_config_path at "
            "configs/datasets/sequential_amp/config11.json."
        )
    if args.objective == "predict" and not 0 <= args.target_task < dataset_config.num_prediction_tasks:
        raise SystemExit(
            f"--target_task {args.target_task} is out of range for a "
            f"{dataset_config.num_prediction_tasks}-task checkpoint."
        )

    model, tokenizer = load_model(args, dataset_config, device)

    lm_token_id   = tokenizer.task_token_id("lm")
    pred_token_id = tokenizer.task_token_id("prediction")
    bos_id        = tokenizer.bos_token_id
    eos_id        = tokenizer.eos_token_id
    pad_id        = tokenizer.pad_token_id

    # Fixed prefix: [<lm>, <s>]. TASAR explores one tree, so no batch dimension.
    prefix = torch.tensor([lm_token_id, bos_id], dtype=torch.long, device=device)

    oracle_fn, properties, reward_from_properties = build_oracle(
        args, model, tokenizer, pred_token_id, pad_id, device
    )
    # A constant rescaling: controls how hard a good sequence pulls the search
    # tree toward itself in the next round.
    advantage_fn = lambda reward: args.advantage_scale * reward  # noqa: E731

    sampler = TasarSampler(model, prediction_task_type=dataset_config.prediction_task_type)

    n_tasks = dataset_config.num_prediction_tasks
    label_names = MIC_LABELS if n_tasks == len(MIC_LABELS) else [f"task_{i}" for i in range(n_tasks)]

    results = {}  # sequence -> row
    console.info(
        f"Running {args.n_rounds} TASAR round(s), beam_width={args.beam_width}, "
        f"replan_steps={args.replan_steps}"
    )

    for round_idx in range(args.n_rounds):
        round_seed = args.seed + round_idx
        set_seed(round_seed)
        rng = np.random.default_rng(round_seed)

        with torch.inference_mode():
            samples = sampler.generate(
                prefix_input_ids=prefix,
                oracle_fn=oracle_fn,
                advantage_fn=advantage_fn,
                eos_token_id=eos_id,
                beam_width=args.beam_width,
                nucleus_top_p=args.nucleus_top_p,
                temperature=args.temperature,
                max_sequence_length=args.max_len,
                replan_steps=args.replan_steps,
                rng=rng,
            )

        kept = 0
        for token_ids in samples:
            token_ids = token_ids.detach().cpu().view(-1)
            seq = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True).strip().upper()
            if not is_valid_peptide(seq, args.min_len, args.max_len):
                continue
            if seq in results:
                continue

            preds = properties(token_ids)
            if not np.all(np.isfinite(preds)):
                continue

            row = {
                "sequence": seq,
                "length": len(seq),
                "reward": oracle_fn(token_ids),
                "objective": reward_from_properties(preds),
                # Reported separately: the two screens are strict and often
                # disagree, so a single combined flag hides which one bit.
                "passes_ok": int(is_peptide_ok(seq)),
                "passes_probable": int(is_peptide_probable(seq)),
                "round": round_idx,
                "predictions": [float(x) for x in preds],
            }
            results[seq] = row
            kept += 1

        console.info(
            f"  round {round_idx + 1}/{args.n_rounds}: "
            f"{len(samples)} candidates, +{kept} new valid, {len(results)} total"
        )

    if not results:
        console.warning("No valid sequences were generated.")
        return

    rows = sorted(results.values(), key=lambda r: r["reward"], reverse=True)
    if args.top_k is not None:
        rows = rows[: args.top_k]

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["sequence", "length", "reward", "objective", "passes_ok", "passes_probable", "round"]
            + label_names
        )
        for r in rows:
            writer.writerow(
                [r["sequence"], r["length"], r["reward"], r["objective"],
                 r["passes_ok"], r["passes_probable"], r["round"]]
                + r["predictions"]
            )

    n_ok = sum(r["passes_ok"] for r in rows)
    n_probable = sum(r["passes_probable"] for r in rows)
    n_both = sum(r["passes_ok"] and r["passes_probable"] for r in rows)
    console.info(f"Done. {len(rows)} sequences saved to {args.out_csv}")
    console.info(
        f"  rule screens: {n_ok} pass is_peptide_ok, {n_probable} pass is_peptide_probable, "
        f"{n_both} pass both"
    )
    console.info(f"  best reward: {rows[0]['reward']:.4f}  ({rows[0]['sequence']})")


def parse_args():
    p = argparse.ArgumentParser(
        description="Oracle-guided peptide sampling from a Hyformer checkpoint using TASAR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model_config_path",     required=True, help="Path to model config JSON")
    p.add_argument("--tokenizer_config_path", required=True, help="Path to tokenizer config JSON")
    p.add_argument("--dataset_config_path",   required=True,
                   help="Path to dataset config JSON (defines the prediction head used as oracle)")
    p.add_argument("--ckpt_path",             required=True,
                   help="Path to a checkpoint with a trained prediction head (.pt)")

    # Objective
    p.add_argument("--objective", choices=["predict", "selectivity"], default="predict",
                   help="'predict' maximizes one prediction task; 'selectivity' maximizes "
                        "Gram-negative minus Gram-positive potency (11-task MIC checkpoint only)")
    p.add_argument("--target_task", type=int, default=0,
                   help="Prediction task index to optimize when --objective predict")
    p.add_argument("--minimize", action="store_true",
                   help="Minimize the target task instead of maximizing it")
    p.add_argument("--rule_penalty", type=float, default=6.0,
                   help="Reward penalty per failed biological rule screen (0 disables the screens)")

    # TASAR search
    p.add_argument("--n_rounds",        type=int,   default=10,
                   help="Independent TASAR runs; each starts from the unmodified model")
    p.add_argument("--beam_width",      type=int,   default=50,
                   help="Beam width for one round of stochastic beam search")
    p.add_argument("--replan_steps",    type=int,   default=10,
                   help="SBS rounds per run; log-probs are reweighted after each")
    p.add_argument("--advantage_scale", type=float, default=0.25,
                   help="Scales the oracle reward into an advantage (higher = greedier search)")
    p.add_argument("--nucleus_top_p",   type=float, default=1.0,
                   help="Nucleus threshold during beam expansion (1.0 = disabled)")
    p.add_argument("--temperature",     type=float, default=1.0,
                   help="Temperature on the next-token logits")

    # Output / filtering
    p.add_argument("--out_csv",  default="tasar_sequences.csv",
                   help="Output CSV, sorted by reward (descending)")
    p.add_argument("--top_k",    type=int, default=None,
                   help="Keep only the best K sequences (default: keep all)")
    p.add_argument("--min_len",  type=int, default=8,  help="Minimum sequence length to keep")
    p.add_argument("--max_len",  type=int, default=50, help="Maximum sequence length to generate and keep")

    # Misc
    p.add_argument("--gpu",  type=int, default=0, help="GPU index to use")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    sample(args)
