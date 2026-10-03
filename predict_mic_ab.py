#!/usr/bin/env python3
"""
Predict log2_MIC_uM (Acinetobacter baumannii) for a list of peptide sequences
using the fine-tuned Hyformer checkpoint.

Usage
-----
# From a text file (one sequence per line):
python predict_mic_ab.py --sequences sequences.txt --ckpt_path ckpt.pt --out_csv predictions.csv

# Inline sequences:
python predict_mic_ab.py --sequences KWKLFKKIEK GIWDTIKSMG --ckpt_path ckpt.pt

# All config paths default to the bundled configs in this repo.
# Override if needed:
python predict_mic_ab.py --sequences sequences.txt \
    --ckpt_path ckpt.pt \
    --model_config_path configs/models/hyformer/config.json \
    --tokenizer_config_path configs/tokenizers/amino_acid/config.json

Checkpoint download
-------------------
Download ckpt.pt from Google Drive and pass its local path via --ckpt_path.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

from hyformer.configs.dataset import DatasetConfig
from hyformer.configs.model import ModelConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.models.auto import AutoModel
from hyformer.utils.tokenizers.auto import AutoTokenizer

# ------------------------------------------------------------------ #
# Defaults (relative to this script's directory)
# ------------------------------------------------------------------ #
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_CFG     = os.path.join(_HERE, "configs", "models",     "hyformer",    "config.json")
_DEFAULT_TOKENIZER_CFG = os.path.join(_HERE, "configs", "tokenizers", "amino_acid",  "config.json")
_DEFAULT_DATASET_CFG   = os.path.join(_HERE, "configs", "datasets",   "ab_mic",      "config.json")

VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


# ------------------------------------------------------------------ #
# helpers
# ------------------------------------------------------------------ #
def is_valid_peptide(seq: str, min_len: int = 2, max_len: int = 200) -> bool:
    return bool(seq) and (min_len <= len(seq) <= max_len) and all(c in VALID_AAS for c in seq)


def load_model(
    model_cfg_path: str,
    tokenizer_cfg_path: str,
    dataset_cfg_path: str,
    ckpt_path: str,
    device: torch.device,
):
    tok_cfg  = TokenizerConfig.from_config_filepath(tokenizer_cfg_path)
    mdl_cfg  = ModelConfig.from_config_filepath(model_cfg_path)
    dset_cfg = DatasetConfig.from_config_filepath(dataset_cfg_path)

    tokenizer = AutoTokenizer.from_config(tok_cfg)
    model = AutoModel.from_config(
        mdl_cfg,
        prediction_task_type=dset_cfg.prediction_task_type,
        num_prediction_tasks=dset_cfg.num_prediction_tasks,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)            # handle both wrapped and bare state dicts
    model.load_pretrained(state_dict=state, discard_prediction_head=False)
    model.to(device).eval()
    return model, tokenizer


def to_prediction_inputs(gen_ids: torch.Tensor, pred_task_token_id: int) -> torch.Tensor:
    pred_ids = gen_ids.clone()
    pred_ids[:, 0] = pred_task_token_id
    return pred_ids


@torch.inference_mode()
def predict_no_dropout(model, input_ids, attention_mask):
    was_training = model.training
    model.eval()
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task="prediction",
        return_loss=False,
    )
    mean = out["logits"].detach().cpu().numpy()
    std = np.zeros_like(mean, dtype=np.float32)
    model.train(was_training)
    return mean, std


@torch.inference_mode()
def predict_batch(
    sequences: list,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return predicted log2_MIC_uM for each sequence (shape: [N])."""
    pred_token_id = tokenizer.task_token_id("prediction")
    pad_id = tokenizer.pad_token_id

    all_preds = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]

        # tokenise with lm task to get standard token ids (list of lists)
        enc = tokenizer(batch_seqs, task="prediction")
        ids_list = enc["input_ids"]

        # pad to max length in batch
        max_len = max(len(ids) for ids in ids_list)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in ids_list]

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # swap first token to prediction task token, build mask
        pred_ids  = to_prediction_inputs(input_ids, pred_token_id)
        attn_mask = (pred_ids != pad_id).long()

        mean_pred, _ = predict_no_dropout(model, pred_ids, attn_mask)
        preds = mean_pred[:, 0]            # single task → shape (batch,)
        all_preds.append(preds)

    return np.concatenate(all_preds, axis=0)


# ------------------------------------------------------------------ #
# main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Predict log2_MIC_uM (A. baumannii) for peptide sequences."
    )
    parser.add_argument(
        "--sequences", nargs="+", required=True,
        help="Peptide sequences (uppercase AA strings) OR a path to a .txt file "
             "with one sequence per line.",
    )
    parser.add_argument(
        "--ckpt_path", required=True,
        help="Path to the fine-tuned checkpoint (ckpt.pt). "
             "Download from the link provided in the README.",
    )
    parser.add_argument("--model_config_path",     default=_DEFAULT_MODEL_CFG)
    parser.add_argument("--tokenizer_config_path", default=_DEFAULT_TOKENIZER_CFG)
    parser.add_argument("--dataset_config_path",   default=_DEFAULT_DATASET_CFG)
    parser.add_argument("--out_csv",  default=None,
                        help="Write results to this CSV file (optional).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on.")
    args = parser.parse_args()

    # --- resolve device ---
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Using device: {device}")

    # --- resolve sequences ---
    raw_seqs = args.sequences
    if len(raw_seqs) == 1 and os.path.isfile(raw_seqs[0]):
        with open(raw_seqs[0]) as f:
            raw_seqs = [line.strip().upper() for line in f if line.strip()]
        print(f"[*] Loaded {len(raw_seqs)} sequences from {args.sequences[0]}")
    else:
        raw_seqs = [s.strip().upper() for s in raw_seqs]

    # --- validate ---
    valid_seqs, invalid = [], []
    for s in raw_seqs:
        if is_valid_peptide(s):
            valid_seqs.append(s)
        else:
            invalid.append(s)
    if invalid:
        print(f"[!] Skipping {len(invalid)} invalid sequence(s): {invalid[:5]}")
    if not valid_seqs:
        print("[!] No valid sequences to predict. Exiting.")
        sys.exit(1)

    # --- load model ---
    print("[*] Loading model...")
    model, tokenizer = load_model(
        args.model_config_path,
        args.tokenizer_config_path,
        args.dataset_config_path,
        args.ckpt_path,
        device,
    )

    # --- predict ---
    print(f"[*] Predicting {len(valid_seqs)} sequences...")
    log2mic_preds = predict_batch(valid_seqs, model, tokenizer, device, args.batch_size)

    # log2_MIC_uM -> MIC_uM
    mic_um_preds = 2.0 ** log2mic_preds

    # --- print results ---
    header = f"{'Sequence':<40}  {'log2_MIC_uM':>14}  {'MIC_uM':>10}"
    print("\n" + header)
    print("-" * len(header))
    for seq, l2, mic in zip(valid_seqs, log2mic_preds, mic_um_preds):
        print(f"{seq:<40}  {l2:>14.4f}  {mic:>10.4f}")

    # --- optional CSV ---
    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sequence", "log2_MIC_uM", "MIC_uM"])
            for seq, l2, mic in zip(valid_seqs, log2mic_preds, mic_um_preds):
                w.writerow([seq, round(float(l2), 6), round(float(mic), 6)])
        print(f"\n[+] Results written to {args.out_csv}")


if __name__ == "__main__":
    main()
