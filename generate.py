#!/usr/bin/env python3
"""
Generate peptide sequences from a Hyformer checkpoint.

Example
-------
python generate.py \
    --model_config_path     configs/models/hyformer/config.json \
    --tokenizer_config_path configs/tokenizers/amino_acid/config.json \
    --ckpt_path             /path/to/ckpt.pt \
    --n_sequences           10000 \
    --out_csv               generated.csv
"""

import argparse
import csv
import logging
import os
import sys

import torch

from hyformer.configs.model import ModelConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.models.auto import AutoModel
from hyformer.utils.tokenizers.auto import AutoTokenizer

console = logging.getLogger("generate")
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.set_float32_matmul_precision("high")

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def is_valid(seq: str, min_len: int, max_len: int) -> bool:
    return bool(seq) and min_len <= len(seq) <= max_len and all(c in _VALID_AA for c in seq)


def generate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    console.info(f"Device: {device}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_config(
        TokenizerConfig.from_config_filepath(args.tokenizer_config_path)
    )
    model = AutoModel.from_config(
        ModelConfig.from_config_filepath(args.model_config_path)
    )
    console.info(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)  # handle both wrapped checkpoints and bare state dicts
    model.load_pretrained(state_dict=state, discard_prediction_head=True)
    model.to(device).eval()

    lm_token_id = tokenizer.task_token_id("lm")
    bos_id      = tokenizer.bos_token_id
    eos_id      = tokenizer.eos_token_id
    pad_id      = tokenizer.pad_token_id

    # Fixed prefix: [<lm>, <s>]
    prefix = torch.tensor([[lm_token_id, bos_id]], dtype=torch.long, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    seen = set()
    written = 0

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "length"])

        while written < args.n_sequences:
            batch_size = min(args.batch_size, args.n_sequences - written)
            batched_prefix = prefix.expand(batch_size, -1)

            with torch.inference_mode():
                gen_ids = model.generate(
                    prefix_input_ids=batched_prefix,
                    num_tokens_to_generate=args.max_len,
                    eos_token_id=eos_id,
                    pad_token_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    use_cache=False,
                )

            seqs = [
                tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip().upper()
                for ids in gen_ids
            ]

            rows = []
            for seq in seqs:
                if not is_valid(seq, args.min_len, args.max_len):
                    continue
                if args.deduplicate and seq in seen:
                    continue
                seen.add(seq)
                rows.append([seq, len(seq)])

            if rows:
                writer.writerows(rows)
                f.flush()
                written += len(rows)
                console.info(f"  {written} / {args.n_sequences}")

            del gen_ids
            torch.cuda.empty_cache()

    console.info(f"Done. {written} sequences saved to {args.out_csv}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate peptide sequences from a Hyformer checkpoint.")

    # Required
    p.add_argument("--model_config_path",     required=True,
                   help="Path to model config JSON")
    p.add_argument("--tokenizer_config_path", required=True,
                   help="Path to tokenizer config JSON")
    p.add_argument("--ckpt_path",             required=True,
                   help="Path to checkpoint (.pt)")

    # Output
    p.add_argument("--out_csv",       default="generated.csv",
                   help="Output CSV file (columns: sequence, length)")
    p.add_argument("--n_sequences",   type=int, default=10_000,
                   help="Number of valid sequences to generate")

    # Sampling
    p.add_argument("--temperature",   type=float, default=1.0,
                   help="Sampling temperature (lower = more conservative)")
    p.add_argument("--top_k",         type=int,   default=None,
                   help="Top-k filtering (None = disabled)")
    p.add_argument("--top_p",         type=float, default=None,
                   help="Nucleus sampling threshold (None = disabled)")

    # Filtering
    p.add_argument("--min_len",       type=int,   default=5,
                   help="Minimum sequence length to keep")
    p.add_argument("--max_len",       type=int,   default=50,
                   help="Maximum sequence length to generate and keep")
    p.add_argument("--deduplicate",   default=True,
                   action=argparse.BooleanOptionalAction,
                   help="Remove duplicate sequences (default: True)")

    # Misc
    p.add_argument("--batch_size",    type=int,   default=512)
    p.add_argument("--gpu",           type=int,   default=0,
                   help="GPU index to use")
    p.add_argument("--seed",          type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    generate(args)
