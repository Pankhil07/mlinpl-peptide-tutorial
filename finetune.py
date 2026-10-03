#!/usr/bin/env python3
"""
Fine-tune a pre-trained Hyformer checkpoint for sequence property prediction
(e.g. MIC, activity, any regression/classification target) from a CSV file.

Example
-------
python finetune.py \
    --train_csv          my_dataset.csv \
    --sequence_column    sequence \
    --target_column      log2_mic \
    --task_type          regression \
    --ckpt_path          checkpoints/ckpt.pt \
    --out_dir            checkpoints/finetuned

The training CSV needs at least two columns: one with peptide sequences and
one with the numeric target. If --val_csv is not given, a held-out split is
carved out of --train_csv automatically.

The resulting checkpoint (out_dir/ckpt.pt) is compatible with predict.py /
predict_mic_ab.py-style prediction scripts.
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from hyformer.configs.model import ModelConfig
from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.trainer import TrainerConfig
from hyformer.models.auto import AutoModel
from hyformer.trainers.trainer import Trainer
from hyformer.utils.datasets.sequence import SequenceDataset
from hyformer.utils.reproducibility import set_seed
from hyformer.utils.tokenizers.auto import AutoTokenizer

console = logging.getLogger("finetune")
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.set_float32_matmul_precision("high")

_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ------------------------------------------------------------------ #
# Defaults (relative to this script's directory)
# ------------------------------------------------------------------ #
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_CFG   = os.path.join(_HERE, "configs", "models",   "hyformer", "config.json")
_DEFAULT_TOKENIZER_CFG = os.path.join(_HERE, "configs", "tokenizers", "amino_acid", "config.json")
_DEFAULT_TRAINER_CFG  = os.path.join(_HERE, "configs", "trainers", "finetune", "config.json")


def load_csv(path: str, sequence_column: str, target_column: str):
    df = pd.read_csv(path)
    for col in (sequence_column, target_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Available columns: {list(df.columns)}")

    df = df[[sequence_column, target_column]].dropna()
    df[sequence_column] = df[sequence_column].astype(str).str.strip().str.upper()

    valid_mask = df[sequence_column].apply(lambda s: bool(s) and all(c in _VALID_AA for c in s))
    n_invalid = (~valid_mask).sum()
    if n_invalid:
        console.info(f"  Skipping {n_invalid} row(s) with non-standard amino acid characters")
    df = df[valid_mask]

    sequences = df[sequence_column].to_numpy()
    targets = df[target_column].to_numpy()
    return sequences, targets


def to_dataset(sequences, targets, task_type: str) -> SequenceDataset:
    dtype = np.int64 if task_type == "classification" else np.float32
    target = np.asarray(targets, dtype=dtype).reshape(-1, 1)
    return SequenceDataset(
        data=np.asarray(sequences),
        target=target,
        prediction_task_type=task_type,
        num_prediction_tasks=1,
        test_metric="roc_auc" if task_type == "classification" else "rmse",
    )


def finetune(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    console.info(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # --- data ---
    console.info(f"Loading training data: {args.train_csv}")
    train_seqs, train_targets = load_csv(args.train_csv, args.sequence_column, args.target_column)

    if args.val_csv:
        console.info(f"Loading validation data: {args.val_csv}")
        val_seqs, val_targets = load_csv(args.val_csv, args.sequence_column, args.target_column)
    else:
        console.info(f"No --val_csv given, splitting off {args.val_split:.0%} of --train_csv for validation")
        stratify = train_targets if args.task_type == "classification" else None
        train_seqs, val_seqs, train_targets, val_targets = train_test_split(
            train_seqs, train_targets,
            test_size=args.val_split, random_state=args.seed, stratify=stratify,
        )
    console.info(f"Train: {len(train_seqs)} sequences | Val: {len(val_seqs)} sequences")

    train_dataset = to_dataset(train_seqs, train_targets, args.task_type)
    val_dataset = to_dataset(val_seqs, val_targets, args.task_type)

    # --- tokenizer & model ---
    tokenizer = AutoTokenizer.from_config(TokenizerConfig.from_config_filepath(args.tokenizer_config_path))
    model = AutoModel.from_config(
        ModelConfig.from_config_filepath(args.model_config_path),
        prediction_task_type=args.task_type,
        num_prediction_tasks=1,
    )
    assert len(tokenizer) == model.vocab_size, (
        f"Tokenizer vocab size {len(tokenizer)} does not match model vocab size {model.vocab_size}"
    )

    # --- trainer ---
    trainer_config = TrainerConfig.from_config_filepath(args.trainer_config_path)
    if args.max_epochs is not None:
        trainer_config.max_epochs = args.max_epochs
    if args.learning_rate is not None:
        trainer_config.learning_rate = args.learning_rate
    if args.batch_size is not None:
        trainer_config.batch_size = args.batch_size

    trainer = Trainer.from_config(
        config=trainer_config,
        model=model,
        device=device,
        tokenizer=tokenizer,
        out_dir=args.out_dir,
        logger=None,
        worker_seed=args.seed,
    )

    if args.ckpt_path:
        console.info(f"Loading pre-trained backbone from {args.ckpt_path}")
        trainer.resume_from_checkpoint(args.ckpt_path, resume_training=False, discard_prediction_head=True)
    else:
        console.info("No --ckpt_path given, fine-tuning a randomly initialized model from scratch")

    console.info(f"Tasks: {trainer_config.tasks}")
    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        task_specific_validation="prediction" if "prediction" in trainer_config.tasks else None,
        patience=args.patience,
    )

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    console.info(f"Done. Best checkpoint saved to {ckpt_path}")

    if args.test_csv:
        console.info(f"Loading test data: {args.test_csv}")
        test_seqs, test_targets = load_csv(args.test_csv, args.sequence_column, args.target_column)
        test_dataset = to_dataset(test_seqs, test_targets, args.task_type)

        trainer.resume_from_checkpoint(ckpt_path, resume_training=False, discard_prediction_head=False)
        metric_name = "rmse" if args.task_type == "regression" else "roc_auc"
        metric_value = trainer.test(test_dataset, metric=metric_name)
        console.info(f"Test {metric_name}: {metric_value:.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Hyformer for sequence property prediction.")

    # Data
    p.add_argument("--train_csv", required=True, help="CSV file with sequence + target columns")
    p.add_argument("--val_csv", default=None, help="Optional validation CSV (else split off --train_csv)")
    p.add_argument("--test_csv", default=None, help="Optional test CSV, evaluated after training")
    p.add_argument("--sequence_column", default="sequence")
    p.add_argument("--target_column", default="target")
    p.add_argument("--val_split", type=float, default=0.1, help="Validation fraction if --val_csv not given")
    p.add_argument("--task_type", choices=["regression", "classification"], default="regression")

    # Model / checkpoint
    p.add_argument("--ckpt_path", default=None, help="Pre-trained checkpoint to fine-tune (backbone only)")
    p.add_argument("--model_config_path", default=_DEFAULT_MODEL_CFG)
    p.add_argument("--tokenizer_config_path", default=_DEFAULT_TOKENIZER_CFG)
    p.add_argument("--trainer_config_path", default=_DEFAULT_TRAINER_CFG)

    # Output
    p.add_argument("--out_dir", required=True, help="Directory to save the fine-tuned checkpoint")

    # Trainer overrides (optional, else taken from --trainer_config_path)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs)")

    # Misc
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    finetune(args)
