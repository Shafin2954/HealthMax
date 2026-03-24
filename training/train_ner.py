"""
Fine-tune BanglaBERT for medical NER on the local silver-labeled dataset.

Expected input files:
- data/processed/ner_silver_train.jsonl
- data/processed/ner_silver_validation.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = os.getenv("NER_BASE_MODEL", "sagorsarker/bangla-bert-base")
TRAIN_DATASET_PATH = Path("data/processed/ner_silver_train.jsonl")
VALIDATION_DATASET_PATH = Path("data/processed/ner_silver_validation.jsonl")
DATASET_SUMMARY_PATH = Path("data/processed/ner_silver_summary.json")
OUTPUT_DIR = Path("models/ner-banglabert-medical")
SUMMARY_OUTPUT_PATH = Path("models/ner_training_summary.json")

LABEL_LIST = [
    "O",
    "B-SYMPTOM",
    "I-SYMPTOM",
    "B-DISEASE",
    "I-DISEASE",
    "B-MEDICINE",
    "I-MEDICINE",
]
LABEL2ID = {label: index for index, label in enumerate(LABEL_LIST)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

MAX_LENGTH = int(os.getenv("NER_MAX_LENGTH", "128"))
BATCH_SIZE = int(os.getenv("NER_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("NER_GRAD_ACCUMULATION", "2"))
NUM_EPOCHS = int(os.getenv("NER_EPOCHS", "4"))
LEARNING_RATE = float(os.getenv("NER_LEARNING_RATE", "2e-5"))


def _configure_output_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="backslashreplace")
            except ValueError:
                pass


def _prepare_transformers_backend() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TORCH", "1")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record["ner_tags"] = [LABEL2ID[tag] for tag in record["ner_tags"]]
            records.append(record)
    return records


def _load_dataset() -> DatasetDict:
    if not TRAIN_DATASET_PATH.exists() or not VALIDATION_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Silver NER dataset not found. Run `python data/build_ner_dataset.py` first."
        )

    train_records = _load_jsonl(TRAIN_DATASET_PATH)
    validation_records = _load_jsonl(VALIDATION_DATASET_PATH)

    return DatasetDict(
        {
            "train": Dataset.from_list(train_records),
            "validation": Dataset.from_list(validation_records),
        }
    )


def _tokenize_and_align_labels(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )

    aligned_labels: List[List[int]] = []
    for batch_index, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        labels_for_tokens: List[int] = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels_for_tokens.append(-100)
            elif word_id != previous_word_id:
                labels_for_tokens.append(word_labels[word_id])
            else:
                labels_for_tokens.append(-100)
            previous_word_id = word_id

        aligned_labels.append(labels_for_tokens)

    tokenized["labels"] = aligned_labels
    return tokenized


def _compute_metrics(eval_prediction: Any) -> Dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    true_predictions: List[List[str]] = []
    true_labels: List[List[str]] = []

    for prediction_row, label_row in zip(predictions, labels):
        filtered_predictions: List[str] = []
        filtered_labels: List[str] = []

        for predicted_label, gold_label in zip(prediction_row, label_row):
            if gold_label == -100:
                continue
            filtered_predictions.append(LABEL_LIST[int(predicted_label)])
            filtered_labels.append(LABEL_LIST[int(gold_label)])

        true_predictions.append(filtered_predictions)
        true_labels.append(filtered_labels)

    return {
        "precision": float(precision_score(true_labels, true_predictions)),
        "recall": float(recall_score(true_labels, true_predictions)),
        "f1": float(f1_score(true_labels, true_predictions)),
        "accuracy": float(accuracy_score(true_labels, true_predictions)),
    }


def train_ner_model() -> Dict[str, Any]:
    _prepare_transformers_backend()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    print("=" * 60)
    print("HealthMax BanglaBERT NER Fine-tuning")
    print("=" * 60)

    raw_dataset = _load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = raw_dataset.map(
        lambda batch: _tokenize_and_align_labels(batch, tokenizer),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
    )

    print(
        f"Training on {'CUDA' if torch.cuda.is_available() else 'CPU'} with "
        f"{len(raw_dataset['train'])} train / {len(raw_dataset['validation'])} validation examples..."
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    dataset_summary: Dict[str, Any] = {}
    if DATASET_SUMMARY_PATH.exists():
        with open(DATASET_SUMMARY_PATH, "r", encoding="utf-8") as file:
            dataset_summary = json.load(file)

    summary = {
        "base_model": MODEL_NAME,
        "output_dir": str(OUTPUT_DIR),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_train_examples": len(raw_dataset["train"]),
        "num_validation_examples": len(raw_dataset["validation"]),
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "eval_precision": eval_metrics.get("eval_precision"),
        "eval_recall": eval_metrics.get("eval_recall"),
        "eval_f1": eval_metrics.get("eval_f1"),
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "dataset_summary": dataset_summary,
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[OK] Model saved to {OUTPUT_DIR}")
    print(
        f"[OK] Validation F1={summary['eval_f1']:.4f}, "
        f"Precision={summary['eval_precision']:.4f}, Recall={summary['eval_recall']:.4f}"
    )
    print(f"[OK] Training summary saved to {SUMMARY_OUTPUT_PATH}")
    return summary


if __name__ == "__main__":
    _configure_output_streams()
    train_ner_model()
