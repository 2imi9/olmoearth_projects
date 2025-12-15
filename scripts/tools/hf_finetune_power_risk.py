"""Example script for fine-tuning a text model on power & climate datasets.
This script loads multiple Hugging Face datasets, concatenates them, and
fine-tunes a transformer model for regression (default) or classification.
It also demonstrates pushing the resulting model back to the Hugging Face Hub.
Example usage (regression target):
    uv run python scripts/tools/hf_finetune_power_risk.py \
        --model-name allenai/OLMo-2B-LM \
        --text-column description \
        --label-column target_value \
        --push-to-hub-id YOUR_USERNAME/power-risk-olmo \
        --num-train-epochs 5
Assumptions:
- The datasets expose a text-like field that can be tokenized. Use
  ``--text-column`` to point at it.
- The label column contains either floats (regression) or integer class
  ids (classification). Use ``--task-type`` to control the head that is
  attached to the model.
Required dependencies (installable with ``uv add`` or ``pip install``):
    datasets, evaluate, numpy, transformers
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.features import Value
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_DATASETS = [
    "2imi9/us-power-climate-data-streamflow",
    "2imi9/us-power-climate-data-climate",
    "2imi9/us-power-climate-data-plants",
]


@dataclass
class ScriptArgs:
    """Arguments used for the fine-tuning script."""

    model_name: str
    datasets: List[str]
    text_column: str
    label_column: str
    task_type: str
    max_length: int
    learning_rate: float
    batch_size: int
    num_train_epochs: int
    weight_decay: float
    validation_split: float
    push_to_hub_id: Optional[str]
    hub_private_repo: bool


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        type=str,
        default="allenai/OLMo-1B",
        help="Base model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="One or more dataset repository IDs to merge for training.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the column containing the text to tokenize.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--task-type",
        choices=["regression", "classification"],
        default="regression",
        help="Head to attach to the model.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay to apply.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of the concatenated dataset to reserve for validation when no explicit split exists.",
    )
    parser.add_argument(
        "--push-to-hub-id",
        type=str,
        default=None,
        help=(
            "Hugging Face repo name to push the fine-tuned model to. "
            "If omitted, the model will only be saved locally."
        ),
    )
    parser.add_argument(
        "--hub-private-repo",
        action="store_true",
        help="If set, push the model to a private repo on the Hugging Face Hub.",
    )

    args = parser.parse_args()
    return ScriptArgs(
        model_name=args.model_name,
        datasets=args.datasets,
        text_column=args.text_column,
        label_column=args.label_column,
        task_type=args.task_type,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        validation_split=args.validation_split,
        push_to_hub_id=args.push_to_hub_id,
        hub_private_repo=args.hub_private_repo,
    )


def load_and_concatenate(
    dataset_names: Iterable[str], label_column: str, task_type: str
) -> Dict[str, Optional[Dataset]]:
    """Download datasets from the Hub and concatenate their splits."""

    train_splits = []
    eval_splits = []
    for name in dataset_names:
        loaded = load_dataset(name)
        label_type = Value("float32") if task_type == "regression" else Value("int64")

        if "train" in loaded:
            train_splits.append(loaded["train"].cast_column(label_column, label_type))
        else:
            # If no explicit train split, take the first available split.
            first_split = next(iter(loaded.values()))
            train_splits.append(first_split.cast_column(label_column, label_type))

        # Prefer validation > test as eval data if available.
        eval_split = None
        if "validation" in loaded:
            eval_split = loaded["validation"]
        elif "test" in loaded:
            eval_split = loaded["test"]

        if eval_split is not None:
            eval_splits.append(eval_split.cast_column(label_column, label_type))

    train_dataset = concatenate_datasets(train_splits)
    eval_dataset = concatenate_datasets(eval_splits) if eval_splits else None
    return {"train": train_dataset, "validation": eval_dataset}


def split_dataset(dataset: Mapping[str, Optional[Dataset]], validation_split: float) -> Mapping[str, Dataset]:
    if dataset["validation"] is not None:
        return {"train": dataset["train"], "validation": dataset["validation"]}

    split = dataset["train"].train_test_split(test_size=validation_split, seed=42)
    return {"train": split["train"], "validation": split["test"]}


def tokenize_dataset(
    datasets: Mapping[str, Dataset],
    tokenizer: AutoTokenizer,
    text_column: str,
    label_column: str,
    max_length: int,
) -> Mapping[str, Dataset]:
    def preprocess_function(examples: MutableMapping[str, List[str]]):
        tokenized = tokenizer(
            examples[text_column],
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        tokenized["labels"] = examples[label_column]
        return tokenized

    return {
        split: data.map(preprocess_function, batched=True, remove_columns=data.column_names)
        for split, data in datasets.items()
    }


def compute_regression_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze()
    mse = ((predictions - labels) ** 2).mean()
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(predictions - labels).mean()),
    }


def compute_classification_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)


def main() -> None:
    args = parse_args()

    raw_dataset = load_and_concatenate(args.datasets, args.label_column, args.task_type)
    dataset_splits = split_dataset(raw_dataset, args.validation_split)

    label_values = set(dataset_splits["train"][args.label_column])
    if dataset_splits["validation"] is not None:
        label_values.update(dataset_splits["validation"][args.label_column])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    processed = tokenize_dataset(dataset_splits, tokenizer, args.text_column, args.label_column, args.max_length)

    num_labels = 1 if args.task_type == "regression" else len(label_values)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="regression" if args.task_type == "regression" else None,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    metric = None
    compute_metrics = None
    if args.task_type == "regression":
        compute_metrics = compute_regression_metrics
    else:
        metric = evaluate.load("accuracy")
        compute_metrics = lambda p: compute_classification_metrics(p, metric)  # noqa: E731

    hub_model_id = args.push_to_hub_id
    training_args = TrainingArguments(
        output_dir="./power-risk-model",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse" if args.task_type == "regression" else "accuracy",
        greater_is_better=args.task_type != "regression",
        push_to_hub=hub_model_id is not None,
        hub_model_id=hub_model_id,
        hub_private_repo=args.hub_private_repo,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    if hub_model_id:
        trainer.push_to_hub()
    else:
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()