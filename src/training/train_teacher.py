import argparse
import json
import os
import random

import numpy as np
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


MODEL_NAME = "roberta-base"

ID2LABEL = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an English RoBERTa teacher "
            "on XNLI."
        )
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=2490,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/roberta_teacher_10k",
    )

    return parser.parse_args()


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def select_subset(
    dataset,
    sample_size,
    seed,
):

    if sample_size <= 0:
        return dataset

    actual_size = min(
        sample_size,
        len(dataset),
    )

    return (
        dataset
        .shuffle(seed=seed)
        .select(range(actual_size))
    )


def prepare_dataset(
    dataset,
    tokenizer,
    sample_size,
    max_length,
    seed,
):

    dataset = select_subset(
        dataset=dataset,
        sample_size=sample_size,
        seed=seed,
    )

    def tokenize_batch(examples):

        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
    )

    columns_to_remove = []

    if "premise" in dataset.column_names:
        columns_to_remove.append(
            "premise"
        )

    if "hypothesis" in dataset.column_names:
        columns_to_remove.append(
            "hypothesis"
        )

    if columns_to_remove:

        dataset = dataset.remove_columns(
            columns_to_remove
        )

    if "label" in dataset.column_names:

        dataset = dataset.rename_column(
            "label",
            "labels",
        )

    return dataset


def evaluate_model(
    model,
    dataloader,
    device,
):

    model.eval()

    predictions = []
    true_labels = []

    total_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Evaluating teacher",
        )

        for batch in progress_bar:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += (
                outputs.loss.item()
            )

            predicted_labels = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            predictions.extend(
                predicted_labels
                .detach()
                .cpu()
                .tolist()
            )

            true_labels.extend(
                batch["labels"]
                .detach()
                .cpu()
                .tolist()
            )

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    print(
        f"\nTeacher validation loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"Teacher validation accuracy: "
        f"{accuracy:.4f}"
    )

    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    total_epochs,
):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=(
            f"Teacher Epoch "
            f"{epoch}/{total_epochs}"
        ),
    )

    for batch in progress_bar:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss / len(dataloader)
    )

    return average_loss


def save_results(
    output_dir,
    results,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    path = os.path.join(
        output_dir,
        "results.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nResults saved to: {path}"
    )


def main():

    args = parse_args()

    set_seed(
        args.seed
    )

    print(
        "\n========================================"
    )

    print(
        "LOWRESOURCEX - ENGLISH TEACHER"
    )

    print(
        "========================================"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Training samples: "
        f"{args.train_samples}"
    )

    print(
        f"Validation samples: "
        f"{args.validation_samples}"
    )

    print(
        f"Batch size: "
        f"{args.batch_size}"
    )

    print(
        f"Epochs: "
        f"{args.epochs}"
    )

    print(
        f"Learning rate: "
        f"{args.learning_rate}"
    )

    print(
        f"Seed: {args.seed}"
    )

    device = get_device()

    print(
        "\nUsing device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "\nLoading RoBERTa tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "\nLoading English XNLI..."
    )

    dataset = load_dataset(
        "facebook/xnli",
        "en",
    )

    print(
        "English training examples:",
        len(dataset["train"]),
    )

    print(
        "English validation examples:",
        len(dataset["validation"]),
    )

    print(
        "\nPreparing training data..."
    )

    train_dataset = prepare_dataset(
        dataset=dataset["train"],
        tokenizer=tokenizer,
        sample_size=args.train_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(
        "Preparing validation data..."
    )

    validation_dataset = (
        prepare_dataset(
            dataset=dataset["validation"],
            tokenizer=tokenizer,
            sample_size=(
                args.validation_samples
            ),
            max_length=args.max_length,
            seed=args.seed,
        )
    )

    print(
        "\nPrepared training size:",
        len(train_dataset),
    )

    print(
        "Prepared validation size:",
        len(validation_dataset),
    )

    data_collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer,
            return_tensors="pt",
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        generator=generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "\nLoading RoBERTa teacher model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    model.to(
        device
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=(
            args.weight_decay
        ),
    )

    print(
        "\n========================================"
    )

    print(
        "BEFORE TRAINING"
    )

    print(
        "========================================"
    )

    before_training = evaluate_model(
        model=model,
        dataloader=validation_loader,
        device=device,
    )

    history = []

    best_accuracy = -1.0

    print(
        "\n========================================"
    )

    print(
        "TRAINING TEACHER"
    )

    print(
        "========================================"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        training_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        print(
            f"\nEpoch {epoch} "
            f"average training loss: "
            f"{training_loss:.4f}"
        )

        validation_results = (
            evaluate_model(
                model=model,
                dataloader=validation_loader,
                device=device,
            )
        )

        epoch_results = {
            "epoch": epoch,
            "training_loss": float(
                training_loss
            ),
            "validation_loss": (
                validation_results["loss"]
            ),
            "validation_accuracy": (
                validation_results[
                    "accuracy"
                ]
            ),
        }

        history.append(
            epoch_results
        )

        if (
            validation_results[
                "accuracy"
            ]
            > best_accuracy
        ):

            best_accuracy = (
                validation_results[
                    "accuracy"
                ]
            )

            print(
                "\nNew best teacher."
            )

            print(
                f"Saving to "
                f"{args.output_dir}"
            )

            os.makedirs(
                args.output_dir,
                exist_ok=True,
            )

            model.save_pretrained(
                args.output_dir
            )

            tokenizer.save_pretrained(
                args.output_dir
            )

    experiment_results = {

        "model": MODEL_NAME,

        "configuration": {
            "train_samples": (
                args.train_samples
            ),
            "validation_samples": (
                args.validation_samples
            ),
            "batch_size": (
                args.batch_size
            ),
            "epochs": (
                args.epochs
            ),
            "learning_rate": (
                args.learning_rate
            ),
            "weight_decay": (
                args.weight_decay
            ),
            "max_length": (
                args.max_length
            ),
            "seed": args.seed,
        },

        "before_training": (
            before_training
        ),

        "history": history,

        "best_validation_accuracy": (
            best_accuracy
        ),
    }

    save_results(
        output_dir=args.output_dir,
        results=experiment_results,
    )

    print(
        "\n========================================"
    )

    print(
        "TEACHER TRAINING COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Best English validation accuracy: "
        f"{best_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()