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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# ============================================================
# Constants
# ============================================================

MODEL_NAME = "bert-base-multilingual-cased"

LABEL_NAMES = [
    "entailment",
    "neutral",
    "contradiction",
]

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


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train multilingual BERT on English XNLI "
            "and evaluate zero-shot transfer to Hindi."
        )
    )

    parser.add_argument(
        "--train-samples",
        type=int,
        default=2000,
        help=(
            "Number of English training examples. "
            "Use 0 to use the entire training set."
        ),
    )

    parser.add_argument(
        "--validation-samples",
        type=int,
        default=500,
        help=(
            "Number of validation examples per language. "
            "Use 0 to use the complete validation set."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size used for training and evaluation.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum tokenizer sequence length.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/mbert_xnli_baseline",
        help="Directory where model and metrics are saved.",
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ============================================================
# Dataset utilities
# ============================================================

def select_subset(
    dataset,
    sample_size,
    seed,
):

    # sample_size <= 0 means:
    # use the full split.
    if sample_size <= 0:
        return dataset

    actual_size = min(
        sample_size,
        len(dataset),
    )

    if sample_size > len(dataset):

        print(
            f"Requested {sample_size} examples, "
            f"but dataset contains only {len(dataset)}."
        )

        print(
            f"Using all {len(dataset)} examples instead."
        )

    return (
        dataset
        .shuffle(seed=seed)
        .select(range(actual_size))
    )


def tokenize_dataset(
    dataset,
    tokenizer,
    max_length,
):

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

    # Raw text is no longer required by the model.
    columns_to_remove = []

    if "premise" in dataset.column_names:
        columns_to_remove.append("premise")

    if "hypothesis" in dataset.column_names:
        columns_to_remove.append("hypothesis")

    if columns_to_remove:

        dataset = dataset.remove_columns(
            columns_to_remove
        )

    # Hugging Face models expect the target column
    # to be named "labels".
    if "label" in dataset.column_names:

        dataset = dataset.rename_column(
            "label",
            "labels",
        )

    return dataset


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

    dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    return dataset


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    dataloader,
    device,
    language_name,
):

    model.eval()

    predictions = []
    true_labels = []

    total_loss = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluating {language_name}",
        )

        for batch in progress_bar:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            total_loss += outputs.loss.item()

            batch_predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            predictions.extend(
                batch_predictions
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
        f"\n{language_name} Loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"{language_name} Accuracy: "
        f"{accuracy:.4f}"
    )

    return {
        "loss": float(average_loss),
        "accuracy": float(accuracy),
    }


# ============================================================
# Training
# ============================================================

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
            f"Epoch {epoch}/{total_epochs}"
        ),
    )

    for batch in progress_bar:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        # ---------------------------------------
        # Clear gradients from previous step
        # ---------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ---------------------------------------
        # Forward pass
        # ---------------------------------------

        outputs = model(**batch)

        loss = outputs.loss

        # ---------------------------------------
        # Backward pass
        # ---------------------------------------

        loss.backward()

        # ---------------------------------------
        # Gradient clipping
        # ---------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        # ---------------------------------------
        # Update model parameters
        # ---------------------------------------

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss / len(dataloader)
    )

    return average_loss


# ============================================================
# Save experiment information
# ============================================================

def save_metrics(
    output_dir,
    experiment_data,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    metrics_path = os.path.join(
        output_dir,
        "results.json",
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            experiment_data,
            file,
            indent=4,
        )

    print(
        f"\nMetrics saved to: "
        f"{metrics_path}"
    )


# ============================================================
# Main experiment
# ============================================================

def main():

    args = parse_args()

    set_seed(args.seed)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\n========================================")
    print("LOWRESOURCEX - mBERT BASELINE")
    print("========================================")

    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {args.train_samples}")
    print(
        f"Validation samples: "
        f"{args.validation_samples}"
    )
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(
        f"Learning rate: "
        f"{args.learning_rate}"
    )
    print(f"Weight decay: {args.weight_decay}")
    print(f"Maximum length: {args.max_length}")
    print(f"Seed: {args.seed}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("\nUsing device:")
    print(device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Load raw datasets
    # --------------------------------------------------------

    print("\nLoading English XNLI...")

    english_dataset = load_dataset(
        "facebook/xnli",
        "en",
    )

    print("Loading Hindi XNLI...")

    hindi_dataset = load_dataset(
        "facebook/xnli",
        "hi",
    )

    print("\nRaw dataset sizes:")

    print(
        "English train:",
        len(english_dataset["train"]),
    )

    print(
        "English validation:",
        len(english_dataset["validation"]),
    )

    print(
        "Hindi validation:",
        len(hindi_dataset["validation"]),
    )

    # --------------------------------------------------------
    # Prepare datasets
    # --------------------------------------------------------

    print("\nPreparing English training data...")

    english_train = prepare_dataset(
        dataset=english_dataset["train"],
        tokenizer=tokenizer,
        sample_size=args.train_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(
        "Preparing English validation data..."
    )

    english_validation = prepare_dataset(
        dataset=english_dataset["validation"],
        tokenizer=tokenizer,
        sample_size=args.validation_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    print(
        "Preparing Hindi validation data..."
    )

    hindi_validation = prepare_dataset(
        dataset=hindi_dataset["validation"],
        tokenizer=tokenizer,
        sample_size=args.validation_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    print("\nPrepared dataset sizes:")

    print(
        "English train:",
        len(english_train),
    )

    print(
        "English validation:",
        len(english_validation),
    )

    print(
        "Hindi validation:",
        len(hindi_validation),
    )

    # --------------------------------------------------------
    # Dynamic padding
    # --------------------------------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_generator = torch.Generator()

    train_generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        english_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        generator=train_generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    english_validation_loader = DataLoader(
        english_validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    hindi_validation_loader = DataLoader(
        hindi_validation,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading mBERT model...")

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    model.to(device)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Evaluation before training
    # --------------------------------------------------------

    print("\n========================================")
    print("BEFORE TRAINING")
    print("========================================")

    before_english = evaluate_model(
        model=model,
        dataloader=english_validation_loader,
        device=device,
        language_name="English",
    )

    before_hindi = evaluate_model(
        model=model,
        dataloader=hindi_validation_loader,
        device=device,
        language_name="Hindi",
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_english_accuracy = -1.0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING")
    print("========================================")

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

        # --------------------------------------------
        # Evaluate after every epoch
        # --------------------------------------------

        english_results = evaluate_model(
            model=model,
            dataloader=english_validation_loader,
            device=device,
            language_name="English",
        )

        hindi_results = evaluate_model(
            model=model,
            dataloader=hindi_validation_loader,
            device=device,
            language_name="Hindi",
        )

        transfer_gap = (
            english_results["accuracy"]
            - hindi_results["accuracy"]
        )

        epoch_results = {
            "epoch": epoch,
            "training_loss": float(
                training_loss
            ),
            "english_loss": (
                english_results["loss"]
            ),
            "english_accuracy": (
                english_results["accuracy"]
            ),
            "hindi_loss": (
                hindi_results["loss"]
            ),
            "hindi_accuracy": (
                hindi_results["accuracy"]
            ),
            "transfer_gap": float(
                transfer_gap
            ),
        }

        history.append(
            epoch_results
        )

        print(
            "\nCross-lingual transfer gap: "
            f"{transfer_gap:.4f}"
        )

        # --------------------------------------------
        # Save the model selected by ENGLISH
        # validation accuracy.
        #
        # We deliberately do NOT select using Hindi
        # accuracy because Hindi is our target
        # language evaluation.
        # --------------------------------------------

        if (
            english_results["accuracy"]
            > best_english_accuracy
        ):

            best_english_accuracy = (
                english_results["accuracy"]
            )

            os.makedirs(
                args.output_dir,
                exist_ok=True,
            )

            print(
                "\nNew best English "
                "validation accuracy."
            )

            print(
                f"Saving checkpoint to "
                f"{args.output_dir}"
            )

            model.save_pretrained(
                args.output_dir
            )

            tokenizer.save_pretrained(
                args.output_dir
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    final_results = history[-1]

    print("\n========================================")
    print("FINAL RESULTS")
    print("========================================")

    print(
        "English accuracy: "
        f"{final_results['english_accuracy']:.4f}"
    )

    print(
        "Hindi accuracy: "
        f"{final_results['hindi_accuracy']:.4f}"
    )

    print(
        "Cross-lingual transfer gap: "
        f"{final_results['transfer_gap']:.4f}"
    )

    print(
        "Best English validation accuracy: "
        f"{best_english_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    experiment_data = {

        "model_name": MODEL_NAME,

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
            "seed": (
                args.seed
            ),
        },

        "before_training": {
            "english": before_english,
            "hindi": before_hindi,
        },

        "history": history,

        "best_english_accuracy": (
            best_english_accuracy
        ),
    }

    save_metrics(
        output_dir=args.output_dir,
        experiment_data=experiment_data,
    )

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()