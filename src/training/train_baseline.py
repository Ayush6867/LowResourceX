import os

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


MODEL_NAME = "bert-base-multilingual-cased"

MAX_LENGTH = 128

TRAIN_SAMPLES = 2000
VALIDATION_SAMPLES = 500

BATCH_SIZE = 4
LEARNING_RATE = 2e-5
EPOCHS = 1

SEED = 42

OUTPUT_DIR = "models/mbert_xnli_baseline"


LABEL_NAMES = [
    "entailment",
    "neutral",
    "contradiction",
]


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def tokenize_batch(examples, tokenizer):

    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def prepare_dataset(
    dataset,
    tokenizer,
    sample_size,
):

    dataset = (
        dataset
        .shuffle(seed=SEED)
        .select(range(sample_size))
    )

    dataset = dataset.map(
        lambda examples: tokenize_batch(
            examples,
            tokenizer
        ),
        batched=True,
    )

    dataset = dataset.remove_columns(
        [
            "premise",
            "hypothesis",
        ]
    )

    dataset = dataset.rename_column(
        "label",
        "labels"
    )

    return dataset


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

        for batch in tqdm(
            dataloader,
            desc=f"Evaluating {language_name}",
        ):

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
                .cpu()
                .tolist()
            )

            true_labels.extend(
                batch["labels"]
                .cpu()
                .tolist()
            )

    accuracy = accuracy_score(
        true_labels,
        predictions,
    )

    average_loss = (
        total_loss / len(dataloader)
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
        "loss": average_loss,
        "accuracy": accuracy,
    }


def main():

    # ---------------------------
    # Reproducibility
    # ---------------------------

    torch.manual_seed(SEED)

    # ---------------------------
    # Device
    # ---------------------------

    device = get_device()

    print("Using device:")
    print(device)

    # ---------------------------
    # Tokenizer
    # ---------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ---------------------------
    # Datasets
    # ---------------------------

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

    print("\nPreparing English training data...")

    english_train = prepare_dataset(
        english_dataset["train"],
        tokenizer,
        TRAIN_SAMPLES,
    )

    print("Preparing English validation data...")

    english_validation = prepare_dataset(
        english_dataset["validation"],
        tokenizer,
        VALIDATION_SAMPLES,
    )

    print("Preparing Hindi validation data...")

    hindi_validation = prepare_dataset(
        hindi_dataset["validation"],
        tokenizer,
        VALIDATION_SAMPLES,
    )

    # ---------------------------
    # Dynamic padding
    # ---------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    )

    train_loader = DataLoader(
        english_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    english_validation_loader = DataLoader(
        english_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    hindi_validation_loader = DataLoader(
        hindi_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    # ---------------------------
    # Model
    # ---------------------------

    print("\nLoading mBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        },
        label2id={
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        },
    )

    model.to(device)

    # ---------------------------
    # Optimizer
    # ---------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------
    # Before training
    # ---------------------------

    print("\n==============================")
    print("BEFORE TRAINING")
    print("==============================")

    evaluate_model(
        model,
        english_validation_loader,
        device,
        "English",
    )

    evaluate_model(
        model,
        hindi_validation_loader,
        device,
        "Hindi",
    )

    # ---------------------------
    # Training
    # ---------------------------

    print("\n==============================")
    print("TRAINING")
    print("==============================")

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            ),
        )

        for batch in progress_bar:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # Remove gradients from
            # previous optimizer step.
            optimizer.zero_grad(
                set_to_none=True
            )

            # Forward pass
            outputs = model(**batch)

            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Prevent extremely
            # large gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # Update parameters
            optimizer.step()

            total_train_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_train_loss = (
            total_train_loss /
            len(train_loader)
        )

        print(
            f"\nEpoch {epoch + 1} "
            f"average training loss: "
            f"{average_train_loss:.4f}"
        )

    # ---------------------------
    # After training
    # ---------------------------

    print("\n==============================")
    print("AFTER TRAINING")
    print("==============================")

    english_results = evaluate_model(
        model,
        english_validation_loader,
        device,
        "English",
    )

    hindi_results = evaluate_model(
        model,
        hindi_validation_loader,
        device,
        "Hindi",
    )

    # ---------------------------
    # Cross-lingual transfer gap
    # ---------------------------

    transfer_gap = (
        english_results["accuracy"]
        - hindi_results["accuracy"]
    )

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(
        f"English accuracy: "
        f"{english_results['accuracy']:.4f}"
    )

    print(
        f"Hindi accuracy: "
        f"{hindi_results['accuracy']:.4f}"
    )

    print(
        f"Cross-lingual transfer gap: "
        f"{transfer_gap:.4f}"
    )

    # ---------------------------
    # Save model
    # ---------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print(
        f"\nSaving model to "
        f"{OUTPUT_DIR}"
    )

    model.save_pretrained(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    main()