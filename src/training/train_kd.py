import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


# ============================================================
# Constants
# ============================================================

DEFAULT_TEACHER_DIR = "models/roberta_teacher_10k"

STUDENT_MODEL_NAME = "bert-base-multilingual-cased"


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
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Knowledge distillation from an English "
            "RoBERTa teacher to multilingual BERT."
        )
    )

    parser.add_argument(
        "--teacher-dir",
        type=str,
        default=DEFAULT_TEACHER_DIR,
        help="Directory containing the trained teacher.",
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
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Weight assigned to KD loss. "
            "CE receives weight 1-alpha."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/mbert_kd_10k",
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

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
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


# ============================================================
# Custom raw-text collator
#
# Important:
# Teacher and student use DIFFERENT tokenizers.
#
# Therefore we keep raw premise/hypothesis text until
# inside the training loop.
# ============================================================

def raw_collate_fn(examples):

    premises = [
        example["premise"]
        for example in examples
    ]

    hypotheses = [
        example["hypothesis"]
        for example in examples
    ]

    labels = torch.tensor(
        [
            example["label"]
            for example in examples
        ],
        dtype=torch.long,
    )

    return {
        "premise": premises,
        "hypothesis": hypotheses,
        "labels": labels,
    }


# ============================================================
# Tokenization
# ============================================================

def tokenize_batch(
    tokenizer,
    premises,
    hypotheses,
    max_length,
    device,
):

    encoded = tokenizer(
        premises,
        hypotheses,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    return encoded


# ============================================================
# Evaluation
# ============================================================

def evaluate_student(
    model,
    tokenizer,
    dataloader,
    device,
    max_length,
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

            labels = batch["labels"].to(
                device
            )

            inputs = tokenize_batch(
                tokenizer=tokenizer,
                premises=batch["premise"],
                hypotheses=batch["hypothesis"],
                max_length=max_length,
                device=device,
            )

            outputs = model(
                **inputs,
                labels=labels,
            )

            total_loss += (
                outputs.loss.item()
            )

            batch_predictions = (
                torch.argmax(
                    outputs.logits,
                    dim=-1,
                )
            )

            predictions.extend(
                batch_predictions
                .detach()
                .cpu()
                .tolist()
            )

            true_labels.extend(
                labels
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
# KD training
# ============================================================

def train_one_epoch(
    teacher,
    teacher_tokenizer,
    student,
    student_tokenizer,
    dataloader,
    optimizer,
    device,
    max_length,
    alpha,
    temperature,
    epoch,
    total_epochs,
):

    # Teacher must never train.
    teacher.eval()

    # Student is the model being trained.
    student.train()

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    kd_loss_sum = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=(
            f"KD Epoch "
            f"{epoch}/{total_epochs}"
        ),
    )

    for batch in progress_bar:

        labels = batch["labels"].to(
            device
        )

        # ----------------------------------------------------
        # Teacher tokenization
        # ----------------------------------------------------

        teacher_inputs = tokenize_batch(
            tokenizer=teacher_tokenizer,
            premises=batch["premise"],
            hypotheses=batch["hypothesis"],
            max_length=max_length,
            device=device,
        )

        # ----------------------------------------------------
        # Student tokenization
        # ----------------------------------------------------

        student_inputs = tokenize_batch(
            tokenizer=student_tokenizer,
            premises=batch["premise"],
            hypotheses=batch["hypothesis"],
            max_length=max_length,
            device=device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Teacher forward pass
        #
        # no_grad() is critical.
        # We are NOT training the teacher.
        # ----------------------------------------------------

        with torch.no_grad():

            teacher_outputs = teacher(
                **teacher_inputs
            )

            teacher_logits = (
                teacher_outputs.logits
            )

        # ----------------------------------------------------
        # Student forward pass
        # ----------------------------------------------------

        student_outputs = student(
            **student_inputs
        )

        student_logits = (
            student_outputs.logits
        )

        # ----------------------------------------------------
        # Hard-label cross-entropy loss
        # ----------------------------------------------------

        ce_loss = F.cross_entropy(
            student_logits,
            labels,
        )

        # ----------------------------------------------------
        # Temperature-scaled teacher probabilities
        # ----------------------------------------------------

        teacher_probabilities = F.softmax(
            teacher_logits / temperature,
            dim=-1,
        )

        # ----------------------------------------------------
        # Temperature-scaled student log probabilities
        # ----------------------------------------------------

        student_log_probabilities = (
            F.log_softmax(
                student_logits / temperature,
                dim=-1,
            )
        )

        # ----------------------------------------------------
        # Knowledge-distillation loss
        # ----------------------------------------------------

        kd_loss = F.kl_div(
            student_log_probabilities,
            teacher_probabilities,
            reduction="batchmean",
        )

        # Standard temperature correction.
        kd_loss = (
            kd_loss
            * (temperature ** 2)
        )

        # ----------------------------------------------------
        # Combined loss
        # ----------------------------------------------------

        total_loss = (
            (1.0 - alpha) * ce_loss
            + alpha * kd_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        total_loss_sum += (
            total_loss.item()
        )

        ce_loss_sum += (
            ce_loss.item()
        )

        kd_loss_sum += (
            kd_loss.item()
        )

        progress_bar.set_postfix(
            total=(
                f"{total_loss.item():.4f}"
            ),
            ce=(
                f"{ce_loss.item():.4f}"
            ),
            kd=(
                f"{kd_loss.item():.4f}"
            ),
        )

    num_batches = len(
        dataloader
    )

    return {
        "total_loss": (
            total_loss_sum / num_batches
        ),
        "ce_loss": (
            ce_loss_sum / num_batches
        ),
        "kd_loss": (
            kd_loss_sum / num_batches
        ),
    }


# ============================================================
# Save results
# ============================================================

def save_results(
    output_dir,
    results,
):

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    results_path = os.path.join(
        output_dir,
        "results.json",
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nResults saved to: "
        f"{results_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if not 0.0 <= args.alpha <= 1.0:

        raise ValueError(
            "--alpha must be between 0 and 1."
        )

    if args.temperature <= 0:

        raise ValueError(
            "--temperature must be greater than 0."
        )

    if not os.path.isdir(
        args.teacher_dir
    ):

        raise FileNotFoundError(
            "\nTeacher model directory was not found:\n"
            f"{args.teacher_dir}\n\n"
            "Train the teacher first or restore the "
            "checkpoint before running KD."
        )

    set_seed(
        args.seed
    )

    device = get_device()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "LOWRESOURCEX - KNOWLEDGE DISTILLATION"
    )

    print(
        "========================================"
    )

    print(
        f"Teacher: {args.teacher_dir}"
    )

    print(
        f"Student: {STUDENT_MODEL_NAME}"
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
        f"Alpha: {args.alpha}"
    )

    print(
        f"Temperature: "
        f"{args.temperature}"
    )

    print(
        f"Seed: {args.seed}"
    )

    print(
        "\nUsing device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load teacher
    # --------------------------------------------------------

    print(
        "\nLoading teacher tokenizer..."
    )

    teacher_tokenizer = (
        AutoTokenizer.from_pretrained(
            args.teacher_dir
        )
    )

    print(
        "Loading trained RoBERTa teacher..."
    )

    teacher = (
        AutoModelForSequenceClassification
        .from_pretrained(
            args.teacher_dir
        )
    )

    teacher.to(
        device
    )

    teacher.eval()

    # Freeze teacher parameters permanently.
    for parameter in teacher.parameters():

        parameter.requires_grad = False

    print(
        "Teacher loaded and frozen."
    )

    # --------------------------------------------------------
    # Load student
    # --------------------------------------------------------

    print(
        "\nLoading student tokenizer..."
    )

    student_tokenizer = (
        AutoTokenizer.from_pretrained(
            STUDENT_MODEL_NAME
        )
    )

    print(
        "Loading mBERT student..."
    )

    student = (
        AutoModelForSequenceClassification
        .from_pretrained(
            STUDENT_MODEL_NAME,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    student.to(
        device
    )

    print(
        "Student loaded successfully."
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nLoading English XNLI..."
    )

    english_dataset = load_dataset(
        "facebook/xnli",
        "en",
    )

    print(
        "Loading Hindi XNLI..."
    )

    hindi_dataset = load_dataset(
        "facebook/xnli",
        "hi",
    )

    english_train = select_subset(
        dataset=english_dataset["train"],
        sample_size=args.train_samples,
        seed=args.seed,
    )

    english_validation = select_subset(
        dataset=english_dataset[
            "validation"
        ],
        sample_size=args.validation_samples,
        seed=args.seed,
    )

    hindi_validation = select_subset(
        dataset=hindi_dataset[
            "validation"
        ],
        sample_size=args.validation_samples,
        seed=args.seed,
    )

    print(
        "\nPrepared dataset sizes:"
    )

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
    # DataLoaders
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        english_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=raw_collate_fn,
        generator=generator,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    english_validation_loader = (
        DataLoader(
            english_validation,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=raw_collate_fn,
            pin_memory=(
                device.type == "cuda"
            ),
        )
    )

    hindi_validation_loader = (
        DataLoader(
            hindi_validation,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=raw_collate_fn,
            pin_memory=(
                device.type == "cuda"
            ),
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Student before training
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "STUDENT BEFORE KD"
    )

    print(
        "========================================"
    )

    before_english = evaluate_student(
        model=student,
        tokenizer=student_tokenizer,
        dataloader=(
            english_validation_loader
        ),
        device=device,
        max_length=args.max_length,
        language_name="English",
    )

    before_hindi = evaluate_student(
        model=student,
        tokenizer=student_tokenizer,
        dataloader=(
            hindi_validation_loader
        ),
        device=device,
        max_length=args.max_length,
        language_name="Hindi",
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_english_accuracy = -1.0

    print(
        "\n========================================"
    )

    print(
        "KD TRAINING"
    )

    print(
        "========================================"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        training_results = train_one_epoch(
            teacher=teacher,
            teacher_tokenizer=(
                teacher_tokenizer
            ),
            student=student,
            student_tokenizer=(
                student_tokenizer
            ),
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            max_length=args.max_length,
            alpha=args.alpha,
            temperature=args.temperature,
            epoch=epoch,
            total_epochs=args.epochs,
        )

        print(
            f"\nEpoch {epoch} losses:"
        )

        print(
            "Total loss: "
            f"{training_results['total_loss']:.4f}"
        )

        print(
            "Cross-entropy loss: "
            f"{training_results['ce_loss']:.4f}"
        )

        print(
            "KD loss: "
            f"{training_results['kd_loss']:.4f}"
        )

        # ----------------------------------------------------
        # Student evaluation
        # ----------------------------------------------------

        english_results = evaluate_student(
            model=student,
            tokenizer=student_tokenizer,
            dataloader=(
                english_validation_loader
            ),
            device=device,
            max_length=args.max_length,
            language_name="English",
        )

        hindi_results = evaluate_student(
            model=student,
            tokenizer=student_tokenizer,
            dataloader=(
                hindi_validation_loader
            ),
            device=device,
            max_length=args.max_length,
            language_name="Hindi",
        )

        transfer_gap = (
            english_results["accuracy"]
            - hindi_results["accuracy"]
        )

        print(
            "\nCross-lingual transfer gap: "
            f"{transfer_gap:.4f}"
        )

        epoch_results = {
            "epoch": epoch,

            "training_total_loss": (
                training_results[
                    "total_loss"
                ]
            ),

            "training_ce_loss": (
                training_results[
                    "ce_loss"
                ]
            ),

            "training_kd_loss": (
                training_results[
                    "kd_loss"
                ]
            ),

            "english_loss": (
                english_results["loss"]
            ),

            "english_accuracy": (
                english_results[
                    "accuracy"
                ]
            ),

            "hindi_loss": (
                hindi_results["loss"]
            ),

            "hindi_accuracy": (
                hindi_results[
                    "accuracy"
                ]
            ),

            "transfer_gap": float(
                transfer_gap
            ),
        }

        history.append(
            epoch_results
        )

        # ----------------------------------------------------
        # Save according to English validation only.
        # Hindi must NOT decide checkpoint selection.
        # ----------------------------------------------------

        if (
            english_results["accuracy"]
            > best_english_accuracy
        ):

            best_english_accuracy = (
                english_results[
                    "accuracy"
                ]
            )

            print(
                "\nNew best KD student."
            )

            print(
                f"Saving to "
                f"{args.output_dir}"
            )

            os.makedirs(
                args.output_dir,
                exist_ok=True,
            )

            student.save_pretrained(
                args.output_dir
            )

            student_tokenizer.save_pretrained(
                args.output_dir
            )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    final_results = history[-1]

    print(
        "\n========================================"
    )

    print(
        "FINAL KD RESULTS"
    )

    print(
        "========================================"
    )

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

    experiment_results = {

        "teacher": args.teacher_dir,

        "student": STUDENT_MODEL_NAME,

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
            "alpha": (
                args.alpha
            ),
            "temperature": (
                args.temperature
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

    save_results(
        output_dir=args.output_dir,
        results=experiment_results,
    )

    print(
        "\nKnowledge distillation "
        "experiment completed successfully."
    )


if __name__ == "__main__":
    main()