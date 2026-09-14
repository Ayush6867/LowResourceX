import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)


MODEL_NAME = "bert-base-multilingual-cased"

LABEL_NAMES = [
    "entailment",
    "neutral",
    "contradiction",
]


def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():

    # ----------------------------------
    # Device
    # ----------------------------------

    device = get_device()

    print("Using device:")
    print(device)

    # ----------------------------------
    # Tokenizer
    # ----------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # ----------------------------------
    # Model
    # ----------------------------------

    print("\nLoading mBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    model.to(device)

    print("Model loaded successfully.")

    # ----------------------------------
    # Dataset
    # ----------------------------------

    print("\nLoading English XNLI...")

    dataset = load_dataset(
        "facebook/xnli",
        "en"
    )

    examples = dataset["validation"].select(
        range(2)
    )

    # ----------------------------------
    # Tokenize individual examples
    # ----------------------------------

    tokenized_examples = []

    for example in examples:

        encoded = tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            max_length=128
        )

        encoded["labels"] = example["label"]

        tokenized_examples.append(
            encoded
        )

    # ----------------------------------
    # Dynamic padding
    # ----------------------------------

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt"
    )

    batch = data_collator(
        tokenized_examples
    )

    print("\nBatch keys:")
    print(batch.keys())

    print("\nInput shape:")
    print(batch["input_ids"].shape)

    print("\nLabels:")
    print(batch["labels"])

    # ----------------------------------
    # Move batch to device
    # ----------------------------------

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    # ----------------------------------
    # Forward pass
    # ----------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(
            **batch
        )

    # ----------------------------------
    # Model outputs
    # ----------------------------------

    print("\nLoss:")
    print(outputs.loss.item())

    print("\nRaw logits:")
    print(outputs.logits)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predictions = torch.argmax(
        probabilities,
        dim=-1
    )

    probabilities = probabilities.cpu()
    predictions = predictions.cpu()

    # ----------------------------------
    # Human-readable results
    # ----------------------------------

    for index, example in enumerate(examples):

        true_label = example["label"]
        predicted_label = predictions[index].item()

        print("\n" + "=" * 60)
        print(f"EXAMPLE {index + 1}")
        print("=" * 60)

        print("\nPremise:")
        print(example["premise"])

        print("\nHypothesis:")
        print(example["hypothesis"])

        print("\nTrue label:")
        print(
            LABEL_NAMES[true_label]
        )

        print("\nPredicted label:")
        print(
            LABEL_NAMES[predicted_label]
        )

        print("\nProbabilities:")

        for label_index, label_name in enumerate(
            LABEL_NAMES
        ):

            probability = probabilities[
                index
            ][label_index].item()

            print(
                f"{label_name}: "
                f"{probability:.4f}"
            )


if __name__ == "__main__":
    main()