from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-multilingual-cased"

MAX_LENGTH = 128

TRAIN_SAMPLES = 2000
VALIDATION_SAMPLES = 500


def tokenize_batch(examples, tokenizer):

    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        max_length=MAX_LENGTH
    )


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading English XNLI...")

    english_dataset = load_dataset(
        "facebook/xnli",
        "en"
    )

    print("Loading Hindi XNLI...")

    hindi_dataset = load_dataset(
        "facebook/xnli",
        "hi"
    )

    print("\nOriginal English train size:")
    print(len(english_dataset["train"]))

    print("\nOriginal English validation size:")
    print(len(english_dataset["validation"]))

    print("\nOriginal Hindi validation size:")
    print(len(hindi_dataset["validation"]))

    # -----------------------------------
    # Small datasets for local development
    # -----------------------------------

    english_train_small = (
        english_dataset["train"]
        .shuffle(seed=42)
        .select(range(TRAIN_SAMPLES))
    )

    english_validation_small = (
        english_dataset["validation"]
        .shuffle(seed=42)
        .select(range(VALIDATION_SAMPLES))
    )

    hindi_validation_small = (
        hindi_dataset["validation"]
        .shuffle(seed=42)
        .select(range(VALIDATION_SAMPLES))
    )

    print("\nSmall English training set:")
    print(len(english_train_small))

    print("\nSmall English validation set:")
    print(len(english_validation_small))

    print("\nSmall Hindi validation set:")
    print(len(hindi_validation_small))

    print("\nTokenizing English training set...")

    english_train_tokenized = english_train_small.map(
        lambda examples: tokenize_batch(
            examples,
            tokenizer
        ),
        batched=True
    )

    print("\nTokenizing English validation set...")

    english_validation_tokenized = (
        english_validation_small.map(
            lambda examples: tokenize_batch(
                examples,
                tokenizer
            ),
            batched=True
        )
    )

    print("\nTokenizing Hindi validation set...")

    hindi_validation_tokenized = (
        hindi_validation_small.map(
            lambda examples: tokenize_batch(
                examples,
                tokenizer
            ),
            batched=True
        )
    )

    print("\nTokenized English example:")

    example = english_train_tokenized[0]

    print("Premise:")
    print(example["premise"])

    print("\nHypothesis:")
    print(example["hypothesis"])

    print("\nLabel:")
    print(example["label"])

    print("\nInput IDs:")
    print(example["input_ids"][:20])

    print("\nAttention Mask:")
    print(example["attention_mask"][:20])

    print("\nToken Type IDs:")
    print(example["token_type_ids"][:20])


if __name__ == "__main__":
    main()