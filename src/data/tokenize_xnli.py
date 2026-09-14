from datasets import load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "bert-base-multilingual-cased"


def inspect_tokenization(
    tokenizer,
    premise,
    hypothesis,
    language
):
    print("\n" + "=" * 60)
    print(f"{language} TOKENIZATION")
    print("=" * 60)

    print("\nPremise:")
    print(premise)

    print("\nHypothesis:")
    print(hypothesis)

    encoded = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"][0]

    tokens = tokenizer.convert_ids_to_tokens(
        input_ids
    )

    print("\nFirst 40 tokens:")
    print(tokens[:40])

    print("\nFirst 40 token IDs:")
    print(input_ids[:40])

    print("\nFirst 40 attention-mask values:")
    print(encoded["attention_mask"][0][:40])

    if "token_type_ids" in encoded:
        print("\nFirst 40 token-type IDs:")
        print(encoded["token_type_ids"][0][:40])

    print("\nTensor shape:")
    print(encoded["input_ids"].shape)


def main():

    print("Loading mBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Tokenizer loaded successfully.")

    print("\nVocabulary size:")
    print(tokenizer.vocab_size)

    print("\nSpecial tokens:")
    print("CLS:", tokenizer.cls_token)
    print("SEP:", tokenizer.sep_token)
    print("PAD:", tokenizer.pad_token)
    print("UNK:", tokenizer.unk_token)

    print("\nLoading XNLI examples...")

    english_dataset = load_dataset(
        "facebook/xnli",
        "en"
    )

    hindi_dataset = load_dataset(
        "facebook/xnli",
        "hi"
    )

    english_example = english_dataset["validation"][0]
    hindi_example = hindi_dataset["validation"][0]

    inspect_tokenization(
        tokenizer=tokenizer,
        premise=english_example["premise"],
        hypothesis=english_example["hypothesis"],
        language="ENGLISH"
    )

    inspect_tokenization(
        tokenizer=tokenizer,
        premise=hindi_example["premise"],
        hypothesis=hindi_example["hypothesis"],
        language="HINDI"
    )


if __name__ == "__main__":
    main()