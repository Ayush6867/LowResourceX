from datasets import load_dataset


def print_example(dataset, split, index, label_names, language):
    example = dataset[split][index]

    print(f"\n{'=' * 30}")
    print(f"{language} EXAMPLE")
    print(f"{'=' * 30}")

    print("\nPremise:")
    print(example["premise"])

    print("\nHypothesis:")
    print(example["hypothesis"])

    label_id = example["label"]

    print("\nLabel ID:")
    print(label_id)

    print("\nLabel Name:")
    print(label_names[label_id])


def main():

    print("Loading English XNLI...")
    english_dataset = load_dataset(
        "facebook/xnli",
        "en"
    )

    print("\nLoading Hindi XNLI...")
    hindi_dataset = load_dataset(
        "facebook/xnli",
        "hi"
    )

    print("\n==============================")
    print("ENGLISH DATASET")
    print("==============================")

    print(english_dataset)

    print("\n==============================")
    print("HINDI DATASET")
    print("==============================")

    print(hindi_dataset)

    label_names = english_dataset["train"].features["label"].names

    print("\n==============================")
    print("LABEL MAPPING")
    print("==============================")

    for index, label in enumerate(label_names):
        print(index, "->", label)

    print("\n==============================")
    print("ENGLISH SPLIT SIZES")
    print("==============================")

    for split_name in english_dataset.keys():
        print(
            split_name,
            ":",
            len(english_dataset[split_name])
        )

    print("\n==============================")
    print("HINDI SPLIT SIZES")
    print("==============================")

    for split_name in hindi_dataset.keys():
        print(
            split_name,
            ":",
            len(hindi_dataset[split_name])
        )

    print_example(
        dataset=english_dataset,
        split="validation",
        index=0,
        label_names=label_names,
        language="ENGLISH"
    )

    print_example(
        dataset=hindi_dataset,
        split="validation",
        index=0,
        label_names=label_names,
        language="HINDI"
    )


if __name__ == "__main__":
    main()