import json
import random
from pathlib import Path


SEED = 20260809

GQA_PATH = Path(
    "datasets/raw/gqa/questions1.2/"
    "val_balanced_questions.json"
)

OUTPUT_DIR = Path(
    "datasets/manifests/gqa"
)

SEMANTIC_TYPES = [
    "rel",
    "attr",
    "obj",
    "cat",
    "global",
]

DEV_COUNTS = {
    "rel": 233,
    "attr": 160,
    "obj": 59,
    "cat": 33,
    "global": 15,
}

FINAL_COUNTS = {
    "rel": 933,
    "attr": 639,
    "obj": 236,
    "cat": 131,
    "global": 61,
}


def load_gqa(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_groups(data):
    groups = {
        semantic_type: []
        for semantic_type in SEMANTIC_TYPES
    }

    for question_id, question_data in data.items():
        sample = {
            "question_id": question_id,
            "image_id": question_data["imageId"],
            "question": question_data["question"],
            "answer": question_data["answer"],
            "full_answer": question_data["fullAnswer"],
            "semantic_type": question_data["types"]["semantic"],
            "structural_type": question_data["types"]["structural"],
            "detailed_type": question_data["types"]["detailed"],
            "dataset": "gqa",
            "split": "val_balanced",
        }

        semantic_type = sample["semantic_type"]

        if semantic_type not in groups:
            raise ValueError(
                f"Unknown semantic type: {semantic_type}"
            )

        groups[semantic_type].append(sample)

    return groups


def shuffle_groups(groups):
    rng = random.Random(SEED)

    for semantic_type in SEMANTIC_TYPES:
        group = groups[semantic_type]

        group.sort(
            key=lambda sample: sample["question_id"]
        )

        rng.shuffle(group)


def build_subset(groups, counts):
    samples = []

    for semantic_type in SEMANTIC_TYPES:
        count = counts[semantic_type]

        if len(groups[semantic_type]) < count:
            raise ValueError(
                f"Not enough samples for "
                f"{semantic_type}: "
                f"required {count}, "
                f"found {len(groups[semantic_type])}"
            )

        samples.extend(
            groups[semantic_type][:count]
        )

    return samples


def write_jsonl(samples, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        for sample in samples:
            json.dump(
                sample,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def main():
    if not GQA_PATH.exists():
        raise FileNotFoundError(
            f"GQA annotation not found: {GQA_PATH}"
        )

    data = load_gqa(GQA_PATH)

    groups = build_groups(data)

    shuffle_groups(groups)

    dev_samples = build_subset(
        groups,
        DEV_COUNTS,
    )

    final_samples = build_subset(
        groups,
        FINAL_COUNTS,
    )

    write_jsonl(
        dev_samples,
        OUTPUT_DIR / "dev500.jsonl",
    )

    write_jsonl(
        final_samples,
        OUTPUT_DIR / "final2000.jsonl",
    )

    print(
        f"[OK] GQA dev: {len(dev_samples)} samples"
    )

    print(
        f"[OK] GQA final: {len(final_samples)} samples"
    )


if __name__ == "__main__":
    main()
