import json
import random
from pathlib import Path


SEED = 20260809

OUTPUT_PATH = Path(
    "datasets/manifests/golden_baseline.jsonl"
)


SOURCES = {
    "gqa": {
        "path": Path(
            "datasets/manifests/gqa/dev500.jsonl"
        ),
        "count": 10,
        "id_field": "question_id",
    },

    "refcoco": {
        "path": Path(
            "datasets/manifests/refcoco/refcoco/dev200.jsonl"
        ),
        "count": 5,
        "id_field": "sample_id",
    },

    "refcoco+": {
        "path": Path(
            "datasets/manifests/refcoco/refcoco+/dev200.jsonl"
        ),
        "count": 5,
        "id_field": "sample_id",
    },

    "refcocog": {
        "path": Path(
            "datasets/manifests/refcoco/refcocog/dev200.jsonl"
        ),
        "count": 5,
        "id_field": "sample_id",
    },

    "pope": {
        "path": Path(
            "datasets/manifests/pope/dev300.jsonl"
        ),
        "count": 10,
        "id_field": "sample_id",
    },
}


def load_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def select_samples(
    samples,
    count,
    id_field,
    seed,
):
    samples = sorted(
        samples,
        key=lambda sample: str(sample[id_field]),
    )

    rng = random.Random(seed)
    rng.shuffle(samples)

    if len(samples) < count:
        raise ValueError(
            f"Required {count} samples, "
            f"but found only {len(samples)}"
        )

    return samples[:count]


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
    golden_samples = []

    for index, (task, config) in enumerate(
        SOURCES.items()
    ):
        path = config["path"]

        if not path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {path}"
            )

        samples = load_jsonl(path)

        selected = select_samples(
            samples=samples,
            count=config["count"],
            id_field=config["id_field"],
            seed=SEED + index,
        )

        for sample in selected:
            # 不破坏原 dict。
            golden_sample = sample.copy()

            golden_sample["golden_task"] = task

            golden_samples.append(golden_sample)

        print(
            f"{task}: selected {len(selected)}"
        )

    write_jsonl(
        golden_samples,
        OUTPUT_PATH,
    )

    print(
        f"Golden baseline: "
        f"{len(golden_samples)} total samples"
    )


if __name__ == "__main__":
    main()
