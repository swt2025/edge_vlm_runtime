import json
import random
from pathlib import Path


SEED = 20260809

POPE_ROOT = Path("datasets/raw/pope/output/coco")
COCO_VAL2014_ROOT = Path("datasets/raw/coco/val2014")

OUTPUT_ROOT = Path("datasets/manifests/pope")

DEV_COUNT_PER_CATEGORY = 100


POPE_FILES = {
    "random": POPE_ROOT / "coco_pope_random.json",
    "popular": POPE_ROOT / "coco_pope_popular.json",
    "adversarial": POPE_ROOT / "coco_pope_adversarial.json",
}


def load_pope_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def convert_samples(raw_samples, category):
    samples = []

    for raw_sample in raw_samples:
        question_id = raw_sample["question_id"]

        sample = {
            "sample_id": f"pope_{category}_{question_id}",
            "question_id": question_id,
            "image": raw_sample["image"],
            "question": raw_sample["text"],
            "answer": raw_sample["label"],
            "category": category,
            "dataset": "pope",
            "split": "official",
        }

        samples.append(sample)

    return samples


def select_dev_samples(samples, count, category):
    samples = sorted(
        samples,
        key=lambda sample: sample["question_id"],
    )

    # 给不同 category 派生一个固定 seed，
    # 避免依赖外层调用顺序。
    category_seed = {
        "random": SEED,
        "popular": SEED + 1,
        "adversarial": SEED + 2,
    }[category]

    rng = random.Random(category_seed)
    rng.shuffle(samples)

    if len(samples) < count:
        raise ValueError(
            f"{category}: required {count} samples, "
            f"but only found {len(samples)}"
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


def check_images(samples):
    missing_images = []

    for sample in samples:
        image_path = COCO_VAL2014_ROOT / sample["image"]

        if not image_path.exists():
            missing_images.append(image_path)

    if missing_images:
        print(
            f"Warning: {len(missing_images)} POPE images are missing."
        )

        for image_path in missing_images[:5]:
            print(f"  missing: {image_path}")

        return False

    return True


def main():
    all_dev_samples = []

    for category, input_path in POPE_FILES.items():

        if not input_path.exists():
            raise FileNotFoundError(
                f"POPE annotation not found: {input_path}"
            )

        raw_samples = load_pope_jsonl(input_path)

        samples = convert_samples(
            raw_samples,
            category,
        )

        print(
            f"{category}: {len(samples)} official samples"
        )

        # 完整官方数据也转换成我们自己的统一格式。
        write_jsonl(
            samples,
            OUTPUT_ROOT / f"{category}.jsonl",
        )

        dev_samples = select_dev_samples(
            samples,
            DEV_COUNT_PER_CATEGORY,
            category,
        )

        all_dev_samples.extend(dev_samples)

        check_images(samples)

    # 三种策略各 100 条，因此 dev 一共 300 条。
    all_dev_samples = sorted(
        all_dev_samples,
        key=lambda sample: sample["sample_id"],
    )

    write_jsonl(
        all_dev_samples,
        OUTPUT_ROOT / "dev300.jsonl",
    )

    print(
        f"POPE dev set: {len(all_dev_samples)} samples"
    )


if __name__ == "__main__":
    main()
