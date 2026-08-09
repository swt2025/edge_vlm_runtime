import json
import pickle
import random
from pathlib import Path


SEED = 20260809

REFER_ROOT = Path("datasets/raw/refer")
COCO_IMAGE_ROOT = Path("datasets/raw/coco/train2014")

OUTPUT_ROOT = Path("datasets/manifests/refcoco")

DEV_COUNT = 200
FINAL_COUNT = 500


DATASETS = {
    "refcoco": {
        "split_by": "unc",
        "refs_file": "refs(unc).p",
    },
    "refcoco+": {
        "split_by": "unc",
        "refs_file": "refs(unc).p",
    },
    "refcocog": {
        "split_by": "umd",
        "refs_file": "refs(umd).p",
    },
}


def load_pickle(path):
    with open(path, "rb") as file:
        # RefCOCO 的原始 pickle 可能由 Python 2 生成，
        # latin1 可以提高 Python 3 加载旧 pickle 的兼容性。
        return pickle.load(file, encoding="latin1")


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_samples(dataset_name, split_by, refs, instances):
    images = {
        image["id"]: image
        for image in instances["images"]
    }

    annotations = {
        annotation["id"]: annotation
        for annotation in instances["annotations"]
    }

    categories = {
        category["id"]: category["name"]
        for category in instances["categories"]
    }

    samples = []

    for ref in refs:
        # 当前只构造 validation evaluation set。
        if ref["split"] != "val":
            continue

        image_id = ref["image_id"]
        ann_id = ref["ann_id"]

        image = images[image_id]
        annotation = annotations[ann_id]

        # COCO bbox:
        # [x, y, width, height]
        x, y, width, height = annotation["bbox"]

        # 项目内部统一转成：
        # [x1, y1, x2, y2]
        bbox_xyxy = [
            x,
            y,
            x + width,
            y + height,
        ]

        # 一个 ref 可能有多个 referring expressions。
        # 每个 sentence 都拆成一个独立 evaluation sample。
        for sentence in ref["sentences"]:
            sent_id = sentence["sent_id"]

            sample = {
                "sample_id": f"{dataset_name}_{sent_id}",
                "ref_id": ref["ref_id"],
                "sent_id": sent_id,
                "ann_id": ann_id,
                "image_id": image_id,
                "file_name": image["file_name"],
                "image_width": image["width"],
                "image_height": image["height"],
                "expression": sentence["sent"],
                "bbox": bbox_xyxy,
                "bbox_format": "xyxy",
                "category_id": ref["category_id"],
                "category": categories[ref["category_id"]],
                "dataset": dataset_name,
                "split": "val",
                "split_by": split_by,
            }

            samples.append(sample)

    return samples


def select_samples(samples):
    # 建立确定的初始顺序。
    samples = sorted(
        samples,
        key=lambda sample: sample["sent_id"],
    )

    rng = random.Random(SEED)
    rng.shuffle(samples)

    if len(samples) < FINAL_COUNT:
        raise ValueError(
            f"Not enough samples: "
            f"required {FINAL_COUNT}, found {len(samples)}"
        )

    final_samples = samples[:FINAL_COUNT]

    # 保证 dev200 是 final500 的子集。
    dev_samples = final_samples[:DEV_COUNT]

    return dev_samples, final_samples


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
        image_path = COCO_IMAGE_ROOT / sample["file_name"]

        if not image_path.exists():
            missing_images.append(str(image_path))

    if missing_images:
        print(
            f"Warning: {len(missing_images)} "
            f"referenced images are missing."
        )

        for path in missing_images[:5]:
            print(f"  missing: {path}")

        return False

    return True


def process_dataset(dataset_name, config):
    dataset_dir = REFER_ROOT / dataset_name

    refs_path = dataset_dir / config["refs_file"]
    instances_path = dataset_dir / "instances.json"

    if not refs_path.exists():
        raise FileNotFoundError(
            f"Refs file not found: {refs_path}"
        )

    if not instances_path.exists():
        raise FileNotFoundError(
            f"Instances file not found: {instances_path}"
        )

    refs = load_pickle(refs_path)
    instances = load_json(instances_path)

    samples = build_samples(
        dataset_name=dataset_name,
        split_by=config["split_by"],
        refs=refs,
        instances=instances,
    )

    print(
        f"{dataset_name}: "
        f"{len(samples)} validation expressions"
    )

    dev_samples, final_samples = select_samples(samples)

    output_dir = OUTPUT_ROOT / dataset_name

    write_jsonl(
        dev_samples,
        output_dir / "dev200.jsonl",
    )

    write_jsonl(
        final_samples,
        output_dir / "final500.jsonl",
    )

    images_ok = check_images(final_samples)

    print(
        f"{dataset_name}: "
        f"dev={len(dev_samples)}, "
        f"final={len(final_samples)}, "
        f"images_ok={images_ok}"
    )


def main():
    for dataset_name, config in DATASETS.items():
        process_dataset(dataset_name, config)


if __name__ == "__main__":
    main()
