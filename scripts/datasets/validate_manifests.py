import json
from pathlib import Path


COCO_TRAIN2017_ROOT = Path(
    "datasets/raw/coco/train2017"
)

COCO_TRAIN2014_ROOT = Path(
    "datasets/raw/coco/train2014"
)

COCO_VAL2014_ROOT = Path(
    "datasets/raw/coco/val2014"
)

GQA_IMAGE_ROOT = Path(
    "datasets/raw/gqa/images"
)


MANIFEST_ROOT = Path(
    "datasets/manifests"
)


def load_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}: {error}"
                ) from error

            samples.append(sample)

    return samples


def require_fields(
    samples,
    required_fields,
    manifest_name,
):
    for index, sample in enumerate(samples):
        for field in required_fields:
            if field not in sample:
                raise ValueError(
                    f"{manifest_name}: "
                    f"sample {index} missing field '{field}'"
                )


def require_unique(
    samples,
    id_field,
    manifest_name,
):
    ids = set()

    for sample in samples:
        sample_id = sample[id_field]

        if sample_id in ids:
            raise ValueError(
                f"{manifest_name}: "
                f"duplicate {id_field}={sample_id}"
            )

        ids.add(sample_id)


def require_count(
    samples,
    expected_count,
    manifest_name,
):
    if len(samples) != expected_count:
        raise ValueError(
            f"{manifest_name}: "
            f"expected {expected_count}, "
            f"found {len(samples)}"
        )


def require_subset(
    smaller,
    larger,
    id_field,
    smaller_name,
    larger_name,
):
    smaller_ids = {
        sample[id_field]
        for sample in smaller
    }

    larger_ids = {
        sample[id_field]
        for sample in larger
    }

    if not smaller_ids.issubset(larger_ids):
        raise ValueError(
            f"{smaller_name} is not a subset "
            f"of {larger_name}"
        )


def check_coco_calibration():
    counts = [
        128,
        256,
        512,
        1024,
    ]

    manifests = {}

    for count in counts:
        path = (
            MANIFEST_ROOT
            / "calibration"
            / f"vision_coco_{count}.jsonl"
        )

        samples = load_jsonl(path)

        require_count(
            samples,
            count,
            path.name,
        )

        require_fields(
            samples,
            [
                "image_id",
                "file_name",
                "width",
                "height",
                "dataset",
                "split",
            ],
            path.name,
        )

        require_unique(
            samples,
            "image_id",
            path.name,
        )

        for sample in samples:
            image_path = (
                COCO_TRAIN2017_ROOT
                / sample["file_name"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"COCO image missing: {image_path}"
                )

        manifests[count] = samples

        print(
            f"[PASS] COCO calibration {count}"
        )

    require_subset(
        manifests[128],
        manifests[256],
        "image_id",
        "COCO 128",
        "COCO 256",
    )

    require_subset(
        manifests[256],
        manifests[512],
        "image_id",
        "COCO 256",
        "COCO 512",
    )

    require_subset(
        manifests[512],
        manifests[1024],
        "image_id",
        "COCO 512",
        "COCO 1024",
    )

    print("[PASS] COCO calibration nesting")


def check_gqa():
    dev_path = (
        MANIFEST_ROOT
        / "gqa"
        / "dev500.jsonl"
    )

    final_path = (
        MANIFEST_ROOT
        / "gqa"
        / "final2000.jsonl"
    )

    dev = load_jsonl(dev_path)
    final = load_jsonl(final_path)

    required_fields = [
        "question_id",
        "image_id",
        "question",
        "answer",
        "semantic_type",
        "structural_type",
        "detailed_type",
        "dataset",
        "split",
    ]

    require_count(dev, 500, "GQA dev500")
    require_count(final, 2000, "GQA final2000")

    require_fields(
        dev,
        required_fields,
        "GQA dev500",
    )

    require_fields(
        final,
        required_fields,
        "GQA final2000",
    )

    require_unique(
        dev,
        "question_id",
        "GQA dev500",
    )

    require_unique(
        final,
        "question_id",
        "GQA final2000",
    )

    require_subset(
        dev,
        final,
        "question_id",
        "GQA dev500",
        "GQA final2000",
    )

    # 如果你的 GQA 图片目录并不是
    # datasets/raw/gqa/images，
    # 只需要修改顶部 GQA_IMAGE_ROOT。
    if GQA_IMAGE_ROOT.exists():
        for sample in final:
            image_path = (
                GQA_IMAGE_ROOT
                / f'{sample["image_id"]}.jpg'
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"GQA image missing: {image_path}"
                )

    print("[PASS] GQA")


def valid_bbox(bbox):
    if not isinstance(bbox, list):
        return False

    if len(bbox) != 4:
        return False

    x1, y1, x2, y2 = bbox

    if x2 <= x1:
        return False

    if y2 <= y1:
        return False

    return True


def check_refcoco_dataset(dataset_name):
    root = (
        MANIFEST_ROOT
        / "refcoco"
        / dataset_name
    )

    dev_path = root / "dev200.jsonl"
    final_path = root / "final500.jsonl"

    dev = load_jsonl(dev_path)
    final = load_jsonl(final_path)

    required_fields = [
        "sample_id",
        "ref_id",
        "sent_id",
        "ann_id",
        "image_id",
        "file_name",
        "image_width",
        "image_height",
        "expression",
        "bbox",
        "bbox_format",
        "category_id",
        "category",
        "dataset",
        "split",
        "split_by",
    ]

    require_count(
        dev,
        200,
        f"{dataset_name} dev200",
    )

    require_count(
        final,
        500,
        f"{dataset_name} final500",
    )

    require_fields(
        dev,
        required_fields,
        dataset_name,
    )

    require_fields(
        final,
        required_fields,
        dataset_name,
    )

    require_unique(
        dev,
        "sample_id",
        f"{dataset_name} dev200",
    )

    require_unique(
        final,
        "sample_id",
        f"{dataset_name} final500",
    )

    require_subset(
        dev,
        final,
        "sample_id",
        f"{dataset_name} dev200",
        f"{dataset_name} final500",
    )

    for sample in final:

        if sample["bbox_format"] != "xyxy":
            raise ValueError(
                f"{dataset_name}: "
                f"unexpected bbox format"
            )

        if not valid_bbox(sample["bbox"]):
            raise ValueError(
                f"{dataset_name}: "
                f"invalid bbox for "
                f"{sample['sample_id']}"
            )

        image_path = (
            COCO_TRAIN2014_ROOT
            / sample["file_name"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"RefCOCO image missing: {image_path}"
            )

    print(f"[PASS] {dataset_name}")


def check_pope():
    categories = [
        "random",
        "popular",
        "adversarial",
    ]

    full_samples = []

    for category in categories:
        path = (
            MANIFEST_ROOT
            / "pope"
            / f"{category}.jsonl"
        )

        samples = load_jsonl(path)

        require_fields(
            samples,
            [
                "sample_id",
                "question_id",
                "image",
                "question",
                "answer",
                "category",
                "dataset",
                "split",
            ],
            f"POPE {category}",
        )

        require_unique(
            samples,
            "sample_id",
            f"POPE {category}",
        )

        for sample in samples:
            image_path = (
                COCO_VAL2014_ROOT
                / sample["image"]
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"POPE image missing: {image_path}"
                )

        full_samples.extend(samples)

    dev_path = (
        MANIFEST_ROOT
        / "pope"
        / "dev300.jsonl"
    )

    dev = load_jsonl(dev_path)

    require_count(
        dev,
        300,
        "POPE dev300",
    )

    require_unique(
        dev,
        "sample_id",
        "POPE dev300",
    )

    require_subset(
        dev,
        full_samples,
        "sample_id",
        "POPE dev300",
        "POPE official sets",
    )

    print("[PASS] POPE")


def check_golden():
    path = (
        MANIFEST_ROOT
        / "golden_baseline.jsonl"
    )

    samples = load_jsonl(path)

    require_count(
        samples,
        35,
        "Golden baseline",
    )

    require_fields(
        samples,
        ["golden_task"],
        "Golden baseline",
    )

    print("[PASS] Golden baseline")


def main():
    check_coco_calibration()

    check_gqa()

    check_refcoco_dataset("refcoco")
    check_refcoco_dataset("refcoco+")
    check_refcoco_dataset("refcocog")

    check_pope()

    check_golden()

    print()
    print("All manifests are valid.")


if __name__ == "__main__":
    main()
