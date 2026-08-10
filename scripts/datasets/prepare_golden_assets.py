import json
import shutil
from pathlib import Path


GOLDEN_MANIFEST_PATH = Path(
    "datasets/manifests/golden_baseline.jsonl"
)

GQA_IMAGE_ROOT = Path(
    "datasets/raw/gqa/images"
)

COCO_TRAIN2014_ROOT = Path(
    "datasets/raw/coco/train2014"
)

COCO_VAL2014_ROOT = Path(
    "datasets/raw/coco/val2014"
)

OUTPUT_ROOT = Path(
    "datasets/golden_assets"
)

OUTPUT_IMAGE_ROOT = OUTPUT_ROOT / "images"

OUTPUT_MANIFEST_PATH = (
    OUTPUT_ROOT / "golden_baseline_portable.jsonl"
)


def load_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            samples.append(json.loads(line))

    return samples


def resolve_image_path(sample):
    task = sample["golden_task"]

    if task == "gqa":
        image_path = (
            GQA_IMAGE_ROOT
            / f'{sample["image_id"]}.jpg'
        )

    elif task in [
        "refcoco",
        "refcoco+",
        "refcocog",
    ]:
        image_path = (
            COCO_TRAIN2014_ROOT
            / sample["file_name"]
        )

    elif task == "pope":
        image_path = (
            COCO_VAL2014_ROOT
            / sample["image"]
        )

    else:
        raise ValueError(
            f"Unsupported golden task: {task}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return image_path


def build_destination_path(sample, source_path):
    task = sample["golden_task"]

    task_dir = OUTPUT_IMAGE_ROOT / task

    task_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return task_dir / source_path.name


def write_jsonl(samples, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        for sample in samples:
            json.dump(
                sample,
                file,
                ensure_ascii=False,
            )

            file.write("\n")


def main():
    if not GOLDEN_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Golden manifest not found: "
            f"{GOLDEN_MANIFEST_PATH}"
        )

    samples = load_jsonl(
        GOLDEN_MANIFEST_PATH
    )

    print(
        f"Golden samples: {len(samples)}"
    )

    OUTPUT_IMAGE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    portable_samples = []

    copied_files = set()

    for index, sample in enumerate(samples):
        source_path = resolve_image_path(
            sample
        )

        destination_path = build_destination_path(
            sample,
            source_path,
        )

        # 同一张图片可能被多个问题引用。
        # 只需要复制一次。
        destination_key = str(
            destination_path.resolve()
        )

        if destination_key not in copied_files:
            shutil.copy2(
                source_path,
                destination_path,
            )

            copied_files.add(
                destination_key
            )

        portable_sample = sample.copy()

        # 保存相对于 golden_assets/ 的路径，
        # 避免写死本机绝对路径。
        relative_image_path = (
            destination_path.relative_to(
                OUTPUT_ROOT
            )
        )

        portable_sample["image_path"] = (
            relative_image_path.as_posix()
        )

        portable_samples.append(
            portable_sample
        )

        print(
            f"[{index + 1}/{len(samples)}] "
            f"{source_path} "
            f"-> {relative_image_path}"
        )

    write_jsonl(
        portable_samples,
        OUTPUT_MANIFEST_PATH,
    )

    print()
    print("Golden assets prepared.")
    print(
        f"Samples: {len(portable_samples)}"
    )
    print(
        f"Unique images: {len(copied_files)}"
    )
    print(
        f"Images: {OUTPUT_IMAGE_ROOT}"
    )
    print(
        f"Manifest: {OUTPUT_MANIFEST_PATH}"
    )


if __name__ == "__main__":
    main()