import json
import random
from pathlib import Path


SEED = 20260809

ANNOTATION_PATH = Path(
    "datasets/raw/coco/annotations/instances_train2017.json"
)

OUTPUT_DIR = Path(
    "datasets/manifests/calibration"
)

CALIBRATION_SIZES = [128, 256, 512, 1024]


def load_images(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["images"]


def write_jsonl(images, output_path):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as file:
        for image in images:
            sample = {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "width": image["width"],
                "height": image["height"],
                "dataset": "coco",
                "split": "train2017",
            }

            json.dump(
                sample,
                file,
                ensure_ascii=False,
            )
            file.write("\n")


def main():
    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"COCO annotation not found: {ANNOTATION_PATH}"
        )

    images = load_images(ANNOTATION_PATH)

    max_size = max(CALIBRATION_SIZES)

    if len(images) < max_size:
        raise ValueError(
            f"Not enough COCO images: "
            f"required at least {max_size}, "
            f"found {len(images)}"
        )

    # 固定原始顺序，保证不同机器上结果一致。
    images = sorted(
        images,
        key=lambda image: image["id"],
    )

    # 固定随机种子，只打乱一次。
    rng = random.Random(SEED)
    rng.shuffle(images)

    for size in CALIBRATION_SIZES:
        selected_images = images[:size]

        output_path = (
            OUTPUT_DIR
            / f"vision_coco_{size}.jsonl"
        )

        write_jsonl(
            selected_images,
            output_path,
        )

        print(
            f"[OK] COCO calibration "
            f"{size}: {output_path}"
        )


if __name__ == "__main__":
    main()
