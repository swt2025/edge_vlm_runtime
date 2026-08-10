import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from transformers import AutoModelForMultimodalLM
GOLDEN_MANIFEST_PATH = Path("datasets/manifests/golden_baseline.jsonl")
MODEL_PATH = Path("models/Qwen3.5-4B")

GQA_IMAGE_ROOT = Path(
    "datasets/raw/gqa/images"
)

COCO_TRAIN2014_ROOT = Path(
    "datasets/raw/coco/train2014"
)

COCO_VAL2014_ROOT = Path(
    "datasets/raw/coco/val2014"
)

def load_jsonl(path) :
    samples = []
    with open(path, "r", encoding="utf-8") as file :
        for line in file :
            line = line.strip()
            if not line :
                continue
            samples.append(json.loads(line))
    return samples

def resolve_image_path(sample) :
    task = sample["golden_task"]

    if task == "gqa" :
        image_path = GQA_IMAGE_ROOT / f'{sample["image_id"]}.jpg'
    elif task in ["refcoco", "refcoco+", "refcocog"] :
        image_path = (COCO_TRAIN2014_ROOT / sample["file_name"])
    elif task == "pope":
        image_path = (
            COCO_VAL2014_ROOT
            / sample["image"]
        )
    else :
        raise ValueError(f"Unsupported golden task:{task}")

    if not image_path.exists() :
        raise FileNotFoundError(f"image path {image_path} in not exist")

    return image_path

def build_prompt(sample) :
    task = sample["golden_task"]

    if task == "gqa" :
        return sample["question"]
    elif task == "pope" :
        return sample["question"]
    elif task in ["refcoco", "refcocog", "refcoco+"] :
        return (
            f'Locate the object described as "{sample["expression"]}". '
            "Return only its bounding box."
        )
    else :
        raise ValueError(f"Unsupported golden task:{task}")



def main() :

    samples = load_jsonl(GOLDEN_MANIFEST_PATH)

    print(f"golden samples:{len(samples)}")

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype = torch.bfloat16,
    )

    model.eval()

    for sample in samples :
        image_path = resolve_image_path(sample)
        prompt = build_prompt(sample)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "path": str(image_path),
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to(model.device)

        with torch.inference_mode() :
            outputs = model.generate(
                **inputs,
                max_new_tokens = 40,
                do_sample = False,
                num_beams = 1,
            )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[-1]:
        ]

        answer = processor.decode(
            generated_tokens,
            skip_special_tokens = True,
        )

        print(sample["golden_task"],
              image_path, 
              prompt,
              answer,
              )


if __name__ == "__main__" :
    main()
