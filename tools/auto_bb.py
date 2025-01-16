import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def main(args):
    if not os.path.exists(args.img_dir):
        print("args image directory dont exist")
        return

    model_id = "IDEA-Research/grounding-dino-base"

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    processor = AutoProcessor.from_pretrained(model_id)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    for row in os.scandir(args.img_dir):
        if row.name.endswith("png") or row.name.endswith("jpg"):
            image = Image.open(row.path)
            inputs = processor(
                images=[image], text=args.prompt, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.3,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]],
            )
            with open("out.csv", "w") as f:
                for lbl, bb in zip(results[0]["labels"], results[0]["boxes"]):
                    bb = bb.cpu().numpy()
                    f.write(f"{row.name},{lbl},{bb[0]}, {bb[1]}, {bb[2]}, {bb[3]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="auto bounding box labeling")
    parser.add_argument("-d", "--img-dir", type=str, help="images", required=True)
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="text query (lowercase ends with .)",
        required=True,
    )
    args = parser.parse_args()
    main(args)
