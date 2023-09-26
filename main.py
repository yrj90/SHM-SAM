import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import torch

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points

def build_sam_model(model_type: str, ckpt_p: str, device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def predict_masks_with_builded_sam(
        model,
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        box: List[List[float]]
):
    if point_coords and point_labels:
        point_coords = np.array([point_coords])
        point_labels = np.array(point_labels)
        box=None
        print(point_coords)
    else:
        print("No assigned points, using default bounding box")
        point_coords = None
        point_labels = None
        box = np.array(box)
        print(box)

    model.set_image(img)
    masks, scores, logits = model.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=False,
    )
    return masks, scores, logits


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


parser = argparse.ArgumentParser()

parser.add_argument("--input_img", type=str, required=True,
    help="Path to a single input img",
)
parser.add_argument("--box", type=int, default=[400, 5, 900, 1560], nargs="+",
    help="Path to a single input img")

parser.add_argument(
    "--point_coords", type=float, nargs='+',
    help="The coordinate of the point prompt, [coord_W coord_H].",
)

parser.add_argument(
    "--point_labels", type=int, nargs='+',
    help="The labels of the point prompt, 1 or 0.",
)

parser.add_argument(
    "--output_dir", type=str, required=True,
    help="Output path to the directory with results.",
)
parser.add_argument(
    "--sam_model_type", type=str,
    default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
    help="The type of sam model to load. Default: 'vit_h"
)
parser.add_argument(
    "--sam_ckpt", type=str, required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    print(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_img_to_array(args.input_img)
    model = build_sam_model(args.sam_model_type, args.sam_ckpt, device)

    masks, _, _ = predict_masks_with_builded_sam(
        model,
        img,
        point_coords=args.point_coords if args.point_coords else None,
        point_labels=args.point_labels if args.point_labels else None,
        box=args.box)

    masks = masks.astype(np.uint8) * 255
    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_box_p = out_dir / f"with_box.png"
        img_point_p = out_dir / f"with_point.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')


        if args.point_coords:
            show_points(plt.gca(), [args.point_coords], args.point_labels,
                    size=(width*0.04)**2)
            plt.savefig(img_point_p, bbox_inches='tight', pad_inches=0)
        else:
            show_box(args.box, plt.gca())
            plt.savefig(img_box_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()