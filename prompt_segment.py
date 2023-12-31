import sys
import argparse
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import torch

# sys.path.append('/home/jupyter/fastapi_debug/app/segment_anything/segment_anything')
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points


def predict_masks_with_sam(
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        box: List[List[float]],
        model_type: str,
        ckpt_p: str,
        device="cuda"
):
    point_coords = np.array(point_coords)
    point_labels = np.array(point_labels)
    # print(point_coords)
    box = None #np.array(box)

    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=box,
        multimask_output=True,
    )
    return masks, scores, logits


def build_sam_model(model_type: str, ckpt_p: str, device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

def build_sam_auto_mask(model_type: str, ckpt_p: str, device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=ckpt_p)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def predict_masks_with_builded_sam(
        model,
        img: np.ndarray,
        point_coords: List[List[float]],
        point_labels: List[int],
        box: List[List[float]]
):
    point_coords = None # np.array(point_coords)
    point_labels = None #np.array(point_labels)
    # print(point_coords)
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



def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+',
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--box", type=float, nargs='+',
        help="The coordinate of the box prompt, [x1, y1, x2, y2].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+',
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
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
    """Example usage:
    python sam_segment.py \
        --input_img FA_demo/FA1_dog.png \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_img_to_array(args.input_img)
    print(args.box)
    print(type(args.box))
    # input_box = np.array(args.box)

    model = build_sam_model(args.sam_model_type, args.sam_ckpt, device)
    # model.set_image(img)
    masks, _, _ = predict_masks_with_builded_sam(
        model,
        img,
        args.box)

    # masks, _, _ = model.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=input_box[None, :],
    #     multimask_output=False,
    # )
    masks = masks.astype(np.uint8) * 255

    # dilate mask to avoid unmasked edge effect
    # if args.dilate_kernel_size is not None:
    #     masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [args.point_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()