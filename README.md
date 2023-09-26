# Instruction on enabling SAM for SHM images

## Installation
``` shell
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

## segmentation on an image
Todo: Need to test whether include dilation as post-processing
Use bounding box prompt as input:
``` commandline
python main.py --input_img frame_0626.png --box 400 5 900 1560 --output_dir ./results --sam_model_type "vit_h" --sam_ckpt sam_vit_h_4b8939.pth
```

Use point coordinates
```commandline
python main.py --input_img frame_0626.png --point_coords 800 200 --point_labels 1 --output_dir ./results --sam_model_type "vit_h" --sam_ckpt sam_vit_h_4b8939.pth
```

## auto segmentation on batch of images