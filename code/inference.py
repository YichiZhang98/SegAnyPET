import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from segment_anything.build_sam3D import sam_model_registry3D
from utils.infer_utils import validate_paired_img_gt


def main():
    parser = argparse.ArgumentParser(description="SegAnyPET Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input PET image (.nii.gz)")
    parser.add_argument("--label", type=str, required=True,
                        help="Path to ground-truth label (.nii.gz) for prompt generation")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save prediction (.nii.gz)")
    parser.add_argument("--num_clicks", type=int, default=5,
                        help="Number of prompt clicks per category (default: 5)")
    parser.add_argument("--crop_size", type=int, default=128,
                        help="ROI crop size (default: 128)")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5],
                        help="Target resampling spacing (default: 1.5 1.5 1.5)")
    parser.add_argument("--seed", type=int, default=233,
                        help="Random seed (default: 233)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device index (default: 0)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint}")
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Device: {device}")

    print(f"Image: {args.image}")
    print(f"Label: {args.label}")
    print(f"Num clicks: {args.num_clicks}")

    t0 = time.time()
    validate_paired_img_gt(
        model=model,
        img_path=args.image,
        gt_path=args.label,
        output_path=args.output,
        num_clicks=args.num_clicks,
        crop_size=args.crop_size,
        target_spacing=tuple(args.target_spacing),
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"Prediction saved to: {args.output}")
    print(f"Elapsed time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
