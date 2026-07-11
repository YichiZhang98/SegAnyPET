import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from segment_anything.build_sam3D import sam_model_registry3D
from utils.infer_utils import validate_paired_img_gt


def main():
    parser = argparse.ArgumentParser(description="SegAnyPET Batch Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing input PET images (.nii.gz)")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Directory containing ground-truth labels (.nii.gz)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predictions (.nii.gz)")
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

    os.makedirs(args.output_dir, exist_ok=True)

    label_files = sorted(glob.glob(os.path.join(args.label_dir, "*.nii.gz")))
    if not label_files:
        print(f"No .nii.gz files found in {args.label_dir}")
        return

    print(f"Found {len(label_files)} cases to process.")
    total_time = 0

    for i, label_path in enumerate(label_files):
        filename = os.path.basename(label_path)
        image_path = os.path.join(args.image_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        if not os.path.exists(image_path):
            print(f"[{i+1}/{len(label_files)}] SKIP (image not found): {filename}")
            continue

        print(f"[{i+1}/{len(label_files)}] Processing: {filename}")
        t0 = time.time()

        try:
            validate_paired_img_gt(
                model=model,
                img_path=image_path,
                gt_path=label_path,
                output_path=output_path,
                num_clicks=args.num_clicks,
                crop_size=args.crop_size,
                target_spacing=tuple(args.target_spacing),
                seed=args.seed,
            )
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"    Done in {elapsed:.1f}s -> {output_path}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\nBatch inference complete. Total time: {total_time:.1f}s")
    print(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
