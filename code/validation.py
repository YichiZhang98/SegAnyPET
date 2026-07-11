import argparse
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from segment_anything.build_sam3D import sam_model_registry3D
from utils.infer_utils import validate_paired_img_gt, read_arr_from_nifti
from utils.metric_utils import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="SegAnyPET Validation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Root directory of test data (contains imagesTs/labelsTs or similar)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save predictions and metrics (default: ./results)")
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
    parser.add_argument("--skip_existing_pred", action="store_true", default=False,
                        help="Skip inference if prediction file already exists")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="Save prediction NIfTI files (default: True)")
    parser.add_argument("--data_type", type=str, default="Ts",
                        help="Data type suffix: Ts, Val, or Tr (default: Ts)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading model from: {args.checkpoint}")
    model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Model loaded. Device: {device}")

    image_label_pairs = collect_image_label_pairs(args.test_data_path, args.data_type)
    if not image_label_pairs:
        print(f"No image-label pairs found in {args.test_data_path} with data_type={args.data_type}")
        return

    print(f"Found {len(image_label_pairs)} cases to validate.")

    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_dice_list = []
    all_nsd_list = []
    per_case_results = OrderedDict()
    per_class_results = defaultdict(lambda: {"dsc": [], "nsd": []})

    total_time = 0.0

    for i, (img_path, label_path) in enumerate(tqdm(image_label_pairs, desc="Validation")):
        case_name = os.path.basename(label_path)
        pred_path = os.path.join(pred_dir, case_name)

        if args.skip_existing_pred and os.path.exists(pred_path):
            print(f"  [{i+1}/{len(image_label_pairs)}] Skipping (exists): {case_name}")
        else:
            t0 = time.time()
            try:
                validate_paired_img_gt(
                    model=model,
                    img_path=img_path,
                    gt_path=label_path,
                    output_path=pred_path,
                    num_clicks=args.num_clicks,
                    crop_size=args.crop_size,
                    target_spacing=tuple(args.target_spacing),
                    seed=args.seed,
                )
                elapsed = time.time() - t0
                total_time += elapsed
            except Exception as e:
                print(f"  [{i+1}/{len(image_label_pairs)}] ERROR on {case_name}: {e}")
                continue

        if not os.path.exists(pred_path):
            continue

        try:
            case_metrics = compute_metrics(label_path, pred_path)
        except Exception as e:
            print(f"  Metric computation failed for {case_name}: {e}")
            continue

        case_dsc_list = []
        case_nsd_list = []
        per_case_results[case_name] = {}

        for class_id_str, metrics in case_metrics.items():
            class_id = int(class_id_str)
            dsc_val = metrics.get("dsc", np.nan)
            nsd_val = metrics.get("nsd", np.nan)

            per_case_results[case_name][class_id_str] = {"dsc": dsc_val, "nsd": nsd_val}
            per_class_results[class_id]["dsc"].append(dsc_val)
            per_class_results[class_id]["nsd"].append(nsd_val)

            if not np.isnan(dsc_val):
                case_dsc_list.append(dsc_val)
            if not np.isnan(nsd_val):
                case_nsd_list.append(nsd_val)

        case_mean_dsc = np.mean(case_dsc_list) if case_dsc_list else np.nan
        case_mean_nsd = np.mean(case_nsd_list) if case_nsd_list else np.nan
        all_dice_list.append(case_mean_dsc)
        all_nsd_list.append(case_mean_nsd)

        tqdm.write(f"  [{i+1}/{len(image_label_pairs)}] {case_name} "
                   f"DSC={case_mean_dsc:.4f} NSD={case_mean_nsd:.4f}")

    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)

    valid_dice = [d for d in all_dice_list if not np.isnan(d)]
    valid_nsd = [n for n in all_nsd_list if not np.isnan(n)]

    mean_dice = np.mean(valid_dice) if valid_dice else np.nan
    mean_nsd = np.mean(valid_nsd) if valid_nsd else np.nan

    print(f"\nOverall Mean DSC: {mean_dice:.4f}")
    print(f"Overall Mean NSD: {mean_nsd:.4f}")
    print(f"Total cases: {len(image_label_pairs)}, Evaluated: {len(valid_dice)}")
    print(f"Total inference time: {total_time:.1f}s")

    if per_class_results:
        print(f"\n{'Class':<10} {'DSC (mean+/-std)':<20} {'NSD (mean+/-std)':<20} {'Count':<8}")
        print("-" * 60)
        for class_id in sorted(per_class_results.keys()):
            dsc_arr = np.array(per_class_results[class_id]["dsc"])
            nsd_arr = np.array(per_class_results[class_id]["nsd"])
            dsc_mean = np.nanmean(dsc_arr)
            dsc_std = np.nanstd(dsc_arr)
            nsd_mean = np.nanmean(nsd_arr)
            nsd_std = np.nanstd(nsd_arr)
            count = len(dsc_arr)
            print(f"{class_id:<10} {dsc_mean:.4f}+/-{dsc_std:.4f}       "
                  f"{nsd_mean:.4f}+/-{nsd_std:.4f}       {count}")

    results_json = {
        "config": {
            "checkpoint": args.checkpoint,
            "test_data_path": args.test_data_path,
            "num_clicks": args.num_clicks,
            "crop_size": args.crop_size,
            "target_spacing": list(args.target_spacing),
            "seed": args.seed,
        },
        "overall": {
            "mean_dsc": float(mean_dice) if not np.isnan(mean_dice) else None,
            "mean_nsd": float(mean_nsd) if not np.isnan(mean_nsd) else None,
            "num_cases": len(image_label_pairs),
            "num_evaluated": len(valid_dice),
            "total_time_seconds": total_time,
        },
        "per_class": {
            str(k): {
                "mean_dsc": float(np.nanmean(v["dsc"])),
                "std_dsc": float(np.nanstd(v["dsc"])),
                "mean_nsd": float(np.nanmean(v["nsd"])),
                "std_nsd": float(np.nanstd(v["nsd"])),
                "count": len(v["dsc"]),
            }
            for k, v in sorted(per_class_results.items())
        },
        "per_case": per_case_results,
    }

    json_path = os.path.join(args.output_dir, "validation_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)
    print(f"\nResults saved to: {json_path}")
    print(f"Predictions saved to: {pred_dir}")


def collect_image_label_pairs(root_path, data_type="Ts"):
    """
    Collect image-label pairs from the data directory.
    Supports two directory structures:
      1) root_path/imagesXX/ and root_path/labelsXX/ (nnUNet-style)
      2) root_path/*/imagesXX/ and root_path/*/labelsXX/ (nested datasets)
    """
    pairs = []

    label_dir = os.path.join(root_path, f"labels{data_type}")
    image_dir = os.path.join(root_path, f"images{data_type}")
    if os.path.isdir(label_dir) and os.path.isdir(image_dir):
        pairs.extend(_scan_dir_pair(image_dir, label_dir))

    if not pairs:
        for sub in sorted(os.listdir(root_path)):
            sub_path = os.path.join(root_path, sub)
            if not os.path.isdir(sub_path):
                continue
            label_dir = os.path.join(sub_path, f"labels{data_type}")
            image_dir = os.path.join(sub_path, f"images{data_type}")
            if os.path.isdir(label_dir) and os.path.isdir(image_dir):
                pairs.extend(_scan_dir_pair(image_dir, label_dir))

    if not pairs:
        for sub in sorted(os.listdir(root_path)):
            sub_path = os.path.join(root_path, sub)
            if not os.path.isdir(sub_path):
                continue
            for sub2 in sorted(os.listdir(sub_path)):
                sub2_path = os.path.join(sub_path, sub2)
                if not os.path.isdir(sub2_path):
                    continue
                label_dir = os.path.join(sub2_path, f"labels{data_type}")
                image_dir = os.path.join(sub2_path, f"images{data_type}")
                if os.path.isdir(label_dir) and os.path.isdir(image_dir):
                    pairs.extend(_scan_dir_pair(image_dir, label_dir))

    return sorted(pairs, key=lambda x: x[1])


def _scan_dir_pair(image_dir, label_dir):
    pairs = []
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        label_path = os.path.join(label_dir, fname)
        image_path = os.path.join(image_dir, fname)
        if os.path.exists(image_path):
            pairs.append((image_path, label_path))
    return pairs


if __name__ == "__main__":
    main()
