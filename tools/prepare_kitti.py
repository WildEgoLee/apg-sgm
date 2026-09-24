#!/usr/bin/env python3
"""
KITTI 2015 Stereo Dataset Preparation Tool for APG-SGM.

Extracts / converts KITTI 2015 stereo pairs into the format expected by APG-SGM:
- Left & right images -> PGM (grayscale P5)
- Ground truth disparity -> Float PFM (single channel 'Pf', disp / 256.0)
- Non-occluded mask -> PGM (P5, 255 = visible/matchable, 0 = occluded)
- Generates manifest with format:
    case_name left_path right_path gt_path dmax dmin vis_path
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image


def write_pfm(file_path: Path, img: np.ndarray):
    """Write 2D float32 numpy array as 1-channel PFM file."""
    h, w = img.shape
    with open(file_path, "wb") as f:
        f.write(b"Pf\n")
        f.write(f"{w} {h}\n".encode("latin-1"))
        f.write(b"-1.0\n")  # Little-endian
        flipped = np.flipud(img).astype(np.float32)
        f.write(flipped.tobytes())


def write_pgm(file_path: Path, img: np.ndarray):
    """Write 2D uint8 numpy array as binary PGM (P5)."""
    h, w = img.shape
    with open(file_path, "wb") as f:
        header = f"P5\n{w} {h}\n255\n".encode("latin-1")
        f.write(header)
        f.write(img.astype(np.uint8).tobytes())


def main():
    parser = argparse.ArgumentParser(description="KITTI 2012/2015 Dataset Preparation Tool")
    parser.add_argument("--in_dir", type=str, required=True, help="Directory of KITTI training set (containing image_2/3 or colored_0/1, and disp_occ/disp_noc)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: benchmarks/data/kitti<year>)")
    parser.add_argument("--manifest", type=str, default="", help="Manifest output path (default: benchmarks/manifests/kitti<year>.txt)")
    parser.add_argument("--max_pairs", type=int, default=0, help="Maximum number of pairs to process (0 = all pairs, default: 0)")
    parser.add_argument("--dmax", type=int, default=192, help="Minimum max disparity search range for KITTI (default 192)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)

    # Detect KITTI 2015 vs KITTI 2012 layout
    if (in_dir / "disp_occ_0").exists():
        version = "2015"
        case_prefix = "kitti15"
        img_l_dir = in_dir / "image_2"
        img_r_dir = in_dir / "image_3"
        disp_occ_dir = in_dir / "disp_occ_0"
        disp_noc_dir = in_dir / "disp_noc_0"
    elif (in_dir / "disp_occ").exists():
        version = "2012"
        case_prefix = "kitti12"
        img_l_dir = (in_dir / "colored_0") if (in_dir / "colored_0").exists() else (in_dir / "image_0")
        img_r_dir = (in_dir / "colored_1") if (in_dir / "colored_1").exists() else (in_dir / "image_1")
        disp_occ_dir = in_dir / "disp_occ"
        disp_noc_dir = in_dir / "disp_noc"
    else:
        raise FileNotFoundError(f"Could not identify KITTI 2012 or 2015 layout in {in_dir} (expected disp_occ_0 or disp_occ)")

    default_out = f"benchmarks/data/kitti{version}"
    default_manifest = f"benchmarks/manifests/kitti{version}.txt"

    out_dir = Path(args.out_dir) if args.out_dir else Path(default_out)
    manifest_path = Path(args.manifest) if args.manifest else Path(default_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fail-fast validation checks: require all four essential directories
    if not img_l_dir.exists():
        raise FileNotFoundError(f"Missing required KITTI left image directory: {img_l_dir}")
    if not img_r_dir.exists():
        raise FileNotFoundError(f"Missing required KITTI right image directory: {img_r_dir}")
    if not disp_occ_dir.exists():
        raise FileNotFoundError(f"Missing required KITTI ground truth disparity directory: {disp_occ_dir}")
    if not disp_noc_dir.exists():
        raise FileNotFoundError(f"Missing required KITTI non-occluded disparity directory: {disp_noc_dir}")

    left_files = sorted(list(img_l_dir.glob("*_10.png")))
    if not left_files:
        left_files = sorted(list(img_l_dir.glob("*.png")))

    if not left_files:
        raise FileNotFoundError(f"No PNG image files found in {img_l_dir}")

    selected_files = left_files[: args.max_pairs] if args.max_pairs > 0 else left_files

    manifest_lines = [
        f"# APG-SGM KITTI {version} Dataset Manifest",
        f"# Total pairs: {len(selected_files)}",
        "# Format: case_name left_img right_img gt_disp [dmax] [dmin] [vis_mask]",
    ]

    for lf in selected_files:
        pair_id = lf.stem
        case_name = f"{case_prefix}_{pair_id}"
        rf = img_r_dir / lf.name
        if not rf.exists():
            raise FileNotFoundError(f"Missing corresponding right image for {lf.name}: {rf}")

        gt_path = disp_occ_dir / lf.name
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing required GT file (disp_occ) for {lf.name}: {gt_path}")

        noc_path = disp_noc_dir / lf.name
        if not noc_path.exists():
            raise FileNotFoundError(f"Missing required non-occluded GT file (disp_noc) for {lf.name}: {noc_path}")

        print(f"Processing KITTI pair: {case_name}...")
        im_l = Image.open(lf).convert("L")
        im_r = Image.open(rf).convert("L")
        w, h = im_l.size

        lp_out = out_dir / f"{case_name}_left.pgm"
        rp_out = out_dir / f"{case_name}_right.pgm"
        write_pgm(lp_out, np.array(im_l, dtype=np.uint8))
        write_pgm(rp_out, np.array(im_r, dtype=np.uint8))

        # Disparity GT conversion (KITTI stores 16-bit uint PNG, disp = val / 256.0, 0 = invalid)
        gp_out = out_dir / f"{case_name}_gt.pfm"
        disp_png = np.array(Image.open(gt_path), dtype=np.float32)
        disp_gt = np.where(disp_png > 0, disp_png / 256.0, -1.0)
        write_pfm(gp_out, disp_gt)

        # Calculate max valid GT disparity to prevent censoring
        valid_disp = disp_gt[disp_gt > 0]
        if len(valid_disp) == 0:
            raise RuntimeError(f"No valid ground truth pixels found in {gt_path}")
        max_gt_d = float(np.max(valid_disp))
        # Compute safe dmax aligned to multiple of 16, bounded below by args.dmax
        scene_dmax = max(args.dmax, int(np.ceil((max_gt_d + 1.0) / 16.0)) * 16)

        # Visibility mask (non-occluded ground truth)
        vp_out = out_dir / f"{case_name}_vis.pgm"
        noc_png = np.array(Image.open(noc_path), dtype=np.uint16)
        vis_arr = np.where(noc_png > 0, 255, 0).astype(np.uint8)
        write_pgm(vp_out, vis_arr)

        rel_lp = os.path.relpath(lp_out, manifest_path.parent).replace("\\", "/")
        rel_rp = os.path.relpath(rp_out, manifest_path.parent).replace("\\", "/")
        rel_gp = os.path.relpath(gp_out, manifest_path.parent).replace("\\", "/")
        rel_vp = os.path.relpath(vp_out, manifest_path.parent).replace("\\", "/")

        manifest_lines.append(f"{case_name} {rel_lp} {rel_rp} {rel_gp} {scene_dmax} 0 {rel_vp}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"\nManifest successfully written to: {manifest_path} ({len(manifest_lines)-3} pairs)")


if __name__ == "__main__":
    main()
