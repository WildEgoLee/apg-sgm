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
    parser = argparse.ArgumentParser(description="KITTI 2015 Dataset Preparation Tool")
    parser.add_argument("--in_dir", type=str, required=True, help="Directory of KITTI 2015 training set (containing image_2, image_3, disp_occ_0, disp_noc_0)")
    parser.add_argument("--out_dir", type=str, default="benchmarks/data/kitti2015", help="Output directory")
    parser.add_argument("--manifest", type=str, default="benchmarks/manifests/kitti2015.txt", help="Manifest output path")
    parser.add_argument("--max_pairs", type=int, default=20, help="Maximum number of pairs to process (default 20)")
    parser.add_argument("--dmax", type=int, default=192, help="Max disparity search range for KITTI (default 192)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    img2_dir = in_dir / "image_2"
    img3_dir = in_dir / "image_3"
    disp_occ_dir = in_dir / "disp_occ_0"
    disp_noc_dir = in_dir / "disp_noc_0"

    if not img2_dir.exists() or not img3_dir.exists():
        print(f"Error: {img2_dir} or {img3_dir} does not exist.")
        return

    left_files = sorted(list(img2_dir.glob("*_10.png")))
    if not left_files:
        left_files = sorted(list(img2_dir.glob("*.png")))

    selected_files = left_files[: args.max_pairs] if args.max_pairs > 0 else left_files

    manifest_lines = [
        "# APG-SGM KITTI 2015 Dataset Manifest",
        f"# Total pairs: {len(selected_files)}",
        "# Format: case_name left_img right_img gt_disp [dmax] [dmin] [vis_mask]",
    ]

    for lf in selected_files:
        pair_id = lf.stem
        case_name = f"kitti15_{pair_id}"
        rf = img3_dir / lf.name
        if not rf.exists():
            continue

        print(f"Processing KITTI pair: {case_name}...")
        im_l = Image.open(lf).convert("L")
        im_r = Image.open(rf).convert("L")
        w, h = im_l.size

        lp_out = out_dir / f"{case_name}_left.pgm"
        rp_out = out_dir / f"{case_name}_right.pgm"
        write_pgm(lp_out, np.array(im_l, dtype=np.uint8))
        write_pgm(rp_out, np.array(im_r, dtype=np.uint8))

        # Disparity GT conversion (KITTI stores 16-bit uint PNG, disp = val / 256.0, 0 = invalid)
        gt_path = disp_occ_dir / lf.name
        gp_out = out_dir / f"{case_name}_gt.pfm"
        has_gt = gt_path.exists()
        if has_gt:
            disp_png = np.array(Image.open(gt_path), dtype=np.float32)
            disp_gt = np.where(disp_png > 0, disp_png / 256.0, -1.0)
            write_pfm(gp_out, disp_gt)

        # Visibility mask (non-occluded ground truth)
        noc_path = disp_noc_dir / lf.name
        vp_out = out_dir / f"{case_name}_vis.pgm"
        has_vis = noc_path.exists()
        if has_vis:
            noc_png = np.array(Image.open(noc_path), dtype=np.uint16)
            vis_arr = np.where(noc_png > 0, 255, 0).astype(np.uint8)
            write_pgm(vp_out, vis_arr)
        elif has_gt:
            vis_arr = np.where(disp_gt >= 0, 255, 0).astype(np.uint8)
            write_pgm(vp_out, vis_arr)
            has_vis = True

        rel_lp = os.path.relpath(lp_out, manifest_path.parent).replace("\\", "/")
        rel_rp = os.path.relpath(rp_out, manifest_path.parent).replace("\\", "/")
        rel_gp = os.path.relpath(gp_out, manifest_path.parent).replace("\\", "/") if has_gt else ""
        rel_vp = os.path.relpath(vp_out, manifest_path.parent).replace("\\", "/") if has_vis else ""

        manifest_lines.append(f"{case_name} {rel_lp} {rel_rp} {rel_gp} {args.dmax} 0 {rel_vp}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"\nManifest successfully written to: {manifest_path} ({len(manifest_lines)-3} pairs)")


if __name__ == "__main__":
    main()
