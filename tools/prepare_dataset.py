#!/usr/bin/env python3
"""Dataset preparation and synthetic benchmark generator for apg-sgm.

This script generates synthetic benchmark stereo pairs with ground truth PFM
using physically correct left-to-right forward warping with z-buffering:
    x_R = x_L - d_{gt}(x_L, y)
    I_R(x_R, y) <- I_L(x_L, y)
"""

import argparse
import math
import os
import struct
import sys
from pathlib import Path


def write_pgm(path: Path, width: int, height: int, pixels: bytes):
    """Write 8-bit PGM image."""
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        f.write(pixels)


def write_pfm(path: Path, width: int, height: int, data: list):
    """Write 1-channel float PFM (bottom-to-top order)."""
    with open(path, "wb") as f:
        f.write(f"Pf\n{width} {height}\n-1.0\n".encode("ascii"))
        # PFM stores bottom scanline first (y = height-1 down to 0)
        for y in reversed(range(height)):
            row = data[y * width : (y + 1) * width]
            f.write(struct.pack(f"<{len(row)}f", *row))


def generate_synthetic_scene(out_dir: Path, name: str, width: int, height: int, scene_type: str, dmax: int = 32):
    """Generates a textured synthetic scene with exact left-to-right forward warping and z-buffering.

    Left reference frame:
        x_R = x_L - d
        Larger d corresponds to closer foreground objects (retained in z-buffer).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = [0.0] * (width * height)
    left_pix = bytearray(width * height)
    right_pix = bytearray([128] * (width * height))
    z_buf = [-1.0] * (width * height)

    for y in range(height):
        for x in range(width):
            if scene_type == "constant":
                # Constant disparity plane
                d = 8
            elif scene_type == "blocks":
                # Multi-plane depth steps: background 6, middle block 16, tall pillar 22
                d = 6
                if 35 <= y <= 75 and 45 <= x <= 95:
                    d = 16
                elif 20 <= y <= 90 and 115 <= x <= 130:
                    d = 22
            elif scene_type == "slanted":
                # Slanted ground ramp (integer steps) + elevated blocks
                d = 6 + int(12.0 * y / max(1, height - 1))
                if 40 <= y <= 110 and 60 <= x <= 130:
                    d = 26
                elif 30 <= y <= 130 and 160 <= x <= 190:
                    d = 34
            else:
                d = 8

            gt[y * width + x] = float(d)

            # High-frequency textured pattern for robust census & gradient matching
            val = (
                128.0
                + 48.0 * math.sin(x * 0.37)
                + 38.0 * math.cos(y * 0.43)
                + 32.0 * math.sin((x * 17 + y * 11) * 0.13)
                + ((x * 41 + y * 23) % 29)
            )
            left_pix[y * width + x] = max(0, min(255, int(val)))

    owner_xl = [-1] * (width * height)
    # Forward warp from left to right with z-buffer (x_R = x_L - d)
    for y in range(height):
        for xl in range(width):
            d_val = gt[y * width + xl]
            d_int = int(round(d_val))
            xr = xl - d_int
            if 0 <= xr < width:
                # Retain surface closer to camera (larger disparity)
                if d_val > z_buf[y * width + xr]:
                    z_buf[y * width + xr] = d_val
                    owner_xl[y * width + xr] = xl
                    right_pix[y * width + xr] = left_pix[y * width + xl]

    # Compute left-reference visibility mask: 255 if matchable and visible in right image
    vis_pix = bytearray(width * height)
    for y in range(height):
        for xl in range(width):
            d_val = gt[y * width + xl]
            d_int = int(round(d_val))
            xr = xl - d_int
            if 0 <= xr < width and owner_xl[y * width + xr] == xl:
                vis_pix[y * width + xl] = 255
            else:
                vis_pix[y * width + xl] = 0

    # For disoccluded / unmapped pixels in right image, synthesize background texture
    for y in range(height):
        for xr in range(width):
            if z_buf[y * width + xr] < 0:
                bg_val = (
                    110.0
                    + 40.0 * math.sin(xr * 0.37)
                    + 30.0 * math.cos(y * 0.43)
                )
                right_pix[y * width + xr] = max(0, min(255, int(bg_val)))

    left_path = out_dir / f"{name}_left.pgm"
    right_path = out_dir / f"{name}_right.pgm"
    gt_path = out_dir / f"{name}_gt.pfm"
    vis_path = out_dir / f"{name}_vis.pgm"

    write_pgm(left_path, width, height, bytes(left_pix))
    write_pgm(right_path, width, height, bytes(right_pix))
    write_pfm(gt_path, width, height, gt)
    write_pgm(vis_path, width, height, bytes(vis_pix))

    return left_path, right_path, gt_path, vis_path


def main():
    parser = argparse.ArgumentParser(description="Dataset & Manifest Preparation Tool")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test datasets")
    parser.add_argument("--out_dir", type=str, default="benchmarks/data/synthetic", help="Output directory for synthetic data")
    parser.add_argument("--manifest", type=str, default="benchmarks/manifests/synthetic.txt", help="Manifest output path")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        ("synth_plane", 160, 120, "constant", 32),
        ("synth_blocks", 160, 120, "blocks", 32),
        ("synth_slanted", 240, 160, "slanted", 48),
    ]

    manifest_lines = [
        "# APG-SGM Dataset Manifest",
        "# Format: case_name left_img right_img gt_disp [dmax] [dmin] [vis_mask]",
    ]

    for name, w, h, stype, dmax in cases:
        print(f"Generating synthetic scene: {name} ({w}x{h}, {stype}, dmax={dmax})...")
        lp, rp, gp, vp = generate_synthetic_scene(out_dir, name, w, h, stype, dmax)
        rel_lp = os.path.relpath(lp, manifest_path.parent).replace("\\", "/")
        rel_rp = os.path.relpath(rp, manifest_path.parent).replace("\\", "/")
        rel_gp = os.path.relpath(gp, manifest_path.parent).replace("\\", "/")
        rel_vp = os.path.relpath(vp, manifest_path.parent).replace("\\", "/")
        manifest_lines.append(f"{name} {rel_lp} {rel_rp} {rel_gp} {dmax} 0 {rel_vp}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
