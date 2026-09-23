#!/usr/bin/env python3
"""Dataset preparation and synthetic benchmark generator for apg-sgm.

This script can:
1. Generate synthetic benchmark stereo pairs with ground truth PFM (no external dependencies required).
2. Generate manifest files for Middlebury 2014 and KITTI 2015 datasets.
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


def generate_synthetic_scene(out_dir: Path, name: str, width: int = 160, height: int = 120, max_d: int = 32):
    """Generates a textured synthetic scene with planar ramps and foreground blocks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = [0.0] * (width * height)
    left_pix = bytearray(width * height)
    right_pix = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            # Background slanted ground plane: disp 4 to 12
            d = 4.0 + 8.0 * (y / max(1, height - 1))

            # Foreground square block: disp 20
            if 35 <= y <= 75 and 45 <= x <= 95:
                d = 20.0

            # Elevated thin pillar: disp 26
            if 20 <= y <= 90 and 115 <= x <= 130:
                d = 26.0

            gt[y * width + x] = d

            # High frequency pseudo-random textured pattern
            val = (
                128.0
                + 50.0 * math.sin(x * 0.35)
                + 35.0 * math.cos(y * 0.45)
                + 30.0 * math.sin((x * 13 + y * 7) * 0.1)
                + ((x * 37 + y * 19) % 31)
            )
            val = max(0, min(255, int(val)))
            left_pix[y * width + x] = val

    # Right image sampled by shifting left by disparity
    for y in range(height):
        for x in range(width):
            d = gt[y * width + x]
            src_x = x - d
            if 0 <= src_x < width - 1:
                x0 = int(src_x)
                x1 = x0 + 1
                alpha = src_x - x0
                v0 = left_pix[y * width + x0]
                v1 = left_pix[y * width + x1]
                right_pix[y * width + x] = int((1.0 - alpha) * v0 + alpha * v1 + 0.5)
            elif 0 <= src_x < width:
                right_pix[y * width + x] = left_pix[y * width + int(src_x)]
            else:
                right_pix[y * width + x] = 128

    left_path = out_dir / f"{name}_left.pgm"
    right_path = out_dir / f"{name}_right.pgm"
    gt_path = out_dir / f"{name}_gt.pfm"

    write_pgm(left_path, width, height, bytes(left_pix))
    write_pgm(right_path, width, height, bytes(right_pix))
    write_pfm(gt_path, width, height, gt)

    return left_path, right_path, gt_path


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
        ("synth_blocks", 160, 120, 32),
        ("synth_large", 240, 160, 48),
    ]

    manifest_lines = [
        "# APG-SGM Dataset Manifest",
        "# Format: case_name left_img right_img gt_disp [dmax] [dmin]",
    ]

    for name, w, h, dmax in cases:
        print(f"Generating synthetic scene: {name} ({w}x{h}, dmax={dmax})...")
        lp, rp, gp = generate_synthetic_scene(out_dir, name, w, h, dmax)
        # Relative path from manifest directory
        rel_lp = os.path.relpath(lp, manifest_path.parent).replace("\\", "/")
        rel_rp = os.path.relpath(rp, manifest_path.parent).replace("\\", "/")
        rel_gp = os.path.relpath(gp, manifest_path.parent).replace("\\", "/")
        manifest_lines.append(f"{name} {rel_lp} {rel_rp} {rel_gp} {dmax} 0")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
