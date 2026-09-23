#!/usr/bin/env python3
"""
Middlebury 2014 Stereo Dataset Preparation Tool for APG-SGM.

Extracts / converts Middlebury 2014 stereo pairs into the format expected by APG-SGM:
- Left & right images -> PGM (grayscale P5) or PPM (color P6)
- Ground truth disparity -> Float PFM (single channel 'Pf')
- Non-occluded mask -> PGM (P5, 255 = visible/matchable, 0 = occluded)
- Generates manifest with format:
    case_name left_path right_path gt_path dmax dmin vis_path
"""

import argparse
import os
import re
from pathlib import Path
import numpy as np
from PIL import Image


def read_calib(calib_path: Path):
    """Parse Middlebury calib.txt file for disparity search bounds and image dimensions."""
    params = {}
    if not calib_path.exists():
        return params
    with open(calib_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                try:
                    if "." in v:
                        params[k] = float(v)
                    else:
                        params[k] = int(v)
                except ValueError:
                    params[k] = v
    return params


def read_pfm(file_path: Path):
    """Read 1-channel PFM file."""
    with open(file_path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        if header != "Pf":
            raise ValueError(f"Expected single channel PFM ('Pf'), got '{header}'")
        dim_line = f.readline().decode("latin-1").strip()
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("latin-1").strip()
        w, h = map(int, dim_line.split())
        scale_line = f.readline().decode("latin-1").strip()
        scale = float(scale_line)
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        data = np.reshape(data, (h, w))
        # PFM stores rows from bottom to top, flip vertically
        return np.flipud(data).copy()


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


def process_scene(scene_dir: Path, out_dir: Path, scale: float = 1.0, color: bool = False):
    """Convert one Middlebury scene directory."""
    scene_name = scene_dir.name
    calib = read_calib(scene_dir / "calib.txt")

    im0_path = scene_dir / "im0.png"
    im1_path = scene_dir / "im1.png"
    disp0_path = scene_dir / "disp0GT.pfm"
    if not disp0_path.exists():
        disp0_path = scene_dir / "disp0.pfm"
    mask_path = scene_dir / "mask0nocc.png"

    if not im0_path.exists() or not im1_path.exists():
        print(f"Skipping {scene_name}: im0.png or im1.png not found")
        return None

    im0 = Image.open(im0_path)
    im1 = Image.open(im1_path)

    orig_w, orig_h = im0.size
    target_w = int(round(orig_w * scale))
    target_h = int(round(orig_h * scale))

    if scale != 1.0:
        im0 = im0.resize((target_w, target_h), Image.Resampling.BILINEAR)
        im1 = im1.resize((target_w, target_h), Image.Resampling.BILINEAR)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Disparity search range from calib:
    # ndisp is the conservative disparity upper bound (algorithms search 0 .. ndisp-1).
    # vmin/vmax are tight GT visualization limits and must NOT be used for dmin.
    ndisp = int(calib.get("ndisp", 128))
    dmin = 0
    dmax = int(np.ceil(ndisp * scale))
    # Align dmax to multiple of 16 for efficient packed/SGM architectures
    dmax_aligned = int(np.ceil(dmax / 16.0) * 16)

    # Write Left & Right images
    suffix = f"_{int(round(scale*100))}" if scale != 1.0 else ""
    left_out = out_dir / f"{scene_name}{suffix}_left.pgm"
    right_out = out_dir / f"{scene_name}{suffix}_right.pgm"

    im0_gray = np.array(im0.convert("L"), dtype=np.uint8)
    im1_gray = np.array(im1.convert("L"), dtype=np.uint8)
    write_pgm(left_out, im0_gray)
    write_pgm(right_out, im1_gray)

    gt_out = out_dir / f"{scene_name}{suffix}_gt.pfm"
    has_gt = disp0_path.exists()
    if has_gt:
        disp_arr = read_pfm(disp0_path)
        if scale != 1.0:
            # Resize disparity map and scale disparity values
            disp_img = Image.fromarray(disp_arr)
            disp_resized = np.array(disp_img.resize((target_w, target_h), Image.Resampling.BILINEAR), dtype=np.float32)
            disp_resized[disp_resized > 1e4] = -1.0
            disp_resized[disp_resized >= 0.0] *= scale
            disp_arr = disp_resized
        else:
            disp_arr[disp_arr > 1e4] = -1.0
        write_pfm(gt_out, disp_arr)

    vis_out = out_dir / f"{scene_name}{suffix}_vis.pgm"
    has_vis = mask_path.exists()
    if has_vis:
        mask_img = Image.open(mask_path).convert("L")
        if scale != 1.0:
            mask_img = mask_img.resize((target_w, target_h), Image.Resampling.NEAREST)
        mask_arr = np.array(mask_img, dtype=np.uint8)
        write_pgm(vis_out, mask_arr)
    elif has_gt:
        # Generate default visibility: matchable and valid GT
        vis_arr = np.zeros((target_h, target_w), dtype=np.uint8)
        for y in range(target_h):
            for x in range(target_w):
                d = disp_arr[y, x]
                if d >= 0 and (x - int(round(d))) >= 0:
                    vis_arr[y, x] = 255
        write_pgm(vis_out, vis_arr)
        has_vis = True

    return {
        "name": f"{scene_name}{suffix}",
        "left": left_out,
        "right": right_out,
        "gt": gt_out if has_gt else None,
        "dmax": dmax_aligned,
        "dmin": dmin,
        "vis": vis_out if has_vis else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Middlebury 2014 Dataset Preparation Tool")
    parser.add_argument("--in_dir", type=str, required=True, help="Directory containing Middlebury 2014 scenes")
    parser.add_argument("--out_dir", type=str, default="benchmarks/data/middlebury2014", help="Output directory")
    parser.add_argument("--manifest", type=str, default="benchmarks/manifests/middlebury2014.txt", help="Manifest output path")
    parser.add_argument("--scale", type=float, default=0.25, help="Scale factor (e.g., 0.25 for quarter res, 0.5 for half res, 1.0 for full)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    scene_dirs = [p for p in in_dir.iterdir() if p.is_dir() and (p / "im0.png").exists()]
    if not scene_dirs:
        # Check if in_dir itself is a single scene
        if (in_dir / "im0.png").exists():
            scene_dirs = [in_dir]
        else:
            print(f"No Middlebury scene directories found in {in_dir}")
            return

    manifest_lines = [
        "# APG-SGM Middlebury 2014 Dataset Manifest",
        f"# Generated with scale={args.scale}",
        "# Format: case_name left_img right_img gt_disp [dmax] [dmin] [vis_mask]",
    ]

    for sd in sorted(scene_dirs):
        print(f"Processing scene: {sd.name} (scale={args.scale})...")
        res = process_scene(sd, out_dir, scale=args.scale)
        if not res:
            continue
        rel_lp = os.path.relpath(res["left"], manifest_path.parent).replace("\\", "/")
        rel_rp = os.path.relpath(res["right"], manifest_path.parent).replace("\\", "/")
        rel_gp = os.path.relpath(res["gt"], manifest_path.parent).replace("\\", "/") if res["gt"] else ""
        rel_vp = os.path.relpath(res["vis"], manifest_path.parent).replace("\\", "/") if res["vis"] else ""
        manifest_lines.append(f"{res['name']} {rel_lp} {rel_rp} {rel_gp} {res['dmax']} {res['dmin']} {rel_vp}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"\nManifest successfully written to: {manifest_path} ({len(manifest_lines)-3} scenes)")


if __name__ == "__main__":
    main()
