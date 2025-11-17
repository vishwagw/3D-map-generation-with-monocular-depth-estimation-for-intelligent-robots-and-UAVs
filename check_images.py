#!/usr/bin/env python3
"""
check_images.py

Scans a frames directory for unreadable/corrupt images. Optionally attempts
to repair by re-opening and re-saving with PIL, and/or moves irreparable files
into a `bad/` folder.

Usage examples (PowerShell):
python .\check_images.py --frames .\data\frames --attempt-repair --bad-dir .\data\bad --repaired-dir .\data\repaired
"""
import argparse
import glob
import os
import shutil
from PIL import Image


def check_and_repair(path, attempt_repair, repaired_dir, bad_dir):
    try:
        # verify only checks file headers; PIL may still raise later on open
        with Image.open(path) as im:
            im.verify()
        return "ok", None
    except Exception as e_verify:
        if not attempt_repair:
            return "bad", str(e_verify)
        # Attempt to open and re-save to repaired_dir
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                os.makedirs(repaired_dir, exist_ok=True)
                outp = os.path.join(repaired_dir, os.path.basename(path))
                im.save(outp, format="PNG")
            return "repaired", None
        except Exception as e_repair:
            # move to bad_dir if provided
            try:
                if bad_dir:
                    os.makedirs(bad_dir, exist_ok=True)
                    dst = os.path.join(bad_dir, os.path.basename(path))
                    shutil.move(path, dst)
                return "moved", str(e_repair)
            except Exception as e_move:
                return "failed", f"repair_err={e_repair}; move_err={e_move}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="frames directory")
    p.add_argument("--attempt-repair", action="store_true", help="Try to repair corrupt images by re-saving with PIL")
    p.add_argument("--bad-dir", default=None, help="Directory to move irreparable/bad files into (optional)")
    p.add_argument("--repaired-dir", default=None, help="Directory to write repaired files into (optional)")
    args = p.parse_args()

    patterns = [os.path.join(args.frames, "*.png"), os.path.join(args.frames, "*.jpg"), os.path.join(args.frames, "*.jpeg")]
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)))
    if len(paths) == 0:
        print("No images found in", args.frames)
        return

    stats = {"ok": 0, "bad": 0, "repaired": 0, "moved": 0, "failed": 0}
    details = []
    for pth in paths:
        status, info = check_and_repair(pth, args.attempt_repair, args.repaired_dir or os.path.join(args.frames, "repaired"), args.bad_dir or os.path.join(args.frames, "bad"))
        stats[status] = stats.get(status, 0) + 1
        if status != "ok":
            details.append((pth, status, info))
            print(f"{status.upper()}: {pth} -> {info}")

    print("--- Summary ---")
    for k in ["ok", "repaired", "moved", "bad", "failed"]:
        print(f"{k}: {stats.get(k,0)}")
    if details:
        print("\nSample problematic files:")
        for pth, status, info in details[:20]:
            print(status, pth, info)


if __name__ == "__main__":
    main()
