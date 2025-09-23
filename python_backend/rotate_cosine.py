#!/usr/bin/env python3
"""
rotate_cosine.py
----------------
Apply a cosine‑wave rotation to every frame of a video.

USAGE
-----
python rotate_cosine.py in.mp4 out.mp4 --amp 10 --period 5

Arguments
---------
in.mp4   : path to input video
out.mp4  : path to output video (same resolution & FPS)

Options
-------
--amp    : rotation amplitude in degrees   (default: 10° peak)
--period : cosine period in seconds        (default: 5 s)
--phase  : phase offset in degrees         (default: 0°)
--border : border mode: constant | reflect (default: constant)

Dependencies
------------
opencv‑python, numpy, tqdm  (tqdm only for progress bar)
"""

import cv2 as cv
import numpy as np
import argparse, math, sys
from tqdm import tqdm
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input",  help="input video")
    p.add_argument("output", help="output video")
    p.add_argument("--amp",    type=float, default=10.0,
                   help="cosine amplitude in degrees (default 10)")
    p.add_argument("--period", type=float, default=5.0,
                   help="cosine period in seconds (default 5)")
    p.add_argument("--phase",  type=float, default=0.0,
                   help="phase offset in degrees (default 0)")
    p.add_argument("--border", choices=["constant","reflect"],
                   default="constant", help="border mode (default constant)")
    return p.parse_args()

def main():
    args = parse_args()

    cap = cv.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f" Could not open {args.input}")

    fps   = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fourcc= cv.VideoWriter_fourcc(*"mp4v")   # works on most systems

    out = cv.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        sys.exit(" Could not open VideoWriter")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    omega = 2*math.pi / args.period                 # rad / s
    phase = math.radians(args.phase)                # rad
    amp   = args.amp

    border_flag = cv.BORDER_CONSTANT if args.border=="constant" else cv.BORDER_REFLECT

    with tqdm(total=total_frames, unit="frame") as bar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps                     # seconds
            angle = amp * math.cos(omega*t + phase) # degrees

            # rotation matrix around image centre
            M = cv.getRotationMatrix2D((width/2, height/2), angle, 1.0)

            rotated = cv.warpAffine(frame, M, (width, height),
                                    flags=cv.INTER_LINEAR,
                                    borderMode=border_flag,
                                    borderValue=(0,0,0))
            out.write(rotated)
            frame_idx += 1
            bar.update(1)

    cap.release()
    out.release()
    print(f" Wrote {frame_idx} frames to {Path(args.output).resolve()}")
    print(f"    Rotation: β(t) = {amp}·cos(2π t/{args.period} + {args.phase}°)")

if __name__ == "__main__":
    main()