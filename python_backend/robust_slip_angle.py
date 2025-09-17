#!/usr/bin/env python3
"""
robust_slip_angle.py  –  better-behaved β estimate

This script estimates the vehicle slip angle (β) from a video recorded 
with a forward-facing camera (e.g., a phone or dashcam). 

Slip angle = the difference between the vehicle's heading (where the 
nose points) and its velocity vector (where it’s actually moving).
Large slip angles occur during drifting, sliding, or cornering.

It does this by:
  1. Detecting and tracking visual features on the road between frames.
  2. Estimating the vehicle motion vector in the image plane.
  3. Converting that motion into a slip angle time series.
  4. Plotting slip angle vs time and reporting summary statistics.
  5. Saving results as CSV.
"""

import sys, cv2 as cv, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ---------- tune here -------------------------------------------------
ROI_FRACTION_Y = 0.8       # fraction of frame height to KEEP at the bottom (road region)
MAX_FEATURES   = 1000      # maximum ORB keypoints per frame
GOOD_MATCH_FRAC = 0.7      # Lowe's ratio test threshold

EMA_ALPHA      = 0.2       # exponential moving average smoothing factor
DOWNSAMPLE     = 4         # process every Nth frame
FOCAL_PX       = 1100      # focal length in pixels (unused in this code)
HEADING_OFFSET = 90.0      # adjust for camera orientation (deg)
# ----------------------------------------------------------------------

def mask_road(gray):
    """Keep only the bottom fraction of the grayscale frame (road region)."""
    h = gray.shape[0]
    roi = np.zeros_like(gray)
    roi[int(h * (1 - ROI_FRACTION_Y)):, :] = 255
    return cv.bitwise_and(gray, roi)

def exponential_smooth(prev, current, alpha=EMA_ALPHA):
    """Exponential moving average (EMA) smoother for motion vectors."""
    return alpha * current + (1 - alpha) * prev

def main(video):
    """Main slip angle estimation pipeline."""
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("Cannot open video")

    fps = cap.get(cv.CAP_PROP_FPS) / DOWNSAMPLE
    dt  = 1.0 / fps

    orb = cv.ORB_create(MAX_FEATURES)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    grabbed, prev = cap.read()
    for _ in range(DOWNSAMPLE - 1):
        cap.grab()
    if not grabbed:
        sys.exit("Empty video")

    prev_g = mask_road(cv.cvtColor(prev, cv.COLOR_BGR2GRAY))
    kp_prev, des_prev = orb.detectAndCompute(prev_g, None)

    times, betas = [], []
    t, smoothed_vec = 0.0, np.array([0., 0.])

    while True:
        # skip frames for down-sampling
        for _ in range(DOWNSAMPLE):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        gray = mask_road(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or des_prev is None:
            kp_prev, des_prev = kp, des
            t += dt
            continue

        matches = bf.knnMatch(des_prev, des, k=2)
        good = [m for m, n in matches if m.distance < GOOD_MATCH_FRAC * n.distance]

        if len(good) < 6:
            kp_prev, des_prev = kp, des
            t += dt
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts_cur  = np.float32([kp[m.trainIdx].pt  for m in good])

        M, _ = cv.estimateAffinePartial2D(
            pts_prev, pts_cur,
            method=cv.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995
        )
        if M is None:
            kp_prev, des_prev = kp, des
            t += dt
            continue

        dx, dy = M[0, 2], M[1, 2]
        vec = -np.array([dx, dy])  # invert sign
        smoothed_vec = exponential_smooth(smoothed_vec, vec)

        beta_rad = np.arctan2(smoothed_vec[0], smoothed_vec[1])
        beta_deg = np.degrees(beta_rad) + HEADING_OFFSET

        betas.append(beta_deg)
        times.append(t)
        t += dt

        kp_prev, des_prev = kp, des

    cap.release()

    if not betas:
        sys.exit("No usable data")

    # ---------- sanity check -----------------------------------------------
    assert len(times) == len(betas)

    # ---------- plot slip angle time series --------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(times, betas)
    plt.axhline(0, color='k', ls='--', lw=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Slip β [deg]")
    plt.title(f"Robust slip angle – {Path(video).name}")
    plt.grid(ls=':')
    plt.tight_layout()
    plt.show()

    # ---------- print summary stats ----------------------------------------
    print(f"mean |β| = {np.mean(np.abs(betas)):.2f}°,  "
          f"peak |β| = {np.max(np.abs(betas)):.2f}°")

    return times, betas

# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python robust_slip_angle.py <video>")

    video_path = sys.argv[1]
    times, betas = main(video_path)

    stem = Path(video_path).with_suffix("").name
    csv_path = Path(f"{stem}_slip_angle.csv")

    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "beta_deg"])
        writer.writerows(zip(times, betas))

    print(f"Saved datapoints to: {csv_path}")
