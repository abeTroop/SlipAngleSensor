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
"""

import sys, cv2 as cv, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ---------- tune here -------------------------------------------------
# These constants let you adapt the algorithm to your video setup.

ROI_FRACTION_Y = .8       # fraction of frame height to KEEP at the bottom (road region).
                          # originally 0.33 (bottom 33%), now 0.8 (bottom 80%).
                          # Smaller = more strict focus on road, less noise from background.

MAX_FEATURES   = 1000     # maximum number of ORB keypoints to detect per frame
GOOD_MATCH_FRAC = 0.7     # Lowe's ratio test threshold: smaller = stricter matches

EMA_ALPHA      = 0.2      # exponential moving average smoothing factor
                          # 0.1 ≈ averaging over ~10 frames

#edge this 

DOWNSAMPLE     = 4        # downsampling factor: skip frames to reduce noise/CPU
                          # e.g. 4 = keep 1 in 4 frames. Here, no downsampling.

FOCAL_PX       = 1100     # focal length in pixels (from camera intrinsics, unused in this code)
HEADING_OFFSET = 90.0     # adjust for camera mounting orientation (deg)
                          # e.g. if camera rotated clockwise, add +90 deg
# ----------------------------------------------------------------------

def mask_road(gray):
    """
    Keep only the bottom portion of the grayscale frame (Region of Interest).
    This assumes the road surface appears at the bottom of the image.

    Parameters:
        gray (ndarray): input grayscale frame.

    Returns:
        masked grayscale frame with only the bottom ROI visible.
    """
    h = gray.shape[0]  # frame height in pixels
    roi = np.zeros_like(gray)  # black mask
    roi[int(h * (1 - ROI_FRACTION_Y)):, :] = 255  # white rectangle covering bottom fraction
    return cv.bitwise_and(gray, roi)  # keep road region only


def exponential_smooth(prev, current, alpha=EMA_ALPHA):
    """
    Exponential moving average (EMA) smoother.

    Smooths noisy motion vectors so that slip angle estimate is less jittery.

    Parameters:
        prev (ndarray): previously smoothed vector.
        current (ndarray): current raw motion vector.
        alpha (float): smoothing factor (0 < alpha ≤ 1).

    Returns:
        ndarray: updated smoothed vector.
    """
    return alpha * current + (1 - alpha) * prev


def main(video):
    """
    Main slip angle estimation pipeline.

    Steps:
      1. Open video and prepare ORB + BFMatcher.
      2. On each frame (or downsampled frame), detect ORB features on the road.
      3. Match features between consecutive frames.
      4. Estimate camera translation (dx, dy) via RANSAC affine fit.
      5. Invert translation → get vehicle motion vector.
      6. Compute slip angle β = arctan2(lateral, forward).
      7. Plot slip angle vs time and print summary statistics.

    Parameters:
        video (str): path to input video file.
    """

    # ---------- open video ---------------------------------------------------
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("Cannot open video")

    # compute effective framerate after downsampling
    fps = cap.get(cv.CAP_PROP_FPS) / DOWNSAMPLE
    dt  = 1.0 / fps  # time step per processed frame

    # feature detector and matcher
    orb = cv.ORB_create(MAX_FEATURES)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    # ---------- process first frame ------------------------------------------
    grabbed, prev = cap.read()
    for _ in range(DOWNSAMPLE - 1):  # skip extra frames if downsampling
        cap.grab()
    if not grabbed:
        sys.exit("Empty video")

    prev_g = mask_road(cv.cvtColor(prev, cv.COLOR_BGR2GRAY))
    kp_prev, des_prev = orb.detectAndCompute(prev_g, None)

    # storage for results
    times, betas = [], []
    t, smoothed_vec = 0.0, np.array([0., 0.])  # time and smoothed motion vector

    # ---------- main frame loop ---------------------------------------------
    while True:
        # skip frames for down-sampling
        for _ in range(DOWNSAMPLE):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        # convert to grayscale + mask road
        gray = mask_road(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

        # detect ORB keypoints and descriptors
        kp, des = orb.detectAndCompute(gray, None)
        if des is None or des_prev is None:
            # no features detected → skip
            kp_prev, des_prev = kp, des
            t += dt
            continue

        # match features between previous and current frame
        matches = bf.knnMatch(des_prev, des, k=2)
        good = [m for m, n in matches if m.distance < GOOD_MATCH_FRAC * n.distance]

        if len(good) < 6:
            # too few matches → unreliable, skip frame
            kp_prev, des_prev = kp, des
            t += dt
            continue

        # get coordinates of matched keypoints
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts_cur  = np.float32([kp[m.trainIdx].pt  for m in good])

        # estimate affine transform (translation + rotation/scale, no shear)
        M, _ = cv.estimateAffinePartial2D(
            pts_prev, pts_cur,
            method=cv.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995
        )
        if M is None:
            # failed to estimate → skip frame
            kp_prev, des_prev = kp, des
            t += dt
            continue

        # translation components: dx, dy in pixels per frame
        dx, dy = M[0, 2], M[1, 2]

        # invert sign: camera wrt world → vehicle wrt road
        vec = -np.array([dx, dy])

        # smooth motion vector
        smoothed_vec = exponential_smooth(smoothed_vec, vec)

        # heading axis = +y in image coordinates
        beta_rad = np.arctan2(smoothed_vec[0], smoothed_vec[1])  # atan2(x, y)
        beta_deg = np.degrees(beta_rad) + HEADING_OFFSET

        # store results
        betas.append(beta_deg)
        times.append(t)
        t += dt

        # update for next loop
        kp_prev, des_prev = kp, des

    cap.release()

    if not betas:
        sys.exit("No usable data")

    # ---------- sanity check -----------------------------------------------
    assert len(times) == len(betas), f"len(times)={len(times)} ≠ len(betas)={len(betas)}"

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


# --------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python robust_slip_angle.py <video>")

    video_path = sys.argv[1]   # <--- define video_path
    main(video_path)

    stem = Path(video_path).with_suffix("").name
    csv_path = Path(f"{stem}_slip_angle.csv")

    import csv
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "beta_deg"])   # header
        writer.writerows(zip(times, betas))

    print(f"Saved datapoints to: {csv_path}")