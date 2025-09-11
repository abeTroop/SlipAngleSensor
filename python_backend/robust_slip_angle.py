#!/usr/bin/env python3
"""
robust_slip_angle.py  –  better‑behaved β estimate
"""

import sys, cv2 as cv, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

# ---------- tune here -------------------------------------------------
ROI_FRACTION_Y = .8       # keep lowest 33 % of the frame; changed this to 1 but expert friend said 33 % 
MAX_FEATURES   = 1000
GOOD_MATCH_FRAC = 0.7
EMA_ALPHA      = 0.1         # 0.1 ≈ 1 / (1/α)‑frame time‑constant
DOWNSAMPLE     = 4           # 240 fps → 60 fps; changed to 1 from 4
FOCAL_PX       = 1100        # <- set this from camera intrinsics
HEADING_OFFSET = 90.0        # deg. add +90 if phone was held clockwise
# ----------------------------------------------------------------------

def mask_road(gray):
    h = gray.shape[0]
    roi = np.zeros_like(gray)
    roi[int(h * (1 - ROI_FRACTION_Y)):, :] = 255
    return cv.bitwise_and(gray, roi)

def exponential_smooth(prev, current, alpha=EMA_ALPHA):
    return alpha * current + (1 - alpha) * prev

def main(video):
    cap = cv.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("Cannot open video")

    fps = cap.get(cv.CAP_PROP_FPS) / DOWNSAMPLE
    dt  = 1.0 / fps

    orb = cv.ORB_create(MAX_FEATURES)
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    # ---------- first frame ------------------------------------------
    grabbed, prev = cap.read()
    for _ in range(DOWNSAMPLE - 1):
        cap.grab()
    if not grabbed:
        sys.exit("Empty video")

    prev_g = mask_road(cv.cvtColor(prev, cv.COLOR_BGR2GRAY))
    kp_prev, des_prev = orb.detectAndCompute(prev_g, None)

    times, betas = [], []
    t, smoothed_vec = 0.0, np.array([0., 0.])

    # ---------- main loop -------------------------------------------
    while True:
        # skip frames for down‑sampling
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

        # affine with only translation + rotation/scale (no shear)
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

        dx, dy = M[0, 2], M[1, 2]        # px / frame  (camera wrt world)
        vec = -np.array([dx, dy])        # world wrt camera = vehicle motion
        smoothed_vec = exponential_smooth(smoothed_vec, vec)

        # heading axis = +y in image coords
        beta_rad = np.arctan2(smoothed_vec[0], smoothed_vec[1])
        beta_deg = np.degrees(beta_rad) + HEADING_OFFSET

        # ------------- collect ---------------------------------------
        betas.append(beta_deg)
        times.append(t)
        t += dt

        kp_prev, des_prev = kp, des

    cap.release()
    if not betas:
        sys.exit("No usable data")

    # ---------- sanity check ----------------------------------------
    assert len(times) == len(betas), f"len(times)={len(times)} ≠ len(betas)={len(betas)}"

    # ---------- plot -------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(times, betas)
    plt.axhline(0, color='k', ls='--', lw=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Slip β [deg]")
    plt.title(f"Robust slip angle – {Path(video).name}")
    plt.grid(ls=':')
    plt.tight_layout()
    plt.show()

    print(f"mean |β| = {np.mean(np.abs(betas)):.2f}°,  "
          f"peak |β| = {np.max(np.abs(betas)):.2f}°")

# --------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python robust_slip_angle.py <video>")
    main(sys.argv[1])