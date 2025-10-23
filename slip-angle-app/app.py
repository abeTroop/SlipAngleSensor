import os, io, uuid, tempfile, csv
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# scientific stack
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend
import matplotlib.pyplot as plt

# enable CORS for Electron
from flask_cors import CORS

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
ROI_FRACTION_Y = 0.8
MAX_FEATURES = 1000
GOOD_MATCH_FRAC = 0.7
EMA_ALPHA = 0.2
DOWNSAMPLE = 4
HEADING_OFFSET = 90.0

UPLOAD_DIR = tempfile.mkdtemp(prefix="slipangle_")
app = Flask(__name__)
CORS(app)  # allows requests from Electron frontend

RESULTS = {}  # in-memory store (job_id → results)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def mask_road(gray):
    """Keep only bottom fraction of grayscale frame (road region)."""
    h = gray.shape[0]
    roi = np.zeros_like(gray)
    roi[int(h * (1 - ROI_FRACTION_Y)):, :] = 255
    return cv.bitwise_and(gray, roi)


def exponential_smooth(prev, current, alpha=EMA_ALPHA):
    return alpha * current + (1 - alpha) * prev


def analyze_video(video_path):
    """Estimate slip angle β from a video."""
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv.CAP_PROP_FPS) / max(1, DOWNSAMPLE)
    if fps <= 0:
        fps = 30.0 / max(1, DOWNSAMPLE)
    dt = 1.0 / fps

    orb = cv.ORB_create(MAX_FEATURES)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    grabbed, prev = cap.read()
    for _ in range(max(1, DOWNSAMPLE) - 1):
        cap.grab()
    if not grabbed:
        raise RuntimeError("Empty video")

    prev_g = mask_road(cv.cvtColor(prev, cv.COLOR_BGR2GRAY))
    kp_prev, des_prev = orb.detectAndCompute(prev_g, None)

    times, betas = [], []
    t, smoothed_vec = 0.0, np.array([0.0, 0.0])

    while True:
        for _ in range(max(1, DOWNSAMPLE)):
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
        good = [m for m, n in matches if n is not None and m.distance < GOOD_MATCH_FRAC * n.distance]
        if len(good) < 6:
            kp_prev, des_prev = kp, des
            t += dt
            continue

        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good])
        pts_cur  = np.float32([kp[m.trainIdx].pt for m in good])

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
        vec = -np.array([dx, dy])  # invert sign to align with forward motion
        smoothed_vec = exponential_smooth(smoothed_vec, vec)

        beta_rad = np.arctan2(smoothed_vec[0], smoothed_vec[1])
        beta_deg = float(np.degrees(beta_rad) + HEADING_OFFSET)

        betas.append(beta_deg)
        times.append(float(t))
        t += dt
        kp_prev, des_prev = kp, des

    cap.release()
    if not betas:
        raise RuntimeError("No usable data")

    return times, betas


def make_plot(times, betas, title):
    """Return matplotlib PNG as bytes."""
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.plot(times, betas)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Slip β [deg]")
    ax.set_title(title)
    ax.grid(ls=":")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def write_csv(path, times, betas):
    """Save (time, beta) CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "beta_deg"])
        writer.writerows(zip(times, betas))


# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.post("/api/analyze")
def api_analyze():
    if "video" not in request.files:
        return jsonify(error="No 'video' field in form"), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify(error="Empty filename"), 400

    job_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}_{filename}")
    file.save(video_path)

    try:
        times, betas = analyze_video(video_path)
    except Exception as e:
        return jsonify(error=str(e)), 500

    # save CSV
    csv_path = Path(UPLOAD_DIR) / f"{job_id}_slip_angle.csv"
    write_csv(csv_path, times, betas)

    mean_abs = float(np.mean(np.abs(betas)))
    peak_abs = float(np.max(np.abs(betas)))

    RESULTS[job_id] = {
        "times": times,
        "betas": betas,
        "csv_path": str(csv_path),
        "title": f"Slip Angle – {filename}",
    }

    return jsonify({
        "job_id": job_id,
        "video": filename,
        "stats": {
            "mean_abs_beta_deg": mean_abs,
            "peak_abs_beta_deg": peak_abs
        },
        "links": {
            "csv": f"/api/csv/{job_id}",
            "plot_png": f"/api/plot/{job_id}"
        }
    })


@app.get("/api/csv/<job_id>")
def api_csv(job_id):
    rec = RESULTS.get(job_id)
    if not rec:
        return jsonify(error="Invalid job_id"), 404
    return send_file(rec["csv_path"], as_attachment=True)


@app.get("/api/plot/<job_id>")
def api_plot(job_id):
    rec = RESULTS.get(job_id)
    if not rec:
        return jsonify(error="Invalid job_id"), 404
    png = make_plot(rec["times"], rec["betas"], rec["title"])
    return send_file(io.BytesIO(png), mimetype="image/png")


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
