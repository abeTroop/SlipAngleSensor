#!/usr/bin/env python3
"""
analyze_rotation.py  —  compare rotated vs original slip-angle CSVs

Inputs: two CSVs with columns: time_s,beta_deg
  - rotated.csv  : output from cosine-rotated video
  - original.csv : output from original (unrotated) video

Pipeline:
  1) Load CSVs.
  2) Align by index (default) or by time (interpolate original to rotated times).
  3) Compute circular angle difference: rotation_signal = wrap_deg(rot - orig).
  4) Unwrap to get a smooth waveform.
  5) If freq not provided, estimate dominant frequency via FFT.
  6) Fit y(t) ≈ A*cos(2π f t + φ) + C by linear least squares.
  7) Report metrics (A, f, φ, C, RMSE, R², corr) and plot.

Usage:
  python analyze_rotation.py rotated.csv original.csv [--align index|time]
                                                      [--freq-hz F]
                                                      [--out figure.png]
                                                      [--no-fft]
"""

import sys, csv, argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_csv(path):
    t, b = [], []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        # tolerate files with/without header
        def is_header_row(h):
            if h is None: return False
            joined = ",".join([s.strip().lower() for s in h])
            return "time" in joined and "beta" in joined
        if not is_header_row(header):
            # header is actually data; rewind
            f.seek(0); r = csv.reader(f)
        for row in r:
            if not row: continue
            try:
                t.append(float(row[0]))
                b.append(float(row[1]))
            except ValueError:
                # skip stray non-numeric lines (e.g., comments)
                continue
    return np.asarray(t), np.asarray(b)

def wrap_deg(x):
    """Wrap degrees to (-180, 180]."""
    return (x + 180.0) % 360.0 - 180.0

def unwrap_deg(x):
    """Unwrap degrees smoothly using radians unwrap."""
    return np.rad2deg(np.unwrap(np.deg2rad(x)))

def align_index(t1, y1, t2, y2):
    n = min(len(y1), len(y2))
    return t1[:n], y1[:n], t2[:n], y2[:n]

def align_time(t_ref, y_ref, t_other, y_other):
    """
    Interpolate y_other onto t_ref’s grid (extrapolation clamped to ends).
    Returns (t_ref, y_ref, y_other_interp)
    """
    # Ensure strictly increasing for interp
    order = np.argsort(t_other)
    t_sorted = t_other[order]
    y_sorted = y_other[order]
    y_interp = np.interp(t_ref, t_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])
    return t_ref, y_ref, y_interp

def estimate_freq_hz(t, y):
    """
    Estimate dominant frequency via FFT (ignore DC).
    Returns frequency in Hz (>= 0).
    """
    # sampling interval (assume quasi-uniform)
    dt = np.median(np.diff(t))
    if dt <= 0:
        return 0.0
    fs = 1.0 / dt

    # Detrend
    y0 = y - np.mean(y)

    # Next power of two for cleaner FFT
    n = int(2 ** np.ceil(np.log2(max(len(y0), 16))))
    Y = np.fft.rfft(y0, n=n)
    freqs = np.fft.rfftfreq(n, d=dt)

    # Ignore 0 Hz (DC)
    if len(freqs) < 3:
        return 0.0
    mag = np.abs(Y)
    mag[0] = 0.0

    # Pick the peak
    k = np.argmax(mag)
    return float(freqs[k])

def fit_cosine(t, y, freq_hz):
    """
    Fit y ≈ A*cos(2π f t + φ) + C via linear least squares using cos/sin basis.
    Returns dict with A, phi (rad), C, y_fit, and components.
    """
    w = 2.0 * np.pi * freq_hz
    c = np.cos(w * t)
    s = np.sin(w * t)
    X = np.column_stack([c, s, np.ones_like(t)])  # [cos, sin, offset]
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, C = coeffs
    A = np.hypot(a, b)
    phi = np.arctan2(-b, a)  # because A*cos(wt+phi) = a*cos + b*sin with a=A*cos(phi), b=-A*sin(phi)
    y_fit = X @ coeffs
    return {"A": float(A), "phi": float(phi), "C": float(C), "y_fit": y_fit, "a": float(a), "b": float(b)}

def metrics(y, y_fit):
    resid = y - y_fit
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    corr = float(np.corrcoef(y, y_fit)[0,1]) if len(y) > 1 else np.nan
    return rmse, r2, corr, resid

def main():
    p = argparse.ArgumentParser(description="Compare rotated vs original slip-angle CSVs and fit a cosine.")
    p.add_argument("rotated_csv", help="CSV from cosine-rotated video (time_s,beta_deg)")
    p.add_argument("original_csv", help="CSV from original video (time_s,beta_deg)")
    p.add_argument("--align", choices=["index","time"], default="index",
                   help="Alignment method: match by index or interpolate by time (default: index)")
    p.add_argument("--freq-hz", type=float, default=None,
                   help="Known rotation frequency in Hz; if omitted, estimated via FFT")
    p.add_argument("--out", type=str, default=None, help="Path to save figure (PNG). If omitted, just shows window.")
    p.add_argument("--no-fft", action="store_true",
                   help="Skip FFT subplot (useful if you only want the fit/residuals).")
    args = p.parse_args()

    t_rot, b_rot = read_csv(args.rotated_csv)
    t_org, b_org = read_csv(args.original_csv)

    if args.align == "index":
        _, b_rot2, _, b_org2 = align_index(t_rot, b_rot, t_org, b_org)
        # Build a common time assuming median dt of rotated:
        n = len(b_rot2)
        if n < 2:
            sys.exit("Not enough datapoints after alignment.")
        dt = np.median(np.diff(t_rot[:n])) if len(t_rot) >= n else np.median(np.diff(t_rot))
        t_common = np.arange(n) * (dt if np.isfinite(dt) and dt > 0 else 1.0/30.0)
    else:
        # Interpolate original → rotated timeline
        t_common, b_rot2, b_org2 = align_time(t_rot, b_rot, t_org, b_org)

    # 1) circular difference in degrees (wrap to [-180,180))
    diff_wrapped = wrap_deg(b_rot2 - b_org2)
    # 2) unwrap to get smooth sinusoid
    rotation_signal = unwrap_deg(diff_wrapped)

    # Estimate frequency if not provided
    if args.freq_hz is None:
        f_est = estimate_freq_hz(t_common, rotation_signal)
        freq_hz = f_est
    else:
        freq_hz = float(args.freq_hz)

    # Fit cosine
    fit = fit_cosine(t_common, rotation_signal, freq_hz=freq_hz if np.isfinite(freq_hz) else 0.0)
    rmse, r2, corr, resid = metrics(rotation_signal, fit["y_fit"])

    # Report
    def deg(rad): return np.degrees(rad)
    print(f"Samples: {len(rotation_signal)}")
    print(f"Freq (Hz): {freq_hz:.6f} {'(estimated)' if args.freq_hz is None else '(provided)'}")
    print(f"Amplitude (deg): {fit['A']:.4f}")
    print(f"Phase (deg):     {deg(fit['phi']):.2f}")
    print(f"Offset C (deg):  {fit['C']:.4f}")
    print(f"RMSE (deg):      {rmse:.4f}")
    print(f"R^2:             {r2:.4f}")
    print(f"Corr:            {corr:.4f}")

    # Plot
    fig_h = 8 if args.no_fft else 10
    rows = 2 if args.no_fft else 3
    plt.figure(figsize=(10, fig_h))

    # Top: signal + fit
    ax1 = plt.subplot(rows,1,1)
    ax1.plot(t_common, rotation_signal, label="rotation signal (rot - orig)")
    ax1.plot(t_common, fit["y_fit"], linestyle="--", label="cosine fit")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Degrees")
    ax1.set_title(f"Rotation Signal vs Cosine Fit  |  A={fit['A']:.2f}°, f={freq_hz:.3f} Hz, φ={deg(fit['phi']):.1f}°, C={fit['C']:.2f}°  |  RMSE={rmse:.2f}°, R²={r2:.3f}")
    ax1.grid(ls=":")
    ax1.legend()

    # Middle: residuals
    ax2 = plt.subplot(rows,1,2)
    ax2.plot(t_common, resid, label="residuals")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Degrees")
    ax2.set_title("Residuals (signal − fit)")
    ax2.grid(ls=":")

    if not args.no_fft:
        # Bottom: FFT magnitude (for visualization only)
        dt = np.median(np.diff(t_common))
        if np.isfinite(dt) and dt > 0:
            y0 = rotation_signal - np.mean(rotation_signal)
            n = int(2 ** np.ceil(np.log2(max(len(y0), 16))))
            Y = np.fft.rfft(y0, n=n)
            freqs = np.fft.rfftfreq(n, d=dt)
            ax3 = plt.subplot(rows,1,3)
            ax3.plot(freqs, np.abs(Y))
            ax3.set_xlim(0, (0.5/dt))
            ax3.set_xlabel("Frequency [Hz]")
            ax3.set_ylabel("|FFT|")
            ax3.set_title("FFT (magnitude) of rotation signal")
            ax3.grid(ls=":")
            ax3.axvline(freq_hz, color="r", ls="--", lw=0.8, label="fit freq")
            ax3.legend()

    plt.tight_layout()

    if args.out:
        out_path = Path(args.out)
        plt.savefig(out_path, dpi=150)
        print(f"Saved figure: {out_path}")
    plt.show()

if __name__ == "__main__":
    main()
