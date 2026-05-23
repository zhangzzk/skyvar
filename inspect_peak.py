"""Check if extreme-bias outliers drive the mean b_sel peak at z~0.85."""
import numpy as np
import pandas as pd

FEATHER = "/project/ls-gruen/users/zekang.zhang/cats/galsbi/f24_0_r_ucat_with_mstar_bias_source.feather"
NPZ = (
    "/home/z/Zekang.Zhang/skyvar/bias_variation/output/"
    "full_skyvar_bias1024_stats256_npop2157_f24_mstar/"
    "full_skyvar_bias1024_stats256_npop2157_f24_mstar_detonly_redshift_binned_bias_maps.npz"
)

print("Loading feather ...")
df = pd.read_feather(FEATHER)

# ── 1. Outlier fraction vs redshift ──────────────────────────────────────────
z_edges = np.arange(0.0, 2.05, 0.10)
z_mids  = 0.5 * (z_edges[:-1] + z_edges[1:])

print("\nBias outlier fraction by redshift (feather source table):")
print(f"{'z':>6}  {'n':>8}  {'b_p50':>7}  {'b_p90':>7}  {'b_p99':>7}  {'b_max':>8}  "
      f"{'f_b>5':>7}  {'f_b>20':>8}  {'b_mean':>7}")
for lo, hi, zm in zip(z_edges[:-1], z_edges[1:], z_mids):
    s = df[(df["redshift"] >= lo) & (df["redshift"] < hi)]["bias_tinker10"]
    if len(s) < 100:
        continue
    print(f"{zm:6.2f}  {len(s):8d}  {s.quantile(0.50):7.3f}  "
          f"{s.quantile(0.90):7.3f}  {s.quantile(0.99):7.3f}  {s.max():8.2f}  "
          f"{(s>5).mean():7.4f}  {(s>20).mean():8.5f}  {s.mean():7.3f}")

# ── 2. Verify: if 8.75% are detected and we remove bias>10 outliers,
#    what does mean_bias_selected become? ─────────────────────────────────────
print("\n\nSimulated mean_bias_selected at z~0.85 with outlier removal:")
s = df[(df["redshift"] >= 0.82) & (df["redshift"] < 0.88)]["bias_tinker10"]
n_tot = len(s)
n_det_expected = int(0.0875 * n_tot)

# Scenario: detected = top 8.75% by flux, which has -0.78 correlation with bias
# Approximate as: mean = full, with enrichment
print(f"  Full z=[0.82,0.88): n={n_tot:,}, mean={s.mean():.4f}, median={s.median():.4f}")

# Check top 8.75% by bias
top_b = s.nlargest(n_det_expected)
print(f"  Top 8.75% by bias: mean={top_b.mean():.4f}, median={top_b.median():.4f}")

# Remove extreme outliers and recompute
for cap in [5, 10, 20, np.inf]:
    s_cap = s.clip(upper=cap)
    top_b_cap = s_cap.nlargest(n_det_expected)
    print(f"  After capping at bias={cap:>6}: full_mean={s_cap.mean():.3f}, "
          f"top8.75%_mean={top_b_cap.mean():.3f}, top8.75%_median={top_b_cap.median():.3f}")

# ── 3. NPZ: median vs mean over pixels at z~0.85 ─────────────────────────────
npz = np.load(NPZ)
z_c = npz["z_centers"]
mean_s_map = np.asarray(npz["mean_bias_selected_z"], dtype=float)  # (n_pix, n_z)
cnt_s  = np.asarray(npz["count_selected_z"], dtype=float)

idx = np.argmin(np.abs(z_c - 0.85))
print(f"\n\nNPZ at z={z_c[idx]:.3f}:")
col = mean_s_map[:, idx]
cnt_col = cnt_s[:, idx]
has_data = cnt_col > 0
print(f"  Pixels with detections: {has_data.sum()}")
vals = col[has_data]
print(f"  Mean of pixel-means:   {np.nanmean(vals):.4f}")
print(f"  Median of pixel-means: {np.nanmedian(vals):.4f}")
print(f"  p75 of pixel-means:    {np.nanpercentile(vals, 75):.4f}")
print(f"  p90 of pixel-means:    {np.nanpercentile(vals, 90):.4f}")
print(f"  p95 of pixel-means:    {np.nanpercentile(vals, 95):.4f}")
print(f"  p99 of pixel-means:    {np.nanpercentile(vals, 99):.4f}")
print(f"  max of pixel-means:    {np.nanmax(vals):.4f}")

# Same for z=0.65 (pre-peak) and z=1.10 (post-peak)
for z_tgt in [0.65, 0.85, 1.05, 1.25]:
    idx = np.argmin(np.abs(z_c - z_tgt))
    col = mean_s_map[:, idx]
    cnt_col = cnt_s[:, idx]
    has_data = cnt_col > 0
    vals = col[has_data]
    print(f"  z={z_c[idx]:.3f}: median_pix={np.nanmedian(vals):.4f}  "
          f"mean_pix={np.nanmean(vals):.4f}  "
          f"p95={np.nanpercentile(vals,95):.4f}  "
          f"max={np.nanmax(vals):.4f}  n_pix={has_data.sum()}")
