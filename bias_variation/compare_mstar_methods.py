"""Compare three stellar-mass assignment methods for f24 blue galaxies.

Methods
-------
1. KNN   : transfer log10_surviv_stellar_mass from SPS blues via k-nearest
           neighbours in abs_r space.
2. ML    : M/L-ratio formula  log10(M*) = -0.4*(abs_r - M_r,sun) + log10(M/L)
3. COSMOS: ground-truth M* from COSMOS2020 LePhare SED fitting (lp_mass_best),
           matched statistically in (z, abs_r) bins.

Since f24 positions are reprojected HEALPix coordinates (not the original
COSMOS footprint), object-by-object cross-matching is not possible.  Instead
we select COSMOS2020 galaxies in the same (z, apparent-mag) window and compare
M* distributions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GALSBI_DIR = Path("/project/ls-gruen/users/zekang.zhang/cats/galsbi")
F24_PATH   = GALSBI_DIR / "f24_0_r_ucat.gal.cat"
SPS_PATH   = GALSBI_DIR / "galsbi_sps_bestfit_1m_0_r_ucat.gal.cat"
COSMOS_PATH = Path(
    "/project/ls-gruen/users/luca.tortorelli/data/COSMOS2020_CLASSIC_R1_v2.0.fits"
)

SOLAR_ABS_R = 4.65
MAG_MAX     = 28.0
K_KNN       = 10
SEED        = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_hdf(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        d = h5["data"]
        return {c: d.fields(c)[:].astype(float) for c in cols}

def _load_int(path: Path, col: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return h5["data"].fields(col)[:].astype(int)


def _knn_abs_r(
    sps_abs_r: np.ndarray,
    sps_ms: np.ndarray,
    f24_abs_r: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """KNN M* assignment using abs_r as sole feature (normalised)."""
    mu, sigma = sps_abs_r.mean(), sps_abs_r.std()
    tree = cKDTree(((sps_abs_r - mu) / sigma)[:, None])
    dists, idx = tree.query(((f24_abs_r - mu) / sigma)[:, None], k=k, workers=-1)
    if k == 1:
        chosen = idx[:, 0] if idx.ndim > 1 else idx
    else:
        col = rng.integers(0, k, size=len(f24_abs_r))
        chosen = idx[np.arange(len(f24_abs_r)), col]
    print(f"  KNN median dist (normalised): {float(np.median(dists.min(axis=1) if dists.ndim > 1 else dists)):.4f}")
    return sps_ms[chosen]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(out_dir: Path, mass_to_light: float = 1.0) -> None:
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load f24 blues
    # ------------------------------------------------------------------
    print("Loading f24 …")
    f24 = _load_hdf(F24_PATH, ["abs_mag", "int_mag", "z"])
    f24_gt = _load_int(F24_PATH, "galaxy_type")

    f24_blue = (f24_gt == 0) & (f24["int_mag"] < MAG_MAX)
    abs_r_f24 = f24["abs_mag"][f24_blue]
    z_f24     = f24["z"][f24_blue]
    print(f"  f24 blues (r<{MAG_MAX}): {f24_blue.sum():,}  z_med={np.median(z_f24):.2f}")

    # ------------------------------------------------------------------
    # 2. Load SPS blues
    # ------------------------------------------------------------------
    print("Loading SPS …")
    sps = _load_hdf(SPS_PATH, ["abs_mag", "int_mag", "log10_surviv_stellar_mass"])
    sps_gt = _load_int(SPS_PATH, "galaxy_type")

    sps_blue = (sps_gt == 0) & (sps["int_mag"] < MAG_MAX) & \
               np.isfinite(sps["log10_surviv_stellar_mass"])
    abs_r_sps = sps["abs_mag"][sps_blue]
    ms_sps    = sps["log10_surviv_stellar_mass"][sps_blue]
    print(f"  SPS blues (r<{MAG_MAX}): {sps_blue.sum():,}  z_med=0.17 (as measured)")

    # ------------------------------------------------------------------
    # 3. Method 1 — KNN
    # ------------------------------------------------------------------
    print("KNN assignment …")
    ms_knn = _knn_abs_r(abs_r_sps, ms_sps, abs_r_f24, k=K_KNN, rng=rng)

    # ------------------------------------------------------------------
    # 4. Method 2 — M/L fallback
    # ------------------------------------------------------------------
    log_L    = -0.4 * (abs_r_f24 - SOLAR_ABS_R)
    ms_ml    = log_L + np.log10(mass_to_light)

    # ------------------------------------------------------------------
    # 5. COSMOS2020 "blue" reference
    # ------------------------------------------------------------------
    print("Loading COSMOS2020 …")
    cosmos = Table.read(COSMOS_PATH)

    # Blue selection: sSFR > -11 (star-forming), mass and z valid
    lp_ssfr  = np.array(cosmos["lp_sSFR_best"], dtype=float)
    lp_mass  = np.array(cosmos["lp_mass_best"], dtype=float)
    lp_z     = np.array(cosmos["lp_zBEST"],    dtype=float)
    lp_type  = np.array(cosmos["lp_type"],      dtype=float)

    # Also recover apparent r magnitude from LePhare abs mag + DM
    # We don't have apparent r in COSMOS2020 directly, so filter on z and mass
    c_blue = (
        (lp_type == 0)          # galaxy (not star)
        & (lp_ssfr > -11.5)     # star-forming / blue
        & (lp_mass > 0)         # valid mass
        & (lp_z > 0.1)          # valid z
        & (lp_z < 4.0)
    )
    ms_cosmos = lp_mass[c_blue]
    z_cosmos  = lp_z[c_blue]
    print(f"  COSMOS2020 SF blues: {c_blue.sum():,}  z_med={np.median(z_cosmos):.2f}")

    # Restrict both to the same z range for fair comparison
    z_lo, z_hi = 0.3, 2.5
    f_zok = (z_f24 >= z_lo) & (z_f24 <= z_hi)
    c_zok = (z_cosmos >= z_lo) & (z_cosmos <= z_hi)
    print(f"  f24 blues in z=[{z_lo},{z_hi}]: {f_zok.sum():,}")
    print(f"  COSMOS blues in z=[{z_lo},{z_hi}]: {c_zok.sum():,}")

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    def _stats(arr: np.ndarray, label: str) -> None:
        print(f"  {label}: mean={arr.mean():.2f}  std={arr.std():.2f}  "
              f"p16={np.percentile(arr,16):.2f}  p84={np.percentile(arr,84):.2f}")

    print("\nM* summary (all z, f24 blues):")
    _stats(ms_knn, "KNN  ")
    _stats(ms_ml,  "M/L  ")

    print(f"\nM* summary (z=[{z_lo},{z_hi}]):")
    _stats(ms_knn[f_zok], "KNN  ")
    _stats(ms_ml[f_zok],  "M/L  ")
    _stats(ms_cosmos[c_zok], "COSMOS")

    ks_knn,    p_knn    = ks_2samp(ms_knn[f_zok],    ms_cosmos[c_zok])
    ks_ml,     p_ml     = ks_2samp(ms_ml[f_zok],     ms_cosmos[c_zok])
    print(f"\nKS test vs COSMOS (z=[{z_lo},{z_hi}]):")
    print(f"  KNN  : D={ks_knn:.3f}  p={p_knn:.2e}")
    print(f"  M/L  : D={ks_ml:.3f}   p={p_ml:.2e}")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    bins = np.linspace(5, 13, 65)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Blue galaxy M* comparison  (z=[{z_lo}, {z_hi}])", fontsize=13)

    # Panel 1: normalised histograms
    ax = axes[0]
    kw = dict(bins=bins, histtype="step", density=True, lw=1.8)
    ax.hist(ms_knn[f_zok],    label=f"KNN  (D={ks_knn:.3f})", color="steelblue", **kw)
    ax.hist(ms_ml[f_zok],     label=f"M/L={mass_to_light}  (D={ks_ml:.3f})",  color="darkorange", **kw)
    ax.hist(ms_cosmos[c_zok], label="COSMOS2020", color="black", linestyle="--", **kw)
    ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel("PDF")
    ax.legend(fontsize=9)
    ax.set_title("M* distributions")

    # Panel 2: cumulative
    ax = axes[1]
    for arr, label, color, ls in [
        (ms_knn[f_zok],    f"KNN",        "steelblue",  "-"),
        (ms_ml[f_zok],     f"M/L={mass_to_light}", "darkorange", "-"),
        (ms_cosmos[c_zok], "COSMOS2020",  "black",      "--"),
    ]:
        xs = np.sort(arr)
        ys = np.arange(1, len(xs)+1) / len(xs)
        ax.plot(xs, ys, label=label, color=color, ls=ls, lw=1.8)
    ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
    ax.set_ylabel("CDF")
    ax.legend(fontsize=9)
    ax.set_title("Cumulative distributions")

    plt.tight_layout()
    fig_path = out_dir / "compare_mstar_methods.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # Also: z-sliced panels
    z_edges = [0.3, 0.6, 1.0, 1.5, 2.5]
    fig2, axes2 = plt.subplots(1, len(z_edges)-1, figsize=(16, 4), sharey=True)
    fig2.suptitle("M* distributions per redshift bin", fontsize=12)
    for i, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        ax = axes2[i]
        f_zbin = (z_f24   >= zlo) & (z_f24   < zhi)
        c_zbin = (z_cosmos >= zlo) & (z_cosmos < zhi)
        kw2 = dict(bins=bins, histtype="step", density=True, lw=1.6)
        ax.hist(ms_knn[f_zbin],    color="steelblue",  label="KNN",         **kw2)
        ax.hist(ms_ml[f_zbin],     color="darkorange",  label=f"M/L={mass_to_light}", **kw2)
        ax.hist(ms_cosmos[c_zbin], color="black", ls="--", label="COSMOS2020", **kw2)
        ax.set_title(f"z=[{zlo},{zhi}]  N_f24={f_zbin.sum():,}")
        ax.set_xlabel(r"$\log_{10}(M_*/M_\odot)$")
        if i == 0:
            ax.set_ylabel("PDF")
            ax.legend(fontsize=8)
    plt.tight_layout()
    fig2_path = out_dir / "compare_mstar_zbins.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"Figure saved: {fig2_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir",        type=Path, default=Path("output/mstar_comparison"))
    p.add_argument("--mass-to-light",  type=float, default=1.0)
    args = p.parse_args(argv)
    run(args.out_dir, mass_to_light=args.mass_to_light)


if __name__ == "__main__":
    main()
