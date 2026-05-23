"""Assign log10_surviv_stellar_mass from a GalSBI-SPS catalog to an f24 catalog.

Method
------
Within each galaxy type (red/blue) separately, build a KD-tree on the SPS
catalog in abs_mag (absolute r-band magnitude) space and draw one stellar-mass
value per f24 galaxy from its k nearest SPS neighbours.

Optionally, redshift can be added as a second KNN feature via --use-redshift.
Both features are normalised to zero-mean unit-variance of the SPS pool before
building the tree; --redshift-weight scales the normalised z axis relative to
the normalised abs_mag axis (default 1.0 = equal weight).

Note on blues: SPS blues have z_med~0.38 while f24 blues peak at z_med~0.94.
Adding z for blues may concentrate neighbours in the sparse high-z SPS tail.
Use --reds-only-redshift to restrict 2-D matching to reds and keep blues on
abs_mag only.  A simple M/L fallback is still available via --blue-ml-fallback.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as h5:
        data = h5["data"]
        missing = [c for c in columns if c not in (data.dtype.names or ())]
        if missing:
            raise KeyError(f"{path.name}: missing columns {missing}")
        return {c: data.fields(c)[:].astype(float) for c in columns}


def _load_int(path: Path, column: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return h5["data"].fields(column)[:].astype(int)


# ---------------------------------------------------------------------------
# Stellar-mass assignment
# ---------------------------------------------------------------------------

def _knn_assign(
    sps_abs_r: np.ndarray,
    sps_ms: np.ndarray,
    f24_abs_r: np.ndarray,
    k: int,
    rng: np.random.Generator,
    label: str,
    sps_z: np.ndarray | None = None,
    f24_z: np.ndarray | None = None,
    z_weight: float = 1.0,
) -> tuple[np.ndarray, float]:
    """KNN stellar-mass assignment in abs_mag (and optionally redshift) space.

    When sps_z / f24_z are provided the tree is built in 2-D:
      feature = [normalised_abs_mag, z_weight * normalised_z]
    Both axes are normalised using the SPS pool mean and std.

    Returns
    -------
    assigned_ms : (N_f24,) float array of log10_surviv_stellar_mass
    median_dist : median neighbour distance (in normalised units)
    """
    use_z = sps_z is not None and f24_z is not None

    if len(sps_abs_r) < k:
        raise ValueError(
            f"{label}: SPS pool has only {len(sps_abs_r)} galaxies, need at least k={k}. "
            "Reduce --k or provide a larger SPS catalog."
        )

    # Normalise abs_mag to zero-mean unit-variance of the SPS pool.
    mu_m, sigma_m = sps_abs_r.mean(), sps_abs_r.std()
    if sigma_m == 0:
        raise ValueError(f"{label}: SPS pool has zero variance in abs_mag.")
    sps_feat = (sps_abs_r - mu_m) / sigma_m
    f24_feat = (f24_abs_r - mu_m) / sigma_m

    if use_z:
        mu_z, sigma_z = sps_z.mean(), sps_z.std()
        if sigma_z == 0:
            raise ValueError(f"{label}: SPS pool has zero variance in redshift.")
        sps_feat = np.column_stack([sps_feat, z_weight * (sps_z - mu_z) / sigma_z])
        f24_feat = np.column_stack([f24_feat, z_weight * (f24_z - mu_z) / sigma_z])
    else:
        sps_feat = sps_feat[:, None]
        f24_feat = f24_feat[:, None]

    tree = cKDTree(sps_feat)
    dists, idx = tree.query(f24_feat, k=k, workers=-1)

    # idx shape: (N_f24, k) — draw one neighbour per galaxy
    if k == 1:
        chosen = idx[:, None] if idx.ndim == 1 else idx
        chosen = chosen[:, 0]
    else:
        col = rng.integers(0, k, size=len(f24_abs_r))
        chosen = idx[np.arange(len(f24_abs_r)), col]

    return sps_ms[chosen], float(np.median(dists.min(axis=1) if dists.ndim > 1 else dists))


def assign_stellar_mass(
    f24_path: Path,
    sps_path: Path,
    out_path: Path,
    k: int = 10,
    mag_max: float = 28.0,
    seed: int = 0,
    blue_ml_fallback: bool = False,
    solar_abs_mag_r: float = 4.65,
    blue_mass_to_light: float = 0.19,
    use_redshift: bool = False,
    reds_only_redshift: bool = False,
    z_weight: float = 1.0,
    z_column: str = "z",
) -> None:
    rng = np.random.default_rng(seed)

    # ---- load SPS (only r<mag_max, both types) ----------------------------
    sps_cols = ["int_mag", "abs_mag", "log10_surviv_stellar_mass"]
    if use_redshift:
        sps_cols.append(z_column)
    sps = _load_columns(sps_path, sps_cols)
    sps_gt = _load_int(sps_path, "galaxy_type")

    sps_ok = np.isfinite(sps["log10_surviv_stellar_mass"]) & (sps["int_mag"] < mag_max)
    for key in sps_cols:
        sps[key] = sps[key][sps_ok]
    sps_gt = sps_gt[sps_ok]
    print(f"SPS pool (r<{mag_max}): {sps_ok.sum():,} galaxies  "
          f"(red={( sps_gt==1).sum():,}  blue={(sps_gt==0).sum():,})")

    # ---- load f24 ---------------------------------------------------------
    f24_cols = ["int_mag", "abs_mag"]
    if use_redshift:
        f24_cols.append(z_column)
    f24 = _load_columns(f24_path, f24_cols)
    f24_gt = _load_int(f24_path, "galaxy_type")
    n_f24 = len(f24["abs_mag"])
    print(f"f24 catalog: {n_f24:,} galaxies  "
          f"(red={(f24_gt==1).sum():,}  blue={(f24_gt==0).sum():,})")

    assigned_ms = np.full(n_f24, np.nan)

    # ---- assign per galaxy type -------------------------------------------
    for gtype, label in [(1, "red"), (0, "blue")]:
        f24_mask = f24_gt == gtype
        sps_mask = sps_gt == gtype

        if f24_mask.sum() == 0:
            continue

        if blue_ml_fallback and gtype == 0:
            # M/L fallback: log10(M*) = -0.4*(abs_mag - M_sun) + log10(M/L)
            print(f"[blue] using M/L fallback  (solar_abs_mag_r={solar_abs_mag_r}, "
                  f"M/L={blue_mass_to_light})")
            log_L = -0.4 * (f24["abs_mag"][f24_mask] - solar_abs_mag_r)
            assigned_ms[f24_mask] = log_L + np.log10(blue_mass_to_light)
            continue

        # Decide whether to use z for this galaxy type
        add_z = use_redshift and not (reds_only_redshift and gtype == 0)
        if add_z:
            print(f"[{label}] 2-D KNN: abs_mag + {z_weight:.2f}*z  (z_column={z_column!r})")

        ms_assigned, med_dist = _knn_assign(
            sps_abs_r = sps["abs_mag"][sps_mask],
            sps_ms    = sps["log10_surviv_stellar_mass"][sps_mask],
            f24_abs_r = f24["abs_mag"][f24_mask],
            k=k,
            rng=rng,
            label=label,
            sps_z     = sps[z_column][sps_mask] if add_z else None,
            f24_z     = f24[z_column][f24_mask] if add_z else None,
            z_weight  = z_weight,
        )
        assigned_ms[f24_mask] = ms_assigned
        print(f"[{label}] assigned {f24_mask.sum():,} galaxies, "
              f"median KNN dist (normalised)={med_dist:.3f}")

    n_missing = np.isnan(assigned_ms).sum()
    if n_missing > 0:
        raise RuntimeError(f"{n_missing} f24 galaxies received no stellar mass assignment.")

    # ---- write output -----------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(f24_path, "r") as src, h5py.File(out_path, "w") as dst:
        src.copy("data", dst)
        data = dst["data"]
        new_dtype = np.dtype(data.dtype.descr + [("log10_surviv_stellar_mass", "f4")])
        old = data[:]
        new = np.empty(len(old), dtype=new_dtype)
        for col in old.dtype.names:
            new[col] = old[col]
        new["log10_surviv_stellar_mass"] = assigned_ms.astype("f4")
        del dst["data"]
        dst.create_dataset("data", data=new, compression="gzip", compression_opts=4)

    print(f"Written: {out_path}  ({n_f24:,} rows, log10_surviv_stellar_mass added)")
    print(f"  assigned range: [{assigned_ms.min():.2f}, {assigned_ms.max():.2f}]  "
          f"median={np.median(assigned_ms):.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("f24_path",  type=Path, help="f24 HDF5 catalog (input)")
    p.add_argument("sps_path",  type=Path, help="SPS r-band HDF5 catalog (donor)")
    p.add_argument("out_path",  type=Path, help="output HDF5 path")
    p.add_argument("--k",           type=int,   default=10,
                   help="number of nearest neighbours to draw from (default: 10)")
    p.add_argument("--mag-max",     type=float, default=28.0,
                   help="magnitude limit applied to SPS pool (default: 28)")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--blue-ml-fallback", action="store_true",
                   help="use M/L-ratio formula for blue galaxies instead of KNN")
    p.add_argument("--solar-abs-mag-r", type=float, default=4.65)
    p.add_argument("--blue-mass-to-light", type=float, default=0.19)
    p.add_argument("--use-redshift", action="store_true",
                   help="add redshift as a second KNN feature (2-D matching)")
    p.add_argument("--reds-only-redshift", action="store_true",
                   help="apply 2-D (abs_mag, z) matching to reds only; keep blues on abs_mag alone")
    p.add_argument("--redshift-weight", type=float, default=1.0,
                   help="weight for the normalised z axis relative to normalised abs_mag (default 1.0)")
    p.add_argument("--z-column", default="z",
                   help="HDF5 column name for redshift in both catalogs (default: 'z')")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    assign_stellar_mass(
        f24_path           = args.f24_path,
        sps_path           = args.sps_path,
        out_path           = args.out_path,
        k                  = args.k,
        mag_max            = args.mag_max,
        seed               = args.seed,
        blue_ml_fallback   = args.blue_ml_fallback,
        solar_abs_mag_r    = args.solar_abs_mag_r,
        blue_mass_to_light = args.blue_mass_to_light,
        use_redshift       = args.use_redshift,
        reds_only_redshift = args.reds_only_redshift,
        z_weight           = args.redshift_weight,
        z_column           = args.z_column,
    )


if __name__ == "__main__":
    main()
