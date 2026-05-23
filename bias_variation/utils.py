"""Small shared helpers for the bias-variation package."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def write_json(path: str | Path, payload: dict) -> None:
    """Write a deterministic JSON summary."""

    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_npz_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Load an ``np.savez`` file into an ordinary dict."""

    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def read_pixel_file(path: str | Path) -> np.ndarray:
    """Read integer HEALPix pixels from a whitespace- or comma-separated file."""

    text = Path(path).read_text(encoding="utf-8")
    pixels = [int(token) for token in text.replace(",", " ").split()]
    if not pixels:
        raise ValueError(f"No pixels found in {path}.")
    return np.asarray(pixels, dtype=np.int64)


def finite_stats(values: list[object]) -> dict[str, float | int | None]:
    """Return compact min/median/max stats for finite numeric arrays."""

    if not values:
        return {"finite": 0, "min": None, "median": None, "max": None}
    arr = np.concatenate(values)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"finite": 0, "min": None, "median": None, "max": None}
    vals = arr[finite]
    return {
        "finite": int(vals.size),
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


def spearman_or_nan(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with graceful handling of empty inputs."""

    from scipy.stats import spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    res = spearmanr(x, y)
    return float(getattr(res, "statistic", getattr(res, "correlation", math.nan)))


def summarize_bias_chain(
    df,
    scatter_dex: float,
    bias_column: str = "bias_tinker10",
) -> dict[str, float | int | str]:
    """Return scalar diagnostics for the GalSBI -> SHMR -> bias chain."""

    result: dict[str, float | int | str] = {
        "n_rows": int(len(df)),
        "scatter_dex_requested": float(scatter_dex),
        "finite_log10_mstar_fraction": float(np.isfinite(df["log10_mstar"].to_numpy()).mean()),
        "finite_log10_mhalo_fraction": float(np.isfinite(df["log10_mhalo"].to_numpy()).mean()),
    }
    if bias_column in df:
        result["finite_bias_fraction"] = float(np.isfinite(df[bias_column].to_numpy()).mean())
    else:
        result["finite_bias_fraction"] = 0.0

    result["rho_log10_mstar_log10_mhalo"] = spearman_or_nan(
        df["log10_mstar"].to_numpy(),
        df["log10_mhalo"].to_numpy(),
    )
    if bias_column in df:
        result["rho_log10_mhalo_bias"] = spearman_or_nan(
            df["log10_mhalo"].to_numpy(),
            df[bias_column].to_numpy(),
        )
    else:
        result["rho_log10_mhalo_bias"] = math.nan

    if {"log10_mhalo_median", "log10_mhalo"}.issubset(df.columns):
        delta = df["log10_mhalo"].to_numpy() - df["log10_mhalo_median"].to_numpy()
        delta = delta[np.isfinite(delta)]
        result["scatter_dex_measured"] = float(np.std(delta)) if delta.size else math.nan
    else:
        result["scatter_dex_measured"] = math.nan

    for col in ["redshift", "log10_mstar", "log10_mhalo", bias_column]:
        if col in df:
            vals = df[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                result[f"{col}_p01"] = float(np.percentile(vals, 1))
                result[f"{col}_p50"] = float(np.percentile(vals, 50))
                result[f"{col}_p99"] = float(np.percentile(vals, 99))

    return result
