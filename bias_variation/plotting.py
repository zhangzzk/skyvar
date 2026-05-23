"""Diagnostic plots for the bias-variation prototype."""

from __future__ import annotations

from pathlib import Path

import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


MAP_PLOT_CFG = {
    "map_frac": 0.75,
    "hist_frac": 0.15,
    "gap": 0.06,
    "left": 0.06,
    "width_in": 12.0,
    "margin_top_in": 0.2,
    "margin_bottom_in": 0.5,
    "vspace_in": 0.6,
    "dpi": 200,
    "label_fontsize": 18,
    "tick_fontsize": 14,
    "hist_label_fontsize": 16,
    "hist_tick_fontsize": 13,
    "hist_curve_lw": 1.8,
}


def _finite_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _sample_finite_rows(
    df: pd.DataFrame,
    columns: list[str],
    max_points: int,
    seed: int,
) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.DataFrame()
    out = df[available].replace([np.inf, -np.inf], np.nan).dropna()
    if len(out) > max_points:
        out = out.sample(n=int(max_points), random_state=int(seed))
    return out


def _finite_positive_limits(values: np.ndarray, low: float = 0.2, high: float = 99.8) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0e-3, 1.0
    vmin, vmax = np.percentile(values, [low, high])
    if vmin <= 0 or not np.isfinite(vmin):
        vmin = float(np.nanmin(values[values > 0]))
    if vmin == vmax:
        pad = max(abs(float(vmin)) * 0.05, 1.0e-6)
        return float(max(vmin - pad, 1.0e-12)), float(vmax + pad)
    return float(vmin), float(vmax)


def _finite_limits(values: np.ndarray, low: float = 0.2, high: float = 99.8) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [low, high])
    if vmin == vmax:
        pad = max(abs(float(vmin)) * 0.05, 1.0e-6)
        return float(vmin - pad), float(vmax + pad)
    return float(vmin), float(vmax)


def _scatter_with_running_median(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    log_y: bool = False,
    color: str = "#2f6f8f",
) -> None:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if log_y:
        valid &= y > 0
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return

    ax.scatter(x, y, s=3.0, alpha=0.08, color=color, edgecolors="none", rasterized=True)
    if x.size >= 80:
        edges = np.percentile(x, np.linspace(1.0, 99.0, 18))
        edges = np.unique(edges)
        if edges.size >= 4:
            centers = []
            medians = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (x >= lo) & (x < hi)
                if np.count_nonzero(mask) >= 10:
                    centers.append(0.5 * (lo + hi))
                    medians.append(float(np.nanmedian(y[mask])))
            if centers:
                ax.plot(centers, medians, color="k", lw=1.4)

    ax.set_xlim(*_finite_limits(x))
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(*_finite_positive_limits(y))
    else:
        ax.set_ylim(*_finite_limits(y))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)


def _dual_hexbin(
    ax,
    x_unsel: np.ndarray,
    y_unsel: np.ndarray,
    x_sel: np.ndarray,
    y_sel: np.ndarray,
    *,
    x_label: str,
    y_label: str,
    log_y: bool = False,
) -> None:
    """Overlay unselected (Blues) and selected (Oranges) samples as 2-D histograms."""

    def _clean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(x) & np.isfinite(y)
        if log_y:
            mask &= y > 0
        return x[mask], y[mask]

    x_u, y_u = _clean(x_unsel, y_unsel)
    x_s, y_s = _clean(x_sel, y_sel)

    if x_u.size == 0 and x_s.size == 0:
        ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        return

    x_all = np.concatenate([x_u, x_s]) if x_u.size and x_s.size else (x_u if x_u.size else x_s)
    y_all = np.concatenate([y_u, y_s]) if y_u.size and y_s.size else (y_u if y_u.size else y_s)
    xlim = _finite_limits(x_all)
    ylim = _finite_positive_limits(y_all) if log_y else _finite_limits(y_all)
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    norm = LogNorm()

    if x_u.size > 0:
        ax.hexbin(x_u, y_u, gridsize=38, cmap="Blues", mincnt=1,
                  norm=norm, extent=extent, alpha=0.75, linewidths=0.0, rasterized=True)
    if x_s.size > 0:
        ax.hexbin(x_s, y_s, gridsize=38, cmap="Oranges", mincnt=1,
                  norm=norm, extent=extent, alpha=0.72, linewidths=0.0, rasterized=True)

    # running medians
    handles = []
    for x_arr, y_arr, color, label in [
        (x_u, y_u, "#1a4a7a", "unselected"),
        (x_s, y_s, "#7a2e04", "selected"),
    ]:
        if x_arr.size >= 80:
            edges = np.percentile(x_arr, np.linspace(1.0, 99.0, 18))
            edges = np.unique(edges)
            if edges.size >= 4:
                centers, medians = [], []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    mask = (x_arr >= lo) & (x_arr < hi)
                    if np.count_nonzero(mask) >= 10:
                        centers.append(0.5 * (lo + hi))
                        medians.append(float(np.nanmedian(y_arr[mask])))
                if centers:
                    (line,) = ax.plot(centers, medians, color=color, lw=1.6, label=label)
                    handles.append(line)
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=7, loc="best")

    ax.set_xlim(*xlim)
    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.2)


def _mstar_source_note(df: pd.DataFrame) -> str | None:
    attr_source = df.attrs.get("mstar_source")
    if attr_source is not None:
        source = str(attr_source)
        if source == "log10_column:log10_surviv_stellar_mass":
            return "SPS surviving stellar mass"
        if source == "log10_column:log10_stellar_mass":
            return "SPS formed stellar mass"
        if source.startswith("luminosity_proxy:"):
            return "luminosity-proxy M*: one-to-one with abs mag"
        return source
    if "mstar_source" not in df.columns:
        return None
    values = df["mstar_source"].dropna().astype(str)
    if values.empty:
        return None
    source = values.iloc[0]
    if source == "log10_column:log10_surviv_stellar_mass":
        return "SPS surviving stellar mass"
    if source == "log10_column:log10_stellar_mass":
        return "SPS formed stellar mass"
    if source.startswith("luminosity_proxy:"):
        return "luminosity-proxy M*: one-to-one with abs mag"
    if source.startswith("log10_column:"):
        return f"log10 M* from {source.split(':', 1)[1]}"
    if source.startswith("column:"):
        return f"M* from {source.split(':', 1)[1]}"
    return source


def _plot_y_vs_photometric_grid(
    df: pd.DataFrame,
    output_dir: Path,
    tag: str,
    *,
    y_col: str,
    y_label: str,
    filename_suffix: str,
    log_y: bool,
    color: str,
    max_points: int,
    seed: int,
    df_selected: pd.DataFrame | None = None,
) -> Path | None:
    photometric_panels = [
        (col, label)
        for col, label in [
            ("r", "apparent r"),
            ("abs_mag", "absolute r magnitude"),
            ("redshift", "redshift"),
            ("Re_semimajor", "semimajor Re [arcsec]"),
            ("sersic_n", "Sersic n"),
            ("BA", "axis ratio BA"),
        ]
        if col in df.columns
    ]
    required = [x for x, _ in photometric_panels] + [y_col]
    mstar_note = _mstar_source_note(df) if y_col == "log10_mstar" else None
    plot_df = _sample_finite_rows(df, required, max_points=max_points, seed=seed)
    if plot_df.empty:
        return None

    dual = df_selected is not None and y_col in df_selected.columns
    if dual:
        plot_df_sel = _sample_finite_rows(df_selected, required, max_points=max_points, seed=seed + 1)
        dual = not plot_df_sel.empty

    n_panels = len(photometric_panels)
    if n_panels == 0:
        return None
    n_cols = min(3, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.3 * n_cols, 3.55 * n_rows),
        constrained_layout=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    for ax, (x_col, x_label) in zip(axes, photometric_panels):
        if dual and x_col in plot_df_sel.columns:
            _dual_hexbin(
                ax,
                plot_df[x_col].to_numpy(float),
                plot_df[y_col].to_numpy(float),
                plot_df_sel[x_col].to_numpy(float),
                plot_df_sel[y_col].to_numpy(float),
                x_label=x_label,
                y_label=y_label,
                log_y=log_y,
            )
        else:
            _scatter_with_running_median(
                ax,
                plot_df[x_col].to_numpy(float),
                plot_df[y_col].to_numpy(float),
                x_label=x_label,
                y_label=y_label,
                log_y=log_y,
                color=color,
            )
        if mstar_note is not None and y_col == "log10_mstar" and x_col == "abs_mag":
            ax.text(
                0.04,
                0.94,
                mstar_note,
                ha="left",
                va="top",
                fontsize=8,
                transform=ax.transAxes,
            )
    path = output_dir / f"{tag}_{filename_suffix}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_subset_diagnostics(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Write first-pass mass/bias diagnostic plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    x, y = _finite_xy(df, "log10_mstar", "log10_mhalo")
    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="viridis")
    ax.set_xlabel("log10 stellar mass input/proxy")
    ax.set_ylabel("log10 halo mass from Moster+13 inverse")
    ax.set_title("SHMR inverse check")
    path = output_dir / "subset_logmstar_logmhalo.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    if "bias_tinker10" in df:
        x, y = _finite_xy(df, "log10_mhalo", "bias_tinker10")
        fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
        ax.hexbin(x, y, gridsize=55, mincnt=1, cmap="magma")
        ax.set_xlabel("log10 halo mass")
        ax.set_ylabel("Tinker+10 halo bias")
        ax.set_title("Halo mass to bias check")
        path = output_dir / "subset_logmhalo_bias.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    cols = ["log10_mstar", "log10_mhalo"]
    if "bias_tinker10" in df:
        cols.append("bias_tinker10")
    fig, axes = plt.subplots(1, len(cols), figsize=(4.1 * len(cols), 3.6), constrained_layout=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        vals = df[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=60, histtype="stepfilled", color="#4c6f8f", alpha=0.82)
        ax.set_xlabel(col)
        ax.set_ylabel("count")
    path = output_dir / "subset_chain_histograms.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    written.extend(plot_bias_property_diagnostics(df, output_dir, tag="subset"))
    return written


def plot_bias_property_diagnostics(
    df: pd.DataFrame,
    output_dir: str | Path,
    tag: str = "bias_source",
    max_points: int = 120_000,
    seed: int = 20260516,
    redshift_range: tuple[float, float] | None = None,
    df_selected: pd.DataFrame | None = None,
) -> list[Path]:
    """Write source-catalog scatter diagnostics for assigned mass and bias.

    The input table should already contain the bias-chain columns produced by
    :func:`bias_variation.halo_bias.assign_galaxy_bias`.  For large
    source catalogs the scatter panels use a deterministic subsample.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if redshift_range is not None:
        if "redshift" not in df.columns:
            raise KeyError("redshift_range was requested but the table has no 'redshift' column.")
        z_min, z_max = map(float, redshift_range)
        df = df[(df["redshift"] >= z_min) & (df["redshift"] < z_max)]
        if df_selected is not None and "redshift" in df_selected.columns:
            df_selected = df_selected[
                (df_selected["redshift"] >= z_min) & (df_selected["redshift"] < z_max)
            ]
    written: list[Path] = []

    path = _plot_y_vs_photometric_grid(
        df,
        output_dir,
        tag,
        y_col="bias_tinker10",
        y_label="Tinker+10 bias",
        filename_suffix="bias_vs_photometric_properties",
        log_y=False,
        color="#1f6f8b",
        max_points=max_points,
        seed=seed,
        df_selected=df_selected,
    )
    if path is not None:
        written.append(path)

    path = _plot_y_vs_photometric_grid(
        df,
        output_dir,
        tag,
        y_col="log10_mstar",
        y_label="log10 Mstar [Msun]",
        filename_suffix="mstar_vs_photometric_properties",
        log_y=False,
        color="#4f6f9f",
        max_points=max_points,
        seed=seed,
        df_selected=df_selected,
    )
    if path is not None:
        written.append(path)

    relation_columns = ["log10_mstar", "log10_mhalo", "bias_tinker10"]
    relation_df = _sample_finite_rows(df, relation_columns, max_points=max_points, seed=seed)
    dual_relations = df_selected is not None
    if dual_relations:
        relation_df_sel = _sample_finite_rows(
            df_selected, relation_columns, max_points=max_points, seed=seed + 1
        )
        dual_relations = not relation_df_sel.empty
    if not relation_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1), constrained_layout=True)
        if dual_relations:
            _dual_hexbin(
                axes[0],
                relation_df["log10_mstar"].to_numpy(float),
                relation_df["log10_mhalo"].to_numpy(float),
                relation_df_sel["log10_mstar"].to_numpy(float),
                relation_df_sel["log10_mhalo"].to_numpy(float),
                x_label="log10 Mstar [Msun]",
                y_label="log10 Mhalo [Msun]",
                log_y=False,
            )
            _dual_hexbin(
                axes[1],
                relation_df["log10_mhalo"].to_numpy(float),
                relation_df["bias_tinker10"].to_numpy(float),
                relation_df_sel["log10_mhalo"].to_numpy(float),
                relation_df_sel["bias_tinker10"].to_numpy(float),
                x_label="log10 Mhalo [Msun]",
                y_label="Tinker+10 bias",
                log_y=True,
            )
        else:
            _scatter_with_running_median(
                axes[0],
                relation_df["log10_mstar"].to_numpy(float),
                relation_df["log10_mhalo"].to_numpy(float),
                x_label="log10 Mstar [Msun]",
                y_label="log10 Mhalo [Msun]",
                log_y=False,
                color="#5b6f9f",
            )
            _scatter_with_running_median(
                axes[1],
                relation_df["log10_mhalo"].to_numpy(float),
                relation_df["bias_tinker10"].to_numpy(float),
                x_label="log10 Mhalo [Msun]",
                y_label="Tinker+10 bias",
                log_y=True,
                color="#8b4f7f",
            )
        path = output_dir / f"{tag}_mass_bias_relations.png"
        fig.savefig(path, dpi=220)
        plt.close(fig)
        written.append(path)

    return written


def _nanmedian_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[1], np.nan, dtype=float)
    for i in range(values.shape[1]):
        col = values[:, i]
        if np.isfinite(col).any():
            out[i] = float(np.nanmedian(col))
    return out


def _weighted_global_profile(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    out = np.full(values.shape[1], np.nan, dtype=float)
    for i in range(values.shape[1]):
        v = values[:, i]
        w = weights[:, i]
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if np.any(valid):
            out[i] = float(np.sum(v[valid] * w[valid]) / np.sum(w[valid]))
    return out


def _row_nanmedian(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape[0], np.nan, dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        if np.isfinite(row).any():
            out[i] = float(np.nanmedian(row))
    return out


def _full_healpix_map(
    pixels: np.ndarray,
    values: np.ndarray,
    nside: int,
    unseen: float = hp.UNSEEN,
    mask_zero: bool = False,
) -> np.ndarray:
    out = np.full(hp.nside2npix(int(nside)), unseen, dtype=float)
    pixels = np.asarray(pixels, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    valid = (pixels >= 0) & (pixels < len(out)) & np.isfinite(values)
    if mask_zero:
        valid &= values != 0
    out[pixels[valid]] = values[valid]
    return out


def _cartview_bounds(nside: int, pixels: np.ndarray, margin_deg: float = 1.0) -> dict[str, list[float]]:
    pixels = np.asarray(pixels, dtype=np.int64)
    if len(pixels) == 0:
        return {}
    lon, lat = hp.pix2ang(int(nside), pixels, lonlat=True)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(finite):
        return {}
    lon = lon[finite]
    lat = lat[finite]
    return {
        "lonra": [float(np.min(lon) - margin_deg), float(np.max(lon) + margin_deg)],
        "latra": [float(np.min(lat) - margin_deg), float(np.max(lat) + margin_deg)],
    }


def _project_ra_range(ra_range: tuple[float, float]) -> tuple[float, float, float]:
    """Project an RA interval to contiguous coordinates, matching SkyVar maps."""

    ra0_raw = float(ra_range[0])
    ra1_raw = float(ra_range[1])
    ra0 = ra0_raw % 360.0
    ra1 = ra1_raw % 360.0
    ra_span = (ra1 - ra0) % 360.0
    if np.isclose(ra_span, 0.0):
        ra_span = 360.0
    wraps = (ra0 > ra1) and (ra_span < 360.0)

    target_mid = 0.5 * (ra0_raw + ra1_raw)
    wrapped_mid = ra0 + 0.5 * ra_span
    k = int(np.round((target_mid - wrapped_mid) / 360.0))
    ra_lo = ra0 + 360.0 * k
    ra_hi = ra_lo + ra_span
    if wraps and ra_lo > 180.0:
        ra_lo -= 360.0
        ra_hi -= 360.0
    return float(ra_lo), float(ra_hi), float(ra_span)


def _project_ra_values(lon: np.ndarray, ra_range: tuple[float, float]) -> np.ndarray:
    """Project lon values into the same contiguous coordinates as _project_ra_range."""

    ra_lo, _, _ = _project_ra_range(ra_range)
    ra0 = float(ra_range[0]) % 360.0
    lon = np.asarray(lon, dtype=float) % 360.0
    lon_proj = ((lon - ra0 + 360.0) % 360.0) + ra0
    return lon_proj + (ra_lo - ra0)


def _finite_map_values(map_values: np.ndarray) -> np.ndarray:
    values = np.asarray(map_values, dtype=float)
    return values[(values != hp.UNSEEN) & np.isfinite(values)]


def _robust_limits(values: np.ndarray, symmetric_if_signed: bool = False) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [0.1, 99.9])
    if symmetric_if_signed and vmin < 0 < vmax:
        limit = max(abs(float(vmin)), abs(float(vmax)))
        return -limit, limit
    if vmin == vmax:
        pad = max(abs(float(vmin)) * 0.01, 1e-6)
        return float(vmin - pad), float(vmax + pad)
    return float(vmin), float(vmax)


def _draw_color_histogram(
    ax,
    values: np.ndarray,
    label: str,
    cmap: str,
    norm: Normalize,
    global_value: float,
) -> None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    vmin, vmax = float(norm.vmin), float(norm.vmax)
    in_range = values[(values >= vmin) & (values <= vmax)]
    if in_range.size:
        counts, edges = np.histogram(in_range, bins=64, range=(vmin, vmax), density=True)
    else:
        counts = np.zeros(64, dtype=float)
        edges = np.linspace(vmin, vmax, 65)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ymax = max(float(np.nanmax(counts)) if counts.size else 0.0, 1e-12) * 1.18

    gradient_x = np.linspace(vmin, vmax, 256)
    gradient = norm(gradient_x).reshape(1, -1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=[vmin, vmax, 0, ymax],
        origin="lower",
        vmin=0,
        vmax=1,
    )
    ax.fill_between(centers, 0, counts, color="grey", alpha=0.45)
    ax.plot(centers, counts, color="k", lw=MAP_PLOT_CFG["hist_curve_lw"])
    if np.isfinite(global_value):
        clipped = float(np.clip(global_value, vmin, vmax))
        ax.axvline(clipped, color="white", lw=1.7, ls="--")
        ax.axvline(clipped, color="0.1", lw=0.8, ls="--")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(0, ymax)
    ax.set_yticks([])
    max_abs = max(abs(vmin), abs(vmax))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if max_abs >= 1.0e4 else 4))
    if max_abs >= 1.0e4:
        exponent = int(np.floor(np.log10(max_abs)))
        scale = 10.0**exponent
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x / scale:.4g}"))
        label = f"{label} [1e{exponent}]"
    ax.set_xlabel(label, fontsize=MAP_PLOT_CFG["hist_label_fontsize"])
    ax.tick_params(axis="x", labelsize=MAP_PLOT_CFG["hist_tick_fontsize"])
    ax.grid(False)


def _panel_norm(values: np.ndarray, symmetric: bool, vcenter: float | None) -> Normalize:
    vmin, vmax = _robust_limits(values, symmetric_if_signed=symmetric)
    if vcenter is None or not np.isfinite(vcenter):
        return Normalize(vmin=vmin, vmax=vmax)
    center = float(vcenter)
    if center <= vmin:
        vmin = center - (vmax - center)
    elif center >= vmax:
        vmax = center + (center - vmin)
    return TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)


def _plot_healpix_cart_panels(
    panels: list[tuple],
    path: Path,
    nside: int,
    fallback_view_pixels: np.ndarray,
) -> None:
    """Write SkyVar-style map panels with adjacent color-histogram panels."""

    n_panels = len(panels)
    cfg = MAP_PLOT_CFG
    map_frac = cfg["map_frac"]
    hist_frac = cfg["hist_frac"]
    gap = cfg["gap"]
    map_width_in = cfg["width_in"]

    bounds = _cartview_bounds(nside, fallback_view_pixels, margin_deg=0.0)
    if bounds:
        ra_lo, ra_hi, ra_span = _project_ra_range(tuple(bounds["lonra"]))
        dec_lo, dec_hi = min(bounds["latra"]), max(bounds["latra"])
    else:
        ra_lo, ra_hi, ra_span = 0.0, 1.0, 1.0
        dec_lo, dec_hi = 0.0, 1.0

    row_aspect = max(float(dec_hi - dec_lo), 1e-6) / max(float(ra_span), 1e-6)
    row_h_in = map_width_in * map_frac * row_aspect
    fig_h = (
        n_panels * row_h_in
        + (n_panels - 1) * cfg["vspace_in"]
        + cfg["margin_top_in"]
        + cfg["margin_bottom_in"]
    )
    fig = plt.figure(figsize=(map_width_in, fig_h))

    for i, panel in enumerate(panels):
        map_values, title, cmap, symmetric, vcenter = panel
        row_bottom = (
            cfg["margin_bottom_in"] + (n_panels - 1 - i) * (row_h_in + cfg["vspace_in"])
        ) / fig_h
        map_ax = fig.add_axes([cfg["left"], row_bottom, map_frac, row_h_in / fig_h])
        hist_ax = fig.add_axes([cfg["left"] + map_frac + gap, row_bottom, hist_frac, row_h_in / fig_h])

        all_values = _finite_map_values(map_values)
        norm = _panel_norm(all_values, symmetric=symmetric, vcenter=vcenter)

        pix_res_deg = np.degrees(hp.nside2resol(int(nside)))
        n_ra = max(int(np.ceil(ra_span / pix_res_deg)), 2)
        n_dec = max(int(np.ceil((dec_hi - dec_lo) / pix_res_deg)), 2)
        ra_edges = np.linspace(ra_lo, ra_hi, n_ra + 1)
        dec_edges = np.linspace(dec_lo, dec_hi, n_dec + 1)
        ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
        dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
        ra_grid, dec_grid = np.meshgrid(ra_centers, dec_centers)
        theta = np.radians(90.0 - dec_grid.ravel())
        phi = np.radians(ra_grid.ravel() % 360.0)
        grid_pix = hp.ang2pix(int(nside), theta, phi)
        grid_vals = map_values[grid_pix].reshape(dec_grid.shape)
        grid_vals = np.where(
            (grid_vals == hp.UNSEEN) | ~np.isfinite(grid_vals), np.nan, grid_vals
        )
        ra_plot_edges = _project_ra_values(ra_edges, (ra_lo, ra_hi))
        map_ax.pcolormesh(
            ra_plot_edges,
            dec_edges,
            grid_vals,
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        map_ax.set_xlim(ra_hi, ra_lo)
        map_ax.set_ylim(dec_lo, dec_hi)
        map_ax.set_aspect("equal")
        map_ax.set_xlabel("RA [deg]", fontsize=cfg["label_fontsize"])
        map_ax.set_ylabel("Dec [deg]", fontsize=cfg["label_fontsize"])
        map_ax.tick_params(axis="both", labelsize=cfg["tick_fontsize"])
        for spine in map_ax.spines.values():
            spine.set_visible(False)

        _draw_color_histogram(
            hist_ax,
            all_values,
            title,
            cmap=cmap,
            norm=norm,
            global_value=float(np.nanmean(all_values)) if all_values.size else np.nan,
        )
    fig.savefig(path, dpi=cfg["dpi"], bbox_inches="tight")
    plt.close()


def plot_redshift_binned_bias_summary(
    maps: dict[str, np.ndarray],
    output_dir: str | Path,
    tag: str = "redshift_binned_bias",
) -> list[Path]:
    """Write compact diagnostics for reference-aligned redshift-binned bias maps."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    z = np.asarray(maps["z_centers"], dtype=float)
    mean_u = np.asarray(maps["mean_bias_unselected_z"], dtype=float)
    mean_s = np.asarray(maps["mean_bias_selected_z"], dtype=float)
    delta = np.asarray(maps["delta_bias_z"], dtype=float)
    count_u = np.asarray(maps["count_unselected_z"], dtype=float)
    count_s = np.asarray(maps["count_selected_z"], dtype=float)

    prof_u = _nanmedian_profile(mean_u)
    prof_s = _nanmedian_profile(mean_s)
    delta_of_profiles = prof_s - prof_u
    paired_delta_profile = _nanmedian_profile(delta)
    filled_u = np.isfinite(mean_u).sum(axis=0)
    filled_s = np.isfinite(mean_s).sum(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, constrained_layout=True)
    axes[0].plot(z, prof_u, color="#355f8d", lw=1.8, label="unselected")
    axes[0].plot(z, prof_s, color="#b64f2f", lw=1.8, label="selected")
    axes[0].set_ylabel("median mean bias")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.25)

    axes[1].plot(z, delta_of_profiles, color="#3f7f5f", lw=1.8, label="selected median - unselected median")
    axes[1].plot(z, paired_delta_profile, color="#6b7280", lw=1.2, ls="--", label="median paired pixel delta")
    axes[1].axhline(0.0, color="0.35", lw=0.8)
    axes[1].set_xlabel("redshift")
    axes[1].set_ylabel("bias difference")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    path = output_dir / f"{tag}_bias_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    width = np.median(np.diff(z)) if len(z) > 1 else 0.02
    ax.bar(z - 0.18 * width, filled_u, width=0.32 * width, color="#355f8d", alpha=0.82, label="unselected")
    ax.bar(z + 0.18 * width, filled_s, width=0.32 * width, color="#b64f2f", alpha=0.82, label="selected")
    ax.set_xlabel("redshift")
    ax.set_ylabel("filled pixel-z bins")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / f"{tag}_filled_bins.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(z, count_u.sum(axis=0), color="#355f8d", lw=1.8, label="unselected")
    ax.plot(z, count_s.sum(axis=0), color="#b64f2f", lw=1.8, label="selected")
    ax.set_xlabel("redshift")
    ax.set_ylabel("object count")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    path = output_dir / f"{tag}_object_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    return written


def plot_redshift_binned_bias_maps(
    maps: dict[str, np.ndarray],
    nside: int,
    output_dir: str | Path,
    tag: str = "redshift_binned_bias",
) -> list[Path]:
    """Write the canonical combined HEALPix map diagnostics.

    The n(z) workflow writes compact summary products rather than one file per
    scalar map.  Bias follows the same convention: a median/difference overview
    and an object-count overview, each with the color histogram embedded.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(maps["pixels"], dtype=np.int64)
    mean_u = np.asarray(maps["mean_bias_unselected_z"], dtype=float)
    mean_s = np.asarray(maps["mean_bias_selected_z"], dtype=float)
    count_u = np.asarray(maps["count_unselected_z"], dtype=float)
    count_s = np.asarray(maps["count_selected_z"], dtype=float)

    median_u = _row_nanmedian(mean_u)
    median_s = _row_nanmedian(mean_s)
    median_delta = median_s - median_u
    total_count_s = np.sum(count_s, axis=1)
    total_count_u = np.sum(count_u, axis=1)
    scalar_maps = {
        "unselected_median_bias": (
            _full_healpix_map(pixels, median_u, nside),
            "Median unselected bias",
            "viridis",
            False,
            None,
        ),
        "selected_median_bias": (
            _full_healpix_map(pixels, median_s, nside),
            "Median selected bias",
            "magma",
            False,
            None,
        ),
        "delta_median_bias": (
            _full_healpix_map(pixels, median_delta, nside),
            "Selected median - unselected median bias",
            "coolwarm",
            True,
            float(np.nanmean(median_delta)),
        ),
        "selected_object_count": (
            _full_healpix_map(pixels, total_count_s, nside, mask_zero=True),
            "Selected object count",
            "plasma",
            False,
            None,
        ),
        "unselected_object_count": (
            _full_healpix_map(pixels, total_count_u, nside, mask_zero=True),
            "Unselected object count",
            "plasma",
            False,
            None,
        ),
    }

    written: list[Path] = []
    path = output_dir / f"{tag}_map_medians_and_difference.png"
    _plot_healpix_cart_panels(
        [
            scalar_maps["unselected_median_bias"],
            scalar_maps["selected_median_bias"],
            scalar_maps["delta_median_bias"],
        ],
        path=path,
        nside=nside,
        fallback_view_pixels=pixels,
    )
    written.append(path)

    path = output_dir / f"{tag}_map_object_counts_panel.png"
    _plot_healpix_cart_panels(
        [
            scalar_maps["unselected_object_count"],
            scalar_maps["selected_object_count"],
        ],
        path=path,
        nside=nside,
        fallback_view_pixels=pixels,
    )
    written.append(path)
    return written


def plot_pixel_bz_examples(
    maps: dict[str, np.ndarray],
    output_dir: str | Path,
    tag: str = "redshift_binned_bias",
    n_examples: int | None = None,
    max_z_bins: int | None = 15,
    display_z_step: float | None = None,
) -> list[Path]:
    """Write n(z)-style global and per-pixel b(z) curves."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pixels = np.asarray(maps["pixels"], dtype=np.int64)
    z = np.asarray(maps["z_centers"], dtype=float)
    mean_u = np.asarray(maps["mean_bias_unselected_z"], dtype=float)
    mean_s = np.asarray(maps["mean_bias_selected_z"], dtype=float)
    count_u = np.asarray(maps["count_unselected_z"], dtype=np.int64)
    count_s = np.asarray(maps["count_selected_z"], dtype=np.int64)
    weight_u = np.asarray(maps.get("sum_weight_unselected_z", count_u), dtype=float)
    weight_s = np.asarray(maps.get("sum_weight_selected_z", count_s), dtype=float)

    if max_z_bins is not None and z.size > int(max_z_bins):
        target_bins = int(max_z_bins)
        if target_bins <= 0:
            raise ValueError("max_z_bins must be positive when provided.")
        source_edges = np.asarray(maps.get("z_edges"), dtype=float)
        if source_edges.size != z.size + 1:
            mid = 0.5 * (z[1:] + z[:-1])
            source_edges = np.concatenate(
                (
                    [z[0] - (mid[0] - z[0])],
                    mid,
                    [z[-1] + (z[-1] - mid[-1])],
                )
            )
        target_edges = np.linspace(float(source_edges[0]), float(source_edges[-1]), target_bins + 1)
        source_bin = np.digitize(z, target_edges) - 1
        source_bin[z == target_edges[-1]] = target_bins - 1

        def rebin_profile(
            mean: np.ndarray,
            count: np.ndarray,
            weight: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            rebinned_mean = np.full((mean.shape[0], target_bins), np.nan, dtype=float)
            rebinned_count = np.zeros((count.shape[0], target_bins), dtype=np.int64)
            rebinned_weight = np.zeros((weight.shape[0], target_bins), dtype=float)
            for k in range(target_bins):
                cols = np.where(source_bin == k)[0]
                if cols.size == 0:
                    continue
                w = weight[:, cols]
                rebinned_count[:, k] = np.sum(count[:, cols], axis=1)
                rebinned_weight[:, k] = np.sum(w, axis=1)
                finite = np.isfinite(mean[:, cols]) & np.isfinite(w)
                sum_wv = np.sum(np.where(finite, mean[:, cols] * w, 0.0), axis=1)
                rebinned_mean[:, k] = np.divide(
                    sum_wv,
                    rebinned_weight[:, k],
                    out=np.full(mean.shape[0], np.nan, dtype=float),
                    where=rebinned_weight[:, k] > 0,
                )
            return rebinned_mean, rebinned_count, rebinned_weight

        mean_u, count_u, weight_u = rebin_profile(mean_u, count_u, weight_u)
        mean_s, count_s, weight_s = rebin_profile(mean_s, count_s, weight_s)
        z = 0.5 * (target_edges[:-1] + target_edges[1:])

    selected_totals = np.sum(count_s, axis=1)
    if pixels.size == 0:
        return []
    if n_examples is None or n_examples <= 0:
        chosen = np.arange(len(pixels), dtype=np.int64)
    else:
        active = np.where(selected_totals > 0)[0]
        if active.size == 0:
            return []
        chosen = active[np.argsort(selected_totals[active])[::-1][: int(n_examples)]]

    global_u = _weighted_global_profile(mean_u, weight_u)
    global_s = _weighted_global_profile(mean_s, weight_s)
    global_delta = global_s - global_u

    plot_z = z
    if display_z_step is not None:
        step = float(display_z_step)
        if step > 0 and z.size >= 2:
            native_step = float(np.nanmedian(np.diff(np.sort(z))))
            if np.isfinite(native_step) and step < native_step:
                plot_z = np.arange(float(z[0]), float(z[-1]) + 0.5 * step, step)

    def interp_profile(profile: np.ndarray) -> np.ndarray:
        if plot_z is z:
            return profile
        valid = np.isfinite(z) & np.isfinite(profile)
        if valid.sum() < 2:
            return np.full_like(plot_z, np.nan, dtype=float)
        return np.interp(plot_z, z[valid], profile[valid], left=np.nan, right=np.nan)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.0, 1.35]},
    )
    ax_b, ax_d = axes
    pixel_count = len(chosen)
    pixel_alpha = min(0.18, max(0.012, 22.0 / max(pixel_count, 1)))
    delta_alpha = min(0.20, max(0.014, 24.0 / max(pixel_count, 1)))
    pixel_lw = 0.8 if pixel_count <= 200 else 0.45
    delta_lw = 0.7 if pixel_count <= 200 else 0.4
    for row in chosen:
        row_u = interp_profile(mean_u[row])
        row_s = interp_profile(mean_s[row])
        ax_b.plot(plot_z, row_u, color="#355f8d", lw=pixel_lw, alpha=pixel_alpha, rasterized=True)
        ax_b.plot(plot_z, row_s, color="#b64f2f", lw=pixel_lw, alpha=pixel_alpha, rasterized=True)
        ax_d.plot(plot_z, row_s - row_u, color="0.45", lw=delta_lw, alpha=delta_alpha, rasterized=True)

    ax_b.plot(plot_z, interp_profile(global_u), color="#183a5a", lw=2.6, label="global unselected")
    ax_b.plot(plot_z, interp_profile(global_s), color="#8f2d1f", lw=2.6, label="global selected")
    ax_b.set_ylabel("mean bias")
    ax_b.legend(frameon=False)
    ax_b.grid(alpha=0.25)

    ax_d.plot(plot_z, interp_profile(global_delta), color="k", lw=2.2, label="global selected - unselected")
    ax_d.axhline(0.0, color="0.45", lw=0.9)
    ax_d.set_xlabel("redshift")
    ax_d.set_ylabel("bias delta")
    ax_d.legend(frameon=False)
    ax_d.grid(alpha=0.25)
    ax_d.set_xlim(float(plot_z[0]), float(plot_z[-1]))

    path = output_dir / f"{tag}_pixel_bz_examples.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    rows = []
    for row in chosen:
        for z_index, z_value in enumerate(z):
            rows.append(
                {
                    "pixel": int(pixels[row]),
                    "z_center": float(z_value),
                    "mean_bias_unselected": mean_u[row, z_index],
                    "mean_bias_selected": mean_s[row, z_index],
                    "count_unselected": int(count_u[row, z_index]),
                    "count_selected": int(count_s[row, z_index]),
                }
            )
    csv_path = output_dir / f"{tag}_pixel_bz_examples.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return [path, csv_path]


def plot_tomographic_bias_summary(
    maps: dict[str, np.ndarray],
    output_dir: str | Path,
    tag: str = "tomo_binned_bias",
    max_pixel_curves: int = 120,
) -> list[Path]:
    """Write n(z)-style tomographic true-redshift and bias diagnostics."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    pixels = np.asarray(maps["pixels"], dtype=np.int64)
    z = np.asarray(maps["z_centers"], dtype=float)
    z_edges = np.asarray(maps["z_edges"], dtype=float)
    dz = np.diff(z_edges)
    tomo_edges = np.asarray(maps["tomo_bin_edges"], dtype=float)
    mean_t = np.asarray(maps["mean_bias_tomo_z"], dtype=float)
    weight_t = np.asarray(maps["sum_weight_tomo_z"], dtype=float)
    count_t = np.asarray(maps["count_tomo_z"], dtype=np.int64)

    if mean_t.size == 0:
        return written

    n_tomo = mean_t.shape[0]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(n_tomo, 1)))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )
    ax_nz, ax_bias = axes
    csv_rows = []
    for i in range(n_tomo):
        color = colors[i]
        w_z = np.sum(weight_t[i], axis=0)
        total_w = float(np.sum(w_z))
        dndz = np.divide(w_z, total_w * dz, out=np.zeros_like(w_z, dtype=float), where=(total_w > 0) & (dz > 0))
        global_bias = _weighted_global_profile(mean_t[i], weight_t[i])
        label = f"Bin {i}: {tomo_edges[i]:.2f}-{tomo_edges[i + 1]:.2f}"

        ax_nz.plot(z, dndz, color=color, lw=2.1, label=label)

        totals = np.sum(weight_t[i], axis=1)
        active = np.where(totals > 0)[0]
        if active.size:
            n_plot = min(active.size, int(max_pixel_curves))
            chosen = active[np.argsort(totals[active])[::-1][:n_plot]]
            alpha = min(0.18, 10.0 / max(n_plot, 1))
            for row in chosen:
                ax_bias.plot(z, mean_t[i, row], color=color, alpha=alpha, lw=0.65)
        ax_bias.plot(z, global_bias, color=color, lw=2.35, label=label)

        for z_index, z_value in enumerate(z):
            csv_rows.append(
                {
                    "tomo_index": int(i),
                    "z_pho_min": float(tomo_edges[i]),
                    "z_pho_max": float(tomo_edges[i + 1]),
                    "z_center": float(z_value),
                    "global_dndz": float(dndz[z_index]),
                    "global_mean_bias": float(global_bias[z_index])
                    if np.isfinite(global_bias[z_index])
                    else np.nan,
                    "sum_weight": float(w_z[z_index]),
                    "count": int(np.sum(count_t[i, :, z_index])),
                }
            )

    ax_nz.set_ylabel("weighted n(z)")
    ax_nz.legend(frameon=False, ncol=2, fontsize=8)
    ax_nz.grid(alpha=0.25)
    ax_bias.set_xlabel("true redshift")
    ax_bias.set_ylabel("mean bias")
    ax_bias.legend(frameon=False, ncol=2, fontsize=8)
    ax_bias.grid(alpha=0.25)
    if len(z):
        ax_bias.set_xlim(float(z[0]), float(z[-1]))

    path = output_dir / f"{tag}_tomographic_bins_bias.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    csv_path = output_dir / f"{tag}_tomographic_bins_bias.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    written.append(csv_path)

    nside_value = maps.get("nside")
    if nside_value is not None:
        nside = int(np.asarray(nside_value).item())
    else:
        max_pixel = int(np.max(pixels)) if len(pixels) else 0
        nside = 1
        while hp.nside2npix(nside) <= max_pixel:
            nside *= 2

    panels = []
    for i in range(n_tomo):
        median_bias = _row_nanmedian(mean_t[i])
        panels.append(
            (
                _full_healpix_map(pixels, median_bias, nside),
                f"Median bias bin {i}: {tomo_edges[i]:.2f}-{tomo_edges[i + 1]:.2f}",
                "viridis",
                False,
                None,
            )
        )
    map_path = output_dir / f"{tag}_tomographic_median_bias_maps.png"
    _plot_healpix_cart_panels(panels, path=map_path, nside=nside, fallback_view_pixels=pixels)
    written.append(map_path)

    return written
