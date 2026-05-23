"""SkyVar selection wrapper for bias-variation b(z) products.

This module intentionally does not implement an independent galaxy selection.
It uses ``src.selection`` for per-pixel source draws, classifier-compatible
observing conditions, detection/SNR/MagLim cuts, and tomographic bin assignment.
The only workflow-specific change is the accumulator: instead of building
selected ``n(z)`` histograms, it builds selected/unselected mean-bias maps
``b(z, pixel)`` from the preassigned ``bias_tinker10`` column.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed

from . import config, plotting, utils

LOGGER = logging.getLogger(__name__)

SKYVAR_SELECTION_COLUMNS = (
    "detection",
    "r_input_p",
    "Re_input_p",
    "psf_fwhm_input_p",
    "pixel_rms_input_p",
    "zero_point_input_p",
    "pixel_size_input_p",
    "redshift_input_p",
)

SOURCE_COLUMNS = (
    "sersic_n",
    "r",
    "Re",
    "redshift",
    "BA",
    "angle",
    "bias_tinker10",
)

CLASSIFIED_COLUMNS = (
    *SKYVAR_SELECTION_COLUMNS,
    "pix_idx_input_p",
    "bias_tinker10_input_p",
)

DEFAULT_MODEL_JSON = config.PATHS["model_json"] or None
DEFAULT_BLENDEMU_DIR = config.PATHS["blendemu_dir"] or None


def require_columns(df: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def load_skyvar_selection_module():
    """Load the default SkyVar selection module.

    Some batch environments used for classification do not have plotting
    dependencies such as pyccl.  The selection functions needed here do not use
    those plotting hooks, so install a small stub before importing if needed.
    """

    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    if importlib.util.find_spec("pyccl") is None and "src.plotting" not in sys.modules:
        plotting_stub = types.ModuleType("src.plotting")

        def _plotting_unavailable(*args, **kwargs):
            raise RuntimeError("src.plotting is unavailable because pyccl is not installed.")

        plotting_stub.plt_map = _plotting_unavailable
        sys.modules["src.plotting"] = plotting_stub

    return importlib.import_module("src.selection")


def load_nz_utils(blendemu_dir: str | None, skyvar_selection) -> object:
    resolved = blendemu_dir or DEFAULT_BLENDEMU_DIR or skyvar_selection.config.BLENDEMU_DIR
    if not resolved:
        raise FileNotFoundError("Set BLENDEMU_DIR or pass --blendemu-dir; nz_utils.py is required.")
    path = Path(resolved).expanduser()
    if not (path / "nz_utils.py").exists():
        raise FileNotFoundError(f"Expected nz_utils.py under {path}.")
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
    module = importlib.import_module("nz_utils")
    if not hasattr(module, "icat2cla_v2"):
        raise AttributeError(f"{module.__file__} does not define icat2cla_v2.")
    return module


def load_model(model_json: str | None, skyvar_selection, n_jobs: int) -> xgb.Booster:
    model_path = Path(model_json or DEFAULT_MODEL_JSON or skyvar_selection.config.PATHS["model_json"]).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Detection model JSON not found: {model_path}")
    booster = xgb.Booster({"n_jobs": int(n_jobs)})
    booster.load_model(str(model_path))
    return booster


def find_chunks(path: str | Path) -> list[Path]:
    chunks_dir = Path(path)
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Classified chunk directory does not exist: {chunks_dir}")
    chunks = sorted(chunks_dir.glob("*.feather"))
    if not chunks:
        raise FileNotFoundError(f"No Feather classified chunks found under {chunks_dir}")
    return chunks


def save_chunk(classified: pd.DataFrame, chunks_dir: str | Path, start: int, overwrite: bool) -> Path:
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    path = chunks_dir / f"raw_preds_chunk_{int(start):08d}.feather"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Classified chunk already exists: {path}. Pass overwrite to replace it.")
    classified.reset_index(drop=True).to_feather(path)
    return path


def healpix_pixel_area_arcmin2(nside: int) -> float:
    return float(hp.nside2pixarea(int(nside), degrees=True) * 3600.0)


def source_density_for_n_pop(source_meta: dict[str, object] | None) -> float:
    if source_meta is None or source_meta.get("source_density_arcmin2") is None:
        raise ValueError(
            "Source density not found in source-table summary. "
            "Rebuild the bias source table with `halo_bias assign-source` to populate "
            "'source_density_arcmin2' in the summary JSON."
        )
    density = float(source_meta["source_density_arcmin2"])
    if density <= 0:
        raise ValueError(f"Source density must be positive, got {density}.")
    if density < 10.0 or density > 1000.0:
        LOGGER.warning(
            "Source density %.6g arcmin^-2 is outside the broad expected range [10, 1000].",
            density,
        )
    return density


def n_pop_from_source_density(density_arcmin2: float, nside_sim: int) -> tuple[int, float]:
    pixel_area_arcmin2 = healpix_pixel_area_arcmin2(int(nside_sim))
    n_pop = max(1, int(round(float(density_arcmin2) * pixel_area_arcmin2)))
    return n_pop, pixel_area_arcmin2


def source_table_summary_path(source_table: str | Path) -> Path:
    path = Path(source_table)
    return path.with_name(f"{path.stem}_summary.json")


def load_source_table_summary(source_table: str | Path) -> tuple[dict[str, object], Path | None]:
    path = source_table_summary_path(source_table)
    if not path.exists():
        return {}, None
    return json.loads(path.read_text(encoding="utf-8")), path


def source_density_from_summary(
    summary: dict[str, object],
) -> tuple[float | None, str | None, dict[str, object] | None]:
    value = summary.get("source_density_arcmin2")
    if value is not None:
        return float(value), str(summary.get("source_density_source", "source_table_summary")), summary

    diagnostic = summary.get("source_density_catalog_diagnostic")
    if isinstance(diagnostic, dict):
        value = diagnostic.get("post_quality_cut_density_arcmin2")
        if value is not None:
            return float(value), "source_density_catalog_diagnostic", diagnostic
    return None, None, None


def source_density_meta_from_table(source_table: str | Path | None) -> dict[str, object] | None:
    if not source_table:
        return None
    summary, summary_path = load_source_table_summary(source_table)
    density, density_source, diagnostic = source_density_from_summary(summary)
    if density is None:
        return None
    if isinstance(diagnostic, dict) and diagnostic.get("reliable_for_requested_source_pixels_density") is False:
        LOGGER.warning(
            "Source density comes from a catalog flagged as cap-limited; regenerate the GalSBI-SPS pool "
            "with density-faithful auto caps before using it for production n_pop."
        )
    return {
        "source_density_arcmin2": density,
        "source_density_source": density_source,
        "source_density_summary": str(summary_path) if summary_path else None,
        "source_density_diagnostic": diagnostic,
    }


def selection_mode(value: str | None) -> str | None:
    """Normalize CLI selection-mode values."""
    if value is None:
        return None
    value = str(value).strip().lower()
    return None if value in ("", "none") else value


def observing_conditions_mode(value: str | None) -> str:
    """Normalize observing-condition mode values."""
    mode = "maps" if value is None else str(value).strip().lower()
    aliases = {
        "map": "maps",
        "maps": "maps",
        "varying": "maps",
        "systematics": "maps",
        "constant": "constant",
        "baseline": "constant",
        "sample_variance": "constant",
        "sample-variance": "constant",
    }
    if mode in aliases:
        return aliases[mode]
    raise ValueError(
        f"Unknown observing-condition mode {value!r}. Use 'maps' or 'constant'."
    )


def resolve_observing_conditions_mode(args: argparse.Namespace) -> str:
    return observing_conditions_mode(args.observing_conditions_mode)


def observing_conditions_metadata(mode: str, skyvar_selection) -> dict[str, object]:
    obs = skyvar_selection.config.OBS_CONDITIONS
    return {
        "observing_conditions_mode": mode,
        "psf_fwhm_mode": "constant_baseline" if mode == "constant" else "map",
        "pixel_rms_mode": "constant_baseline" if mode == "constant" else "map",
        "galactic_extinction_mode": "zero" if mode == "constant" else "map",
        "baseline_pixel_size": float(obs["pixel_size"]),
        "baseline_zero_point": float(obs["zero_point"]),
        "baseline_psf_fwhm": float(obs["psf_fwhm_nominal"]),
        "baseline_pixel_rms": float(obs["pixel_rms_nominal"]),
        "detec_mag_bound": float(obs["detec_mag_bound"]),
    }


def npz_scalar_string(
    values: dict[str, np.ndarray],
    key: str,
    default: str | None = None,
) -> str | None:
    if key not in values:
        return default
    arr = np.asarray(values[key])
    if arr.shape == ():
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar NPZ value for {key!r}, got shape {arr.shape}.")


def load_bias_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load the compact source table produced by ``halo_bias assign-source``."""

    if not args.source_table:
        raise ValueError(
            "Pass --source-table with a bias-assigned Feather table. Build one with "
            "`python -m bias_variation.halo_bias assign-source --catalog <galsbi.cat> "
            "--output-table <source.feather>`."
        )
    source_table = Path(args.source_table).expanduser()
    if not source_table.exists():
        raise FileNotFoundError(f"Bias source table does not exist: {source_table}")

    df = pd.read_feather(source_table)
    require_columns(df, SOURCE_COLUMNS)
    source_summary, source_summary_path = load_source_table_summary(source_table)
    density_from_summary, density_source, density_diagnostic = source_density_from_summary(source_summary)

    density_override = getattr(args, "source_density_arcmin2", None)
    if density_override is not None:
        density_from_summary = float(density_override)
        density_source = "cli_override"
        density_diagnostic = None

    diagnostic_columns = [
        "r",
        "redshift",
        "Re_semimajor",
        "sersic_n",
        "BA",
        "log10_mstar",
        "log10_mhalo",
        "bias_tinker10",
    ]
    diagnostic_df = df[[col for col in diagnostic_columns if col in df.columns]].copy()

    source = pd.DataFrame({col: df[col].to_numpy(np.float32) for col in SOURCE_COLUMNS})
    density_meta = {
        "source_density_arcmin2": density_from_summary,
        "source_density_source": density_source,
    } if density_from_summary is not None else None
    density_arcmin2 = source_density_for_n_pop(density_meta)
    meta = {
        "source_table": str(source_table),
        "source_summary": str(source_summary_path) if source_summary_path else None,
        "source_rows": int(len(df)),
        "source_density_arcmin2": density_arcmin2,
        "source_density_source": density_source or "config_fallback",
        "source_density_diagnostic": density_diagnostic,
        "source_density_override_arcmin2": getattr(args, "source_density_arcmin2", None),
        "bias_assignment_location": "bias_variation.halo_bias assign-source",
    }
    return source, diagnostic_df, meta


def z_config_overrides() -> tuple[float | None, float | None, int | None]:
    settings = config.FULL_WORKFLOW_SETTINGS
    return settings["z_min"], settings["z_max"], settings["z_bins"]


def z_edges(skyvar_selection, args: argparse.Namespace) -> np.ndarray:
    settings = skyvar_selection.config.ANALYSIS_SETTINGS
    z_min_override, z_max_override, z_bins_override = z_config_overrides()
    z_min = float(settings["z_min"] if z_min_override is None else z_min_override)
    z_max = float(settings["z_max"] if z_max_override is None else z_max_override)
    z_bins = int(settings["z_bins"] if z_bins_override is None else z_bins_override)
    if z_bins <= 0:
        raise ValueError("SKYVAR_BIAS_Z_BINS must be positive.")
    if not z_max > z_min:
        raise ValueError("Configured z_max must be greater than z_min.")
    return np.linspace(z_min, z_max, z_bins + 1)


def load_cached_base_maps(
    args: argparse.Namespace,
    observing_conditions_mode: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    if any(value is not None for value in z_config_overrides()):
        raise ValueError(
            "Cached classified chunks use the z grid from --base-redshift-maps; "
            "unset SKYVAR_BIAS_Z_MIN/Z_MAX/Z_BINS."
        )
    base_maps = utils.load_npz_dict(args.base_redshift_maps)
    cached_mode_raw = npz_scalar_string(base_maps, "observing_conditions_mode")
    if cached_mode_raw is None:
        if observing_conditions_mode != "maps":
            raise ValueError(
                "Cached base redshift maps do not record observing_conditions_mode. "
                "Treating older caches as map-varying products; rebuild the classified "
                "chunks and base maps for constant observing conditions."
            )
        cached_mode = "maps"
    else:
        cached_mode = observing_conditions_mode(cached_mode_raw)
        if cached_mode != observing_conditions_mode:
            raise ValueError(
                "Cached base redshift maps were built with "
                f"observing_conditions_mode={cached_mode!r}, but this run requested "
                f"{observing_conditions_mode!r}."
            )
    source_meta = {
        "classified_chunks": str(args.read_chunks_dir),
        "base_redshift_maps": str(args.base_redshift_maps),
        "observing_conditions_mode": cached_mode,
    }
    return base_maps, np.asarray(base_maps["z_edges"], dtype=float), source_meta


def prepare_bias_source(
    args: argparse.Namespace,
    skyvar_selection,
) -> tuple[pd.DataFrame, pd.DataFrame, object, object, np.ndarray, dict[str, object]]:
    source, source_diagnostic_df, source_meta = load_bias_source(args)
    nz_utils = load_nz_utils(args.blendemu_dir, skyvar_selection)
    model = load_model(args.model_json, skyvar_selection, n_jobs=skyvar_selection.N_JOBS)
    z_edges_in = z_edges(skyvar_selection, args)
    return source, source_diagnostic_df, nz_utils, model, z_edges_in, source_meta


def load_selected_from_raw_preds(chunks_dir: str | Path) -> pd.DataFrame | None:
    """Read raw-prediction chunks and return detected sources for diagnostics.

    Applies the detection threshold from ``config.SIM_SETTINGS`` and renames
    ``_input_p`` columns to match the unselected source diagnostic DataFrame so
    that :func:`plotting.plot_bias_property_diagnostics` can overlay both.
    """
    chunks_dir = Path(chunks_dir)
    chunks = sorted(chunks_dir.glob("*.feather"))
    if not chunks:
        return None

    det_threshold = 0.0
    read_cols = ["detection", "r_input_p", "redshift_input_p", "bias_tinker10_input_p"]
    col_map = {
        "r_input_p": "r",
        "redshift_input_p": "redshift",
        "bias_tinker10_input_p": "bias_tinker10",
    }

    pieces: list[pd.DataFrame] = []
    for path in chunks:
        df = pd.read_feather(path, columns=read_cols)
        det = pd.to_numeric(df["detection"], errors="coerce").fillna(0.0)
        sel = df.loc[det > det_threshold]
        if len(sel) > 0:
            pieces.append(sel)

    if not pieces:
        return None

    all_sel = pd.concat(pieces, ignore_index=True)
    diag = {dst: all_sel[src].to_numpy(np.float32) for src, dst in col_map.items() if src in all_sel.columns}
    return pd.DataFrame(diag) if diag else None


def seen_pixels_and_maps(
    args: argparse.Namespace,
    skyvar_selection,
    load_from_chunks: bool,
    base_maps: dict[str, np.ndarray] | None,
    observing_conditions_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if load_from_chunks:
        return np.asarray(base_maps["pixels"], dtype=np.int64), np.array([], dtype=np.int64), None, None, None

    maps_in, seen_stats, seen_sim = skyvar_selection.load_system_maps(return_sim_idx=True)
    max_pixels = config.FULL_WORKFLOW_SETTINGS["max_pixels"]
    if max_pixels is not None:
        n_pixels = int(max_pixels)
        if n_pixels <= 0:
            raise ValueError("SKYVAR_BIAS_MAX_PIXELS must be positive.")
        if n_pixels > len(seen_sim):
            raise ValueError(f"Requested {n_pixels} pixels but only {len(seen_sim)} are available.")
        if args.pixel_selection == "random":
            rng = np.random.default_rng(int(config.SOURCE_SETTINGS["seed"]))
            seen_sim = np.sort(rng.choice(seen_sim, size=n_pixels, replace=False))
        else:
            seen_sim = seen_sim[:n_pixels]
        seen_stats = np.unique(
            skyvar_selection.map_pix_sim_to_stats(
                seen_sim,
                skyvar_selection.SYS_NSIDE_SIM,
                skyvar_selection.SYS_NSIDE_STATS,
            )
        )
    if observing_conditions_mode == "constant":
        obs = skyvar_selection.config.OBS_CONDITIONS
        n_pix_sim = hp.nside2npix(int(skyvar_selection.SYS_NSIDE_SIM))
        psf_map = np.full(n_pix_sim, float(obs["psf_fwhm_nominal"]), dtype=float)
        noise_map = np.full(n_pix_sim, float(obs["pixel_rms_nominal"]), dtype=float)
        galactic_map = np.zeros(n_pix_sim, dtype=float)
    elif observing_conditions_mode == "maps":
        psf_map, noise_map, galactic_map = maps_in
    else:
        raise ValueError(
            f"Unknown observing-condition mode {observing_conditions_mode!r}. "
            "Use 'maps' or 'constant'."
        )
    return seen_stats, seen_sim, psf_map, noise_map, galactic_map


def empty_bias_accumulators(skyvar_selection, z_edges_in: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (skyvar_selection.NPIX, len(z_edges_in) - 1)
    return (
        np.zeros(shape, dtype=np.int64),
        np.zeros(shape, dtype=float),
        np.zeros(shape, dtype=float),
    )


def classify_block(
    block: pd.DataFrame,
    args: argparse.Namespace,
    source_meta: dict[str, object],
    nz_utils,
    model,
    start: int,
) -> pd.DataFrame | None:
    n_degree2 = float(source_meta["source_density_arcmin2"]) * 3600.0
    side_deg = np.sqrt(len(block) / n_degree2)
    rng = np.random.default_rng(int(config.SOURCE_SETTINGS["seed"]) + int(start))
    block["RA"] = rng.uniform(0.0, side_deg, size=len(block))
    block["DEC"] = rng.uniform(0.0, side_deg, size=len(block))

    predictable = ~block["beyond_detec_bound"].astype(bool).to_numpy()
    if not np.any(predictable):
        return None

    classified_full = nz_utils.icat2cla_v2(block.loc[predictable].copy(), block, model, predict=True)
    return compact_classified(classified_full)


def compact_classified(classified: pd.DataFrame) -> pd.DataFrame:
    require_columns(classified, CLASSIFIED_COLUMNS)
    keep = list(CLASSIFIED_COLUMNS)
    compact = classified.loc[:, keep].copy()
    if "pix_idx_input_p" in compact:
        compact["pix_idx_input_p"] = compact["pix_idx_input_p"].astype(np.int64, copy=False)
    return compact


def apply_skyvar_selection(classified: pd.DataFrame, skyvar_selection, mode: str | None) -> pd.DataFrame:
    require_columns(classified, SKYVAR_SELECTION_COLUMNS)
    selected = skyvar_selection.apply_galaxy_selection(
        classified.copy(),
        mode=selection_mode(mode),
    )
    return selected.reset_index(drop=True)


def accumulate_bias_z(
    count: np.ndarray,
    sum_w: np.ndarray,
    sum_wb: np.ndarray,
    pix_sim: np.ndarray,
    redshift: np.ndarray,
    bias: np.ndarray,
    z_edges_in: np.ndarray,
    skyvar_selection,
    weights: np.ndarray | None = None,
) -> tuple[int, float]:
    n_z = len(z_edges_in) - 1
    pix_stats = skyvar_selection.map_pix_sim_to_stats(
        np.asarray(pix_sim, dtype=np.int64),
        skyvar_selection.SYS_NSIDE_SIM,
        skyvar_selection.SYS_NSIDE_STATS,
    )
    z_bin = np.digitize(np.asarray(redshift, dtype=float), z_edges_in) - 1
    bias = np.asarray(bias, dtype=float)
    w = np.ones(len(bias), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    valid = (
        (pix_stats >= 0)
        & (pix_stats < skyvar_selection.NPIX)
        & (z_bin >= 0)
        & (z_bin < n_z)
        & np.isfinite(redshift)
        & np.isfinite(bias)
        & np.isfinite(w)
        & (w > 0)
    )
    if not np.any(valid):
        return 0, 0.0

    flat = pix_stats[valid] * n_z + z_bin[valid]
    minlength = skyvar_selection.NPIX * n_z
    count += np.bincount(flat, minlength=minlength).reshape(count.shape).astype(np.int64)
    sum_w += np.bincount(flat, weights=w[valid], minlength=minlength).reshape(sum_w.shape)
    sum_wb += np.bincount(flat, weights=w[valid] * bias[valid], minlength=minlength).reshape(sum_wb.shape)
    return int(valid.sum()), float(w[valid].sum())


def accumulate_selected_bias(
    selected: pd.DataFrame,
    skyvar_selection,
    z_edges_in: np.ndarray,
    count_s: np.ndarray,
    sum_w_s: np.ndarray,
    sum_wb_s: np.ndarray,
) -> tuple[int, float]:
    if selected.empty:
        return 0, 0.0
    require_columns(selected, ["pix_idx_input_p", "redshift_input_p", "bias_tinker10_input_p", "detection"])
    return accumulate_bias_z(
        count_s,
        sum_w_s,
        sum_wb_s,
        selected["pix_idx_input_p"].to_numpy(np.int64),
        selected["redshift_input_p"].to_numpy(float),
        selected["bias_tinker10_input_p"].to_numpy(float),
        z_edges_in,
        skyvar_selection,
        weights=selected["detection"].to_numpy(float),
    )


def finalize_base_maps(
    seen_stats: np.ndarray,
    z_edges_in: np.ndarray,
    count_u: np.ndarray,
    sum_w_u: np.ndarray,
    sum_wb_u: np.ndarray,
    count_s: np.ndarray,
    sum_w_s: np.ndarray,
    sum_wb_s: np.ndarray,
) -> dict[str, np.ndarray]:
    mean_u = np.divide(sum_wb_u, sum_w_u, out=np.full_like(sum_w_u, np.nan), where=sum_w_u > 0)
    mean_s = np.divide(sum_wb_s, sum_w_s, out=np.full_like(sum_w_s, np.nan), where=sum_w_s > 0)
    pixels = np.asarray(seen_stats, dtype=np.int64)
    return {
        "pixels": pixels,
        "z_edges": np.asarray(z_edges_in, dtype=float),
        "z_centers": 0.5 * (z_edges_in[:-1] + z_edges_in[1:]),
        "mean_bias_unselected_z": mean_u[pixels],
        "mean_bias_selected_z": mean_s[pixels],
        "delta_bias_z": mean_s[pixels] - mean_u[pixels],
        "count_unselected_z": count_u[pixels],
        "count_selected_z": count_s[pixels],
        "sum_weight_unselected_z": sum_w_u[pixels],
        "sum_weight_selected_z": sum_w_s[pixels],
    }


def selected_maps_from_cache(
    base_maps: dict[str, np.ndarray],
    count_s: np.ndarray,
    sum_w_s: np.ndarray,
    sum_wb_s: np.ndarray,
) -> dict[str, np.ndarray]:
    pixels = np.asarray(base_maps["pixels"], dtype=np.int64)
    selected = np.divide(sum_wb_s, sum_w_s, out=np.full_like(sum_w_s, np.nan), where=sum_w_s > 0)
    out = dict(base_maps)
    out["mean_bias_selected_z"] = selected[pixels]
    out["count_selected_z"] = count_s[pixels]
    out["sum_weight_selected_z"] = sum_w_s[pixels]
    out["delta_bias_z"] = out["mean_bias_selected_z"] - out["mean_bias_unselected_z"]
    return out


def attach_tomographic_bins(selected: pd.DataFrame, skyvar_selection) -> pd.DataFrame:
    if selected.empty:
        return selected
    tomo_edges = np.asarray(skyvar_selection.config.ANALYSIS_SETTINGS["tomo_bin_edges"], dtype=float)
    tomo_cols = [f"tomo_p_{i}" for i in range(len(tomo_edges) - 1)]
    if set(tomo_cols).issubset(selected.columns):
        return selected

    processed = skyvar_selection.process_classified_catalog(selected.copy())
    for col in tomo_cols:
        if col in processed.columns:
            selected[col] = processed[col].to_numpy(np.float32, copy=False)
    if "z_pho_input_p" in processed.columns and "z_pho_input_p" not in selected.columns:
        selected["z_pho_input_p"] = processed["z_pho_input_p"].to_numpy(np.float32, copy=False)
    return selected


def empty_tomographic_accumulators(
    skyvar_selection,
    z_edges_in: np.ndarray,
    n_tomo: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (int(n_tomo), skyvar_selection.NPIX, len(z_edges_in) - 1)
    return (
        np.zeros(shape, dtype=np.int64),
        np.zeros(shape, dtype=float),
        np.zeros(shape, dtype=float),
    )


def accumulate_selected_and_tomo_bias(
    selected: pd.DataFrame,
    skyvar_selection,
    z_edges_in: np.ndarray,
    tomo_cols: list[str],
    count_s: np.ndarray,
    sum_w_s: np.ndarray,
    sum_wb_s: np.ndarray,
    count_t: np.ndarray,
    sum_w_t: np.ndarray,
    sum_wb_t: np.ndarray,
    tomo_rows_in_z: np.ndarray,
    tomo_weight_in_z: np.ndarray,
) -> tuple[int, float]:
    if selected.empty:
        return 0, 0.0

    selected = attach_tomographic_bins(selected, skyvar_selection)
    rows, weight = accumulate_selected_bias(selected, skyvar_selection, z_edges_in, count_s, sum_w_s, sum_wb_s)

    base_weights = selected["detection"].to_numpy(float)
    for i, col in enumerate(tomo_cols):
        if col not in selected.columns:
            continue
        t_rows, t_weight = accumulate_bias_z(
            count_t[i],
            sum_w_t[i],
            sum_wb_t[i],
            selected["pix_idx_input_p"].to_numpy(np.int64),
            selected["redshift_input_p"].to_numpy(float),
            selected["bias_tinker10_input_p"].to_numpy(float),
            z_edges_in,
            skyvar_selection,
            weights=base_weights * selected[col].to_numpy(float),
        )
        tomo_rows_in_z[i] += t_rows
        tomo_weight_in_z[i] += t_weight
    return rows, weight


def finalize_tomo_maps(
    base_maps: dict[str, np.ndarray],
    tomo_bin_edges: np.ndarray,
    count_t: np.ndarray,
    sum_w_t: np.ndarray,
    sum_wb_t: np.ndarray,
) -> dict[str, np.ndarray]:
    pixels = np.asarray(base_maps["pixels"], dtype=np.int64)
    mean_t = np.divide(sum_wb_t, sum_w_t, out=np.full_like(sum_w_t, np.nan), where=sum_w_t > 0)
    unselected = np.asarray(base_maps["mean_bias_unselected_z"], dtype=float)
    selected = np.asarray(base_maps["mean_bias_selected_z"], dtype=float)
    return {
        **base_maps,
        "tomo_bin_edges": np.asarray(tomo_bin_edges, dtype=float),
        "mean_bias_tomo_z": mean_t[:, pixels, :],
        "delta_bias_tomo_minus_unselected_z": mean_t[:, pixels, :] - unselected[None, :, :],
        "delta_bias_tomo_minus_selected_z": mean_t[:, pixels, :] - selected[None, :, :],
        "count_tomo_z": count_t[:, pixels, :],
        "sum_weight_tomo_z": sum_w_t[:, pixels, :],
    }


def summarize_redshift_binned_maps(
    maps: dict[str, np.ndarray],
    nside: int,
    min_count: int = 1,
) -> dict[str, float | int | str]:
    mean_u = maps["mean_bias_unselected_z"]
    mean_s = maps["mean_bias_selected_z"]
    delta = maps["delta_bias_z"]
    finite_delta = delta[np.isfinite(delta)]
    active_unselected = np.any(np.isfinite(mean_u), axis=1) if len(mean_u) else np.array([], dtype=bool)
    active_selected = np.any(np.isfinite(mean_s), axis=1) if len(mean_s) else np.array([], dtype=bool)
    active_union = active_unselected | active_selected
    return {
        "nside": int(nside),
        "n_pix": int(hp.nside2npix(int(nside))),
        "z_bins": int(len(maps["z_centers"])),
        "min_count": int(min_count),
        "output_pixels": int(len(maps["pixels"])),
        "active_pixels_union": int(active_union.sum()),
        "active_pixels_unselected": int(active_unselected.sum()),
        "active_pixels_selected": int(active_selected.sum()),
        "filled_pixel_zbins_unselected": int(np.isfinite(mean_u).sum()),
        "filled_pixel_zbins_selected": int(np.isfinite(mean_s).sum()),
        "max_abs_delta_bias_z": float(np.max(np.abs(finite_delta))) if finite_delta.size else 0.0,
        "median_unselected_bias_z": float(np.nanmedian(mean_u)) if np.isfinite(mean_u).any() else float("nan"),
        "median_selected_bias_z": float(np.nanmedian(mean_s)) if np.isfinite(mean_s).any() else float("nan"),
    }


def weighted_profile(mean: np.ndarray, weight: np.ndarray) -> np.ndarray:
    numerator = np.nansum(mean * weight, axis=0)
    denominator = np.sum(np.where(np.isfinite(mean), weight, 0.0), axis=0)
    return np.divide(numerator, denominator, out=np.full(mean.shape[1], np.nan), where=denominator > 0)


def summarize_tomographic_maps(maps: dict[str, np.ndarray], min_count: int = 1) -> dict[str, object]:
    mean_t = np.asarray(maps["mean_bias_tomo_z"], dtype=float)
    count_t = np.asarray(maps["count_tomo_z"], dtype=np.int64)
    weight_t = np.asarray(maps["sum_weight_tomo_z"], dtype=float)

    tomo_summary = []
    for i in range(mean_t.shape[0]):
        finite = np.isfinite(mean_t[i])
        active_pixels = np.any(finite, axis=1) if finite.size else np.array([], dtype=bool)
        valid_count = count_t[i] >= int(min_count)
        global_profile = weighted_profile(mean_t[i], weight_t[i])
        tomo_summary.append(
            {
                "tomo_index": int(i),
                "z_pho_min": float(maps["tomo_bin_edges"][i]),
                "z_pho_max": float(maps["tomo_bin_edges"][i + 1]),
                "active_pixels": int(active_pixels.sum()),
                "filled_pixel_zbins": int(finite.sum()),
                "filled_pixel_zbins_min_count": int((finite & valid_count).sum()),
                "rows_in_z_range": int(count_t[i].sum()),
                "selected_weight_in_z_range": float(weight_t[i].sum()),
                "median_bias_z": float(np.nanmedian(mean_t[i])) if finite.any() else float("nan"),
                "global_profile_mean": float(np.nanmean(global_profile))
                if np.isfinite(global_profile).any()
                else float("nan"),
                "global_profile_max": float(np.nanmax(global_profile))
                if np.isfinite(global_profile).any()
                else float("nan"),
            }
        )
    return {
        "n_tomo_bins": int(mean_t.shape[0]),
        "tomo_bin_edges": [float(x) for x in np.asarray(maps["tomo_bin_edges"], dtype=float)],
        "tomo_bins": tomo_summary,
    }


def normalize_chunk_args(args: argparse.Namespace, include_tomo: bool) -> None:
    if include_tomo:
        args.read_chunks_dir = args.classified_chunks_dir
        args.write_chunks_dir = args.write_classified_chunks_dir
        args.overwrite_chunks = bool(args.overwrite_classified_chunks)
    else:
        args.read_chunks_dir = args.raw_preds_dir
        args.write_chunks_dir = args.write_raw_preds_dir
        args.overwrite_chunks = bool(args.overwrite_raw_preds)

    if args.read_chunks_dir is not None and args.base_redshift_maps is None:
        raise ValueError("Reading cached classified chunks requires --base-redshift-maps.")
    if args.read_chunks_dir is not None and args.write_chunks_dir is not None:
        raise ValueError("Read and write classified chunk directories are mutually exclusive.")


def run_bias_workflow(args: argparse.Namespace, include_tomo: bool) -> dict[str, object]:
    skyvar_selection = load_skyvar_selection_module()
    normalize_chunk_args(args, include_tomo=include_tomo)
    load_from_chunks = args.read_chunks_dir is not None
    observing_mode = resolve_observing_conditions_mode(args)
    observing_meta = observing_conditions_metadata(observing_mode, skyvar_selection)

    workflow_cfg = config.FULL_WORKFLOW_SETTINGS
    max_pixels = workflow_cfg["max_pixels"]
    pixel_bz_count = int(workflow_cfg["pixel_bz_count"])
    pixel_bz_z_bins = int(workflow_cfg["pixel_bz_z_bins"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir = Path(args.data_output_dir or args.output_dir)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    base_maps_from_cache: dict[str, np.ndarray] | None = None
    source_plot_paths: list[Path] = []
    if load_from_chunks:
        base_maps_from_cache, z_edges_in, source_meta = load_cached_base_maps(args, observing_mode)
        source = nz_utils = model = None
        density_arcmin2 = None
        n_pop = int(skyvar_selection.N_POP_SAMPLE)
        sim_pixel_area_arcmin2 = healpix_pixel_area_arcmin2(skyvar_selection.SYS_NSIDE_SIM)
    else:
        source_density_meta = source_density_meta_from_table(args.source_table)
        density_override = getattr(args, "source_density_arcmin2", None)
        if density_override is not None:
            source_density_meta = {
                "source_density_arcmin2": float(density_override),
                "source_density_source": "cli_or_config_override",
                "source_density_summary": None,
                "source_density_diagnostic": None,
            }
        density_arcmin2 = source_density_for_n_pop(source_density_meta)
        n_pop, sim_pixel_area_arcmin2 = n_pop_from_source_density(
            density_arcmin2,
            skyvar_selection.SYS_NSIDE_SIM,
        )
        print(
            "n_pop from source density: "
            f"{density_arcmin2:.6g} arcmin^-2 * {sim_pixel_area_arcmin2:.6g} arcmin^2 "
            f"= {n_pop}",
            flush=True,
        )
        print(
            "source density source: "
            f"{source_density_meta.get('source_density_source', 'unknown') if source_density_meta else 'unknown'}",
            flush=True,
        )

    if observing_mode == "constant":
        print(
            "observing conditions: constant baseline "
            f"(psf_fwhm={observing_meta['baseline_psf_fwhm']:.6g}, "
            f"pixel_rms={observing_meta['baseline_pixel_rms']:.6g}, "
            "galactic extinction=0)",
            flush=True,
        )
    else:
        print("observing conditions: systematics maps", flush=True)

    observing_tag = "" if observing_mode == "maps" else "_constantobs"
    pop_seg = f"_pop{n_pop}" if density_arcmin2 is not None else ""
    default_tag = (
        f"full_skyvar_tomo_bias_sim{skyvar_selection.SYS_NSIDE_SIM}"
        f"_stats{skyvar_selection.SYS_NSIDE_STATS}{pop_seg}{observing_tag}"
        if include_tomo
        else f"full_skyvar_bias_sim{skyvar_selection.SYS_NSIDE_SIM}"
        f"_stats{skyvar_selection.SYS_NSIDE_STATS}{pop_seg}{observing_tag}"
    )
    tag = args.tag or default_tag

    source_diagnostic_df: pd.DataFrame | None = None
    if not load_from_chunks:
        source, source_diagnostic_df, nz_utils, model, z_edges_in, source_meta = prepare_bias_source(
            args,
            skyvar_selection,
        )

    seen_stats, seen_sim, psf_map, noise_map, galactic_map = seen_pixels_and_maps(
        args,
        skyvar_selection,
        load_from_chunks,
        base_maps_from_cache,
        observing_mode,
    )

    count_u, sum_w_u, sum_wb_u = empty_bias_accumulators(skyvar_selection, z_edges_in)
    count_s, sum_w_s, sum_wb_s = empty_bias_accumulators(skyvar_selection, z_edges_in)

    tomo_bin_edges = np.asarray(skyvar_selection.config.ANALYSIS_SETTINGS["tomo_bin_edges"], dtype=float)
    n_tomo = len(tomo_bin_edges) - 1
    tomo_cols = [f"tomo_p_{i}" for i in range(n_tomo)]
    if include_tomo:
        count_t, sum_w_t, sum_wb_t = empty_tomographic_accumulators(skyvar_selection, z_edges_in, n_tomo)
        tomo_rows_in_z = np.zeros(n_tomo, dtype=np.int64)
        tomo_weight_in_z = np.zeros(n_tomo, dtype=float)
    else:
        count_t = sum_w_t = sum_wb_t = None
        tomo_rows_in_z = np.zeros(0, dtype=np.int64)
        tomo_weight_in_z = np.zeros(0, dtype=float)

    obs = skyvar_selection.config.OBS_CONDITIONS
    conditions = {
        "pixel_size": float(obs["pixel_size"]),
        "zero_point": float(obs["zero_point"]),
        "psf_fwhm": float(obs["psf_fwhm_nominal"]),
        "moffat_beta": float(obs["moffat_beta"]),
        "pixel_rms": float(obs["pixel_rms_nominal"]),
    }

    pixels_per_chunk = max(1, int(skyvar_selection.CHUNK_SIZE) // n_pop)
    n_chunks = int(np.ceil(len(seen_sim) / pixels_per_chunk)) if len(seen_sim) else 0
    classified_rows = 0
    selected_rows = 0
    unselected_rows_in_z = 0
    selected_rows_in_z = 0
    selected_weight_in_z = 0.0
    written_chunks: list[str] = []
    start_time = time.perf_counter()

    if load_from_chunks:
        chunk_paths = find_chunks(args.read_chunks_dir)
        for chunk_index, chunk_path in enumerate(chunk_paths, start=1):
            print(f"classified chunk {chunk_index}/{len(chunk_paths)}: {chunk_path.name}", flush=True)
            classified = pd.read_feather(chunk_path)
            classified_rows += len(classified)
            selected = apply_skyvar_selection(classified, skyvar_selection, mode=args.selection_mode)
            selected_rows += len(selected)
            if include_tomo:
                rows, weight = accumulate_selected_and_tomo_bias(
                    selected,
                    skyvar_selection,
                    z_edges_in,
                    tomo_cols,
                    count_s,
                    sum_w_s,
                    sum_wb_s,
                    count_t,
                    sum_w_t,
                    sum_wb_t,
                    tomo_rows_in_z,
                    tomo_weight_in_z,
                )
            else:
                rows, weight = accumulate_selected_bias(selected, skyvar_selection, z_edges_in, count_s, sum_w_s, sum_wb_s)
            selected_rows_in_z += rows
            selected_weight_in_z += weight
    else:
        for start in range(0, len(seen_sim), pixels_per_chunk):
            stop = min(start + pixels_per_chunk, len(seen_sim))
            block_pixels = seen_sim[start:stop]
            print(f"chunk {start // pixels_per_chunk + 1}/{n_chunks}: {len(block_pixels)} sim pixels", flush=True)

            pieces = Parallel(n_jobs=skyvar_selection.N_JOBS, backend="threading")(
                delayed(skyvar_selection.process_one)(
                    start + i,
                    int(pixel),
                    source,
                    conditions,
                    n_pop,
                    psf_map,
                    noise_map,
                    galactic_map,
                    obs["detec_mag_bound"],
                    skyvar_selection.SYS_NSIDE_SIM,
                )
                for i, pixel in enumerate(block_pixels)
            )
            block = pd.concat(pieces, ignore_index=True)

            rows, _ = accumulate_bias_z(
                count_u,
                sum_w_u,
                sum_wb_u,
                block["pix_idx"].to_numpy(np.int64),
                block["redshift"].to_numpy(float),
                block["bias_tinker10"].to_numpy(float),
                z_edges_in,
                skyvar_selection,
            )
            unselected_rows_in_z += rows

            classified = classify_block(block, args, source_meta, nz_utils, model, start)
            if classified is not None:
                if args.write_chunks_dir:
                    path = save_chunk(classified, args.write_chunks_dir, start=start, overwrite=args.overwrite_chunks)
                    written_chunks.append(str(path))
                classified_rows += len(classified)
                selected = apply_skyvar_selection(classified, skyvar_selection, mode=args.selection_mode)
                selected_rows += len(selected)
                if include_tomo:
                    rows, weight = accumulate_selected_and_tomo_bias(
                        selected,
                        skyvar_selection,
                        z_edges_in,
                        tomo_cols,
                        count_s,
                        sum_w_s,
                        sum_wb_s,
                        count_t,
                        sum_w_t,
                        sum_wb_t,
                        tomo_rows_in_z,
                        tomo_weight_in_z,
                    )
                else:
                    rows, weight = accumulate_selected_bias(
                        selected,
                        skyvar_selection,
                        z_edges_in,
                        count_s,
                        sum_w_s,
                        sum_wb_s,
                    )
                selected_rows_in_z += rows
                selected_weight_in_z += weight
            del block

    if load_from_chunks:
        base_maps = selected_maps_from_cache(base_maps_from_cache, count_s, sum_w_s, sum_wb_s)
        unselected_rows_in_z = int(np.asarray(base_maps.get("count_unselected_z", np.array([]))).sum())
    else:
        base_maps = finalize_base_maps(seen_stats, z_edges_in, count_u, sum_w_u, sum_wb_u, count_s, sum_w_s, sum_wb_s)

    base_maps["nside"] = np.array(int(skyvar_selection.SYS_NSIDE_STATS), dtype=np.int64)
    base_maps["observing_conditions_mode"] = np.array(observing_mode)
    base_maps["galactic_extinction_mode"] = np.array(observing_meta["galactic_extinction_mode"])
    base_maps["baseline_psf_fwhm"] = np.array(observing_meta["baseline_psf_fwhm"], dtype=float)
    base_maps["baseline_pixel_rms"] = np.array(observing_meta["baseline_pixel_rms"], dtype=float)
    full_map_path = data_output_dir / f"{tag}_redshift_binned_bias_maps.npz"

    plot_paths: list[Path] = []
    map_plot_paths: list[Path] = []
    pixel_bz_paths: list[Path] = []
    tomo_plot_paths: list[Path] = []

    if include_tomo:
        out_maps = finalize_tomo_maps(base_maps, tomo_bin_edges, count_t, sum_w_t, sum_wb_t)
        out_maps["nside"] = np.array(int(skyvar_selection.SYS_NSIDE_STATS), dtype=np.int64)
        map_path = data_output_dir / f"{tag}_tomo_binned_bias_maps.npz"
        np.savez_compressed(map_path, **out_maps)
        if not args.skip_full_redshift_products:
            np.savez_compressed(full_map_path, **base_maps)
            plot_paths = plotting.plot_redshift_binned_bias_summary(base_maps, output_dir=output_dir, tag=tag)
            map_plot_paths = plotting.plot_redshift_binned_bias_maps(
                base_maps,
                nside=skyvar_selection.SYS_NSIDE_STATS,
                output_dir=output_dir,
                tag=tag,
            )
            pixel_bz_paths = plotting.plot_pixel_bz_examples(
                base_maps,
                output_dir=output_dir,
                tag=tag,
                n_examples=pixel_bz_count,
                max_z_bins=pixel_bz_z_bins if pixel_bz_z_bins > 0 else None,
            )
        tomo_plot_paths = plotting.plot_tomographic_bias_summary(out_maps, output_dir=output_dir, tag=tag)
    else:
        out_maps = base_maps
        map_path = full_map_path
        np.savez_compressed(map_path, **out_maps)
        plot_paths = plotting.plot_redshift_binned_bias_summary(out_maps, output_dir=output_dir, tag=tag)
        map_plot_paths = plotting.plot_redshift_binned_bias_maps(
            out_maps,
            nside=skyvar_selection.SYS_NSIDE_STATS,
            output_dir=output_dir,
            tag=tag,
        )
        pixel_bz_paths = plotting.plot_pixel_bz_examples(
            out_maps,
            output_dir=output_dir,
            tag=tag,
            n_examples=pixel_bz_count,
            max_z_bins=pixel_bz_z_bins if pixel_bz_z_bins > 0 else None,
        )

    if source_diagnostic_df is not None and not getattr(args, "skip_source_diagnostics", False):
        df_selected = None
        if args.write_chunks_dir:
            print("Loading selected sample from raw prediction chunks ...", flush=True)
            df_selected = load_selected_from_raw_preds(args.write_chunks_dir)
            if df_selected is not None:
                print(f"  {len(df_selected):,} detected sources for diagnostics", flush=True)
        source_plot_paths = plotting.plot_bias_property_diagnostics(
            source_diagnostic_df,
            output_dir=output_dir,
            tag=f"{tag}_source",
            max_points=config.SOURCE_SETTINGS["diagnostic_points"],
            seed=config.SOURCE_SETTINGS["seed"],
            redshift_range=(float(z_edges_in[0]), float(z_edges_in[-1])),
            df_selected=df_selected,
        )

    summary = summarize_redshift_binned_maps(base_maps, nside=skyvar_selection.SYS_NSIDE_STATS, min_count=1)
    if include_tomo:
        summary.update(summarize_tomographic_maps(out_maps, min_count=1))

    summary.update(
        {
            "workflow": "skyvar_selection_bias_tomographic" if include_tomo else "skyvar_selection_bias",
            "selection_source": "src.selection.apply_galaxy_selection",
            "tomographic_binning": "src.selection.process_classified_catalog" if include_tomo else None,
            **observing_meta,
            "sys_nside_sim": int(skyvar_selection.SYS_NSIDE_SIM),
            "sys_nside_stats": int(skyvar_selection.SYS_NSIDE_STATS),
            "n_pop_sample": int(n_pop),
            "n_pop_source": source_meta.get("source_density_source", "source_density_arcmin2_x_sim_pixel_area"),
            "source_density_for_n_pop_arcmin2": float(density_arcmin2) if density_arcmin2 is not None else None,
            "sim_pixel_area_arcmin2": float(sim_pixel_area_arcmin2),
            "configured_n_pop_sample": int(skyvar_selection.N_POP_SAMPLE),
            "chunk_size": int(skyvar_selection.CHUNK_SIZE),
            "pixels_per_chunk": int(pixels_per_chunk),
            "seen_sim_pixels_processed": int(len(seen_sim)),
            "seen_stats_pixels": int(len(seen_stats)),
            "pixel_selection": args.pixel_selection if max_pixels is not None else "all",
            "max_pixels": None if max_pixels is None else int(max_pixels),
            "total_input_draws": int((len(seen_sim) if len(seen_sim) else len(seen_stats)) * n_pop),
            "classified_rows": int(classified_rows),
            "selected_rows": int(selected_rows),
            "unselected_rows_in_z_range": int(unselected_rows_in_z),
            "selected_rows_in_z_range": int(selected_rows_in_z),
            "selected_weight_in_z_range": float(selected_weight_in_z),
            "tomo_rows_in_z_range": [int(x) for x in tomo_rows_in_z],
            "tomo_weight_in_z_range": [float(x) for x in tomo_weight_in_z],
            "selection_mode": args.selection_mode,
            "z_min": float(z_edges_in[0]),
            "z_max": float(z_edges_in[-1]),
            "z_step": float(np.nanmedian(np.diff(z_edges_in))) if len(z_edges_in) > 1 else None,
            "loaded_classified_chunks_dir": str(args.read_chunks_dir) if args.read_chunks_dir else None,
            "base_redshift_maps": str(args.base_redshift_maps) if args.base_redshift_maps else None,
            "written_classified_chunks_dir": str(args.write_chunks_dir) if args.write_chunks_dir else None,
            "written_classified_chunks": written_chunks,
            "elapsed_seconds": float(time.perf_counter() - start_time),
            "source": source_meta,
            "output_dir": str(output_dir),
            "data_output_dir": str(data_output_dir),
            "map_path": str(map_path),
            "full_map_path": str(full_map_path) if include_tomo and not args.skip_full_redshift_products else None,
            "source_plots": [str(path) for path in source_plot_paths],
            "plots": [str(path) for path in plot_paths],
            "map_plots": [str(path) for path in map_plot_paths],
            "pixel_bz_plots": [str(path) for path in pixel_bz_paths],
            "pixel_bz_count": pixel_bz_count,
            "pixel_bz_z_bins": pixel_bz_z_bins if pixel_bz_z_bins > 0 else None,
            "tomo_plots": [str(path) for path in tomo_plot_paths],
        }
    )
    suffix = "tomo_binned_bias_maps_summary.json" if include_tomo else "redshift_binned_bias_maps_summary.json"
    summary_path = output_dir / f"{tag}_{suffix}"
    summary["summary_path"] = str(summary_path)
    utils.write_json(summary_path, summary)
    return summary


def run_full_workflow(args: argparse.Namespace) -> dict[str, object]:
    return run_bias_workflow(args, include_tomo=False)


def run_tomo_workflow(args: argparse.Namespace) -> dict[str, object]:
    return run_bias_workflow(args, include_tomo=True)


def build_common_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source-table",
        default=os.environ.get("SKYVAR_BIAS_SOURCE_TABLE")
        or config.SOURCE_SETTINGS.get("bias_source_table")
        or None,
        help=(
            "Bias-assigned Feather source table from `python -m bias_variation.halo_bias assign-source`. "
            "Defaults to SOURCE_SETTINGS['bias_source_table']."
        ),
    )
    parser.add_argument("--blendemu-dir", default=config.PATHS["blendemu_dir"] or None)
    parser.add_argument("--model-json", default=config.PATHS["model_json"] or None)
    parser.add_argument("--selection-mode", choices=("none", "snr", "maglim"), default="none")
    parser.add_argument(
        "--observing-conditions",
        "--observing-conditions-mode",
        dest="observing_conditions_mode",
        choices=("maps", "constant"),
        default="maps",
        help=(
            "Use map-varying observing conditions, or constant baseline PSF/noise "
            "with zero Galactic extinction."
        ),
    )
    parser.add_argument("--pixel-selection", choices=("first", "random"), default="first")
    parser.add_argument("--output-dir", default=config.PATHS["output_dir"])
    parser.add_argument(
        "--data-output-dir",
        default=None,
        help="Directory for non-plot data products such as map NPZ files; defaults to --output-dir.",
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--source-density-arcmin2",
        type=float,
        default=config.SOURCE_SETTINGS.get("target_density_arcmin2"),
        help=(
            "Override the source density (arcmin^-2) from the bias-source summary JSON. "
            "Use this when the source catalog has a different density than the target survey "
            "(e.g. SPS catalog used with f24 target density). Defaults to "
            "SOURCE_SETTINGS['target_density_arcmin2']; set that to None to use the source-table summary."
        ),
    )
    parser.add_argument("--skip-source-diagnostics", action="store_true")
    parser.add_argument(
        "--base-redshift-maps",
        default=None,
        help="Existing *_redshift_binned_bias_maps.npz used when reselecting cached classified chunks.",
    )
    return parser


def build_full_workflow_parser() -> argparse.ArgumentParser:
    parser = build_common_parser(description="Run SkyVar selection and accumulate full-sample b(z, pixel).")
    parser.add_argument(
        "--raw-preds-dir",
        default=None,
        help="Directory of cached pre-selection classified Feather chunks.",
    )
    parser.add_argument(
        "--write-raw-preds-dir",
        default=None,
        help="Optional directory where pre-selection classified Feather chunks are cached.",
    )
    parser.add_argument("--overwrite-raw-preds", action="store_true")
    return parser


def build_tomo_workflow_parser() -> argparse.ArgumentParser:
    parser = build_common_parser(description="Run SkyVar selection and accumulate tomographic b(z, pixel).")
    parser.add_argument(
        "--skip-full-redshift-products",
        action="store_true",
        help="Only write tomographic products, skipping the full selected/unselected plots.",
    )
    parser.add_argument(
        "--classified-chunks-dir",
        default=None,
        help="Directory of cached pre-selection classified Feather chunks.",
    )
    parser.add_argument(
        "--write-classified-chunks-dir",
        default=None,
        help="Optional directory where pre-selection classified Feather chunks are cached.",
    )
    parser.add_argument("--overwrite-classified-chunks", action="store_true")
    return parser


_COMMANDS = {
    "full": (build_full_workflow_parser, run_full_workflow),
    "tomo": (build_tomo_workflow_parser, run_tomo_workflow),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=sorted(_COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        commands = "\n".join(f"  {name}" for name in sorted(_COMMANDS))
        print(f"{__doc__}\n\nCommands:\n{commands}\n")
        return
    command = argv[0]
    if command not in _COMMANDS:
        raise SystemExit(f"Unknown command {command!r}. Use one of: {', '.join(sorted(_COMMANDS))}")
    build, run = _COMMANDS[command]
    args = build().parse_args(argv[1:])
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
