"""Minimal GalSBI catalog generation and quality-cut helpers."""

from __future__ import annotations

import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd

from . import config, utils


@dataclass(frozen=True)
class CatalogInfo:
    path: Path
    dataset: str
    n_rows: int
    columns: tuple[str, ...]


def load_hdf5_column_table(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a one-dataset-per-column HDF5 table as a structured array."""

    path = Path(path).expanduser()
    with h5py.File(path, "r") as handle:
        if "data" in handle and getattr(handle["data"].dtype, "names", None):
            return handle["data"][:]

        columns = [
            name
            for name, obj in handle.items()
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 1
        ]
        if not columns:
            raise ValueError(f"No loadable HDF5 datasets found in {path}.")

        row_counts = {int(handle[name].shape[0]) for name in columns}
        if len(row_counts) != 1:
            raise ValueError(f"Columns in {path} have inconsistent row counts: {sorted(row_counts)}")
        n_rows = row_counts.pop()

        dtype = []
        for name in columns:
            data = handle[name]
            tail_shape = data.shape[1:]
            dtype.append((name, data.dtype, tail_shape) if tail_shape else (name, data.dtype))

        out = np.empty(n_rows, dtype=dtype)
        for name in columns:
            out[name] = handle[name][:]
    return out


@contextmanager
def use_custom_galsbi_posterior(model_name: str, posterior_path: str | os.PathLike[str] | None):
    """Temporarily make GalSBI load a custom posterior table for one model."""

    if posterior_path in (None, ""):
        yield None
        return

    from galsbi import load

    posterior = load_hdf5_column_table(posterior_path)
    original_loader = load.load_abc_posterior

    def patched_loader(name):
        if name == model_name:
            return posterior
        return original_loader(name)

    load.load_abc_posterior = patched_loader
    try:
        yield posterior
    finally:
        load.load_abc_posterior = original_loader




def inspect_hdf5_catalog(
    path: str | os.PathLike[str],
    dataset: str | None = None,
) -> CatalogInfo:
    """Inspect a GalSBI HDF5 compound dataset without loading rows."""

    dataset = dataset or config.SOURCE_SETTINGS["dataset"]
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"Dataset {dataset!r} not found in {path}.")
        data = handle[dataset]
        if data.dtype.names is None:
            raise TypeError(f"Dataset {dataset!r} is not a compound table.")
        return CatalogInfo(
            path=path,
            dataset=dataset,
            n_rows=int(data.shape[0]),
            columns=tuple(data.dtype.names),
        )


def _read_hdf5_rows(
    path: Path,
    dataset: str,
    row_indices: np.ndarray | slice,
    columns: Sequence[str] | None,
) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        data = handle[dataset]
        if columns is None:
            arr = data[row_indices]
        else:
            missing = sorted(set(columns) - set(data.dtype.names or ()))
            if missing:
                raise KeyError(f"Missing columns in {path}: {missing}")
            arr = data.fields(list(columns))[row_indices]
    return pd.DataFrame.from_records(arr)


def load_galsbi_subset(
    path: str | os.PathLike[str],
    dataset: str | None = None,
    n_rows: int | None = None,
    seed: int | None = None,
    random: bool = False,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a GalSBI catalog subset as a DataFrame."""

    dataset = dataset or config.SOURCE_SETTINGS["dataset"]
    n_rows = config.SOURCE_SETTINGS["source_rows"] if n_rows is None else int(n_rows)
    cat_path = Path(path)
    info = inspect_hdf5_catalog(cat_path, dataset=dataset)

    if n_rows <= 0 or n_rows >= info.n_rows:
        row_indices: np.ndarray | slice = slice(None)
    elif random:
        seed = config.SOURCE_SETTINGS["seed"] if seed is None else int(seed)
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(info.n_rows, size=n_rows, replace=False))
    else:
        row_indices = slice(0, n_rows)
    return _read_hdf5_rows(cat_path, dataset, row_indices, columns)


def e1e2_to_q_phi(e1: Iterable[float], e2: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert distortion ellipticity to axis ratio and position angle."""

    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    ell = np.hypot(e1, e2)
    ell = np.clip(ell, 0.0, 0.999999999)
    q = np.sqrt((1.0 - ell) / (1.0 + ell))
    phi = 0.5 * np.arctan2(e2, e1)
    return q, phi


def standardize_galsbi_columns(
    df: pd.DataFrame,
    size_column: str | None = None,
    size_scale: float | None = None,
) -> pd.DataFrame:
    """Rename GalSBI columns to the conventions used by the SkyVar source cuts."""

    size_column = config.SOURCE_SETTINGS["size_column"] if size_column is None else size_column
    size_scale = config.SOURCE_SETTINGS["size_scale"] if size_scale is None else float(size_scale)

    out = df.copy()
    out = out.rename(columns={"z": "redshift", "int_mag": "r"})
    if size_column is None:
        if "int_r50_arcsec" in out.columns:
            out["Re"] = out["int_r50_arcsec"]
    else:
        if size_column not in out.columns:
            raise KeyError(f"Requested size column {size_column!r} not found.")
        out["Re"] = out[size_column].to_numpy(dtype=float) * float(size_scale)

    if "BA" not in out.columns and {"e1", "e2"}.issubset(out.columns):
        ba, angle = e1e2_to_q_phi(out["e1"], out["e2"])
        out["BA"] = ba
        out["angle"] = angle / np.pi * 180.0 + 90.0
    if {"Re", "BA"}.issubset(out.columns):
        out["Re_semimajor"] = np.divide(
            out["Re"].to_numpy(dtype=float),
            np.sqrt(np.clip(out["BA"].to_numpy(dtype=float), 1e-8, None)),
        )
    return out


def _column(rows, name: str) -> np.ndarray:
    if isinstance(rows, pd.DataFrame):
        return rows[name].to_numpy()
    return rows[name]


def quality_cut_mask(
    rows,
    size_column: str | None = None,
    size_scale: float | None = None,
) -> np.ndarray:
    """Return the configured SkyVar-style source quality mask."""

    quality = config.QUALITY_CUT_SETTINGS
    size_column = size_column or config.SPS_GENERATION_SETTINGS["quality_size_column"]
    size_scale = (
        config.SPS_GENERATION_SETTINGS["quality_size_scale"]
        if size_scale is None
        else float(size_scale)
    )

    ell = np.hypot(_column(rows, "e1").astype(float), _column(rows, "e2").astype(float))
    ell = np.clip(ell, 0.0, 0.999999999)
    ba = np.sqrt((1.0 - ell) / (1.0 + ell))
    re = _column(rows, size_column).astype(float) * float(size_scale)
    re_semimajor = re / np.sqrt(np.clip(ba, 1e-8, None))
    mag = _column(rows, "int_mag").astype(float)
    sersic = _column(rows, "sersic_n").astype(float)
    return (
        (ba > quality["ba_min"])
        & (ba < quality["ba_max"])
        & (mag < quality["mag_max"])
        & (mag > quality["mag_min"])
        & (re_semimajor < quality["re_max"])
        & (re_semimajor > quality["re_min"])
        & (sersic < quality["sersic_max"])
        & (sersic > quality["sersic_min"])
    )


def apply_selection_quality_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply configured morphology and magnitude cuts to a standardized catalog."""

    quality = config.QUALITY_CUT_SETTINGS
    required = {"BA", "r", "Re_semimajor", "sersic_n"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Cannot apply quality cuts; missing columns: {missing}")
    mask = (
        (df["BA"] > quality["ba_min"])
        & (df["BA"] < quality["ba_max"])
        & (df["r"] < quality["mag_max"])
        & (df["r"] > quality["mag_min"])
        & (df["Re_semimajor"] < quality["re_max"])
        & (df["Re_semimajor"] > quality["re_min"])
        & (df["sersic_n"] < quality["sersic_max"])
        & (df["sersic_n"] > quality["sersic_min"])
    )
    return df.loc[mask].reset_index(drop=True)


def validate_bias_source_schema(args, columns: tuple[str, ...]) -> None:
    """Validate the small catalog schema needed by ``halo_bias``."""

    available = set(columns)
    mass_source = getattr(args, "mass_source", config.BIAS_ASSIGNMENT_SETTINGS["mass_source"])
    log_column = getattr(args, "log_stellar_mass_column", None)
    linear_column = getattr(args, "stellar_mass_column", None)
    abs_mag_column = getattr(args, "abs_mag_column", config.BIAS_ASSIGNMENT_SETTINGS["abs_mag_column"])
    size_column = getattr(args, "size_column", config.SOURCE_SETTINGS["size_column"])

    if mass_source == "sps_surviving" and "log10_surviv_stellar_mass" not in available:
        raise KeyError("Mass source 'sps_surviving' requires 'log10_surviv_stellar_mass'.")
    if mass_source == "sps_formed" and "log10_stellar_mass" not in available:
        raise KeyError("Mass source 'sps_formed' requires 'log10_stellar_mass'.")
    if mass_source == "log_column":
        if not log_column:
            raise ValueError("mass_source='log_column' requires log_stellar_mass_column.")
        if log_column not in available:
            raise KeyError(f"Requested log-stellar-mass column {log_column!r} not found.")
    if mass_source == "linear_column":
        if not linear_column:
            raise ValueError("mass_source='linear_column' requires stellar_mass_column.")
        if linear_column not in available:
            raise KeyError(f"Requested stellar-mass column {linear_column!r} not found.")
    if mass_source == "luminosity_proxy" and abs_mag_column not in available:
        raise KeyError(f"Explicit luminosity-proxy run requires {abs_mag_column!r}.")
    if size_column is not None and size_column not in available:
        raise KeyError(f"Requested size column {size_column!r} not found.")
    if size_column is None and "int_r50_arcsec" not in available:
        raise KeyError(
            "No default angular-size column 'int_r50_arcsec' found. "
            "For GalSBI-SPS catalogs set SOURCE_SETTINGS['size_column']='r50' "
            "and SOURCE_SETTINGS['size_scale']=0.168 in config.py."
        )


def derived_generation_summary_path(catalog_path: str | Path) -> Path | None:
    path = Path(catalog_path)
    suffix = "_ucat.gal.cat"
    if not path.name.endswith(suffix):
        return None
    base = path.name[: -len(suffix)]
    if "_" not in base:
        return None
    prefix = base.rsplit("_", 1)[0]
    return path.with_name(f"{prefix}_generation_summary.json")


def load_generation_summary(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    summary_path = Path(path)
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def quality_cut_count(
    path: Path,
    chunk_size: int | None = None,
    size_column: str | None = None,
    size_scale: float | None = None,
    dataset: str | None = None,
) -> int | None:
    dataset = dataset or config.SOURCE_SETTINGS["dataset"]
    chunk_size = int(chunk_size or config.SPS_GENERATION_SETTINGS["summary_chunk_size"])
    size_column = size_column or config.SPS_GENERATION_SETTINGS["quality_size_column"]
    size_scale = (
        config.SPS_GENERATION_SETTINGS["quality_size_scale"]
        if size_scale is None
        else float(size_scale)
    )

    with h5py.File(path, "r") as h5:
        data = h5[dataset]
        names = set(data.dtype.names or ())
        required = {"e1", "e2", "int_mag", "sersic_n", size_column}
        if not required.issubset(names):
            return None
        total = 0
        for start in range(0, int(data.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(data.shape[0]))
            rows = data.fields(list(required))[start:stop]
            total += int(np.count_nonzero(quality_cut_mask(rows, size_column, size_scale)))
    return total


def quality_cut_density(
    path: Path,
    *,
    dataset: str | None = None,
    chunk_size: int | None = None,
    size_column: str | None = None,
    size_scale: float | None = None,
) -> dict[str, object]:
    """Measure post-quality-cut density using occupied RA/Dec HEALPix pixels."""

    dataset = dataset or config.SOURCE_SETTINGS["dataset"]
    chunk_size = int(chunk_size or config.SPS_GENERATION_SETTINGS["summary_chunk_size"])
    size_column = size_column or config.SPS_GENERATION_SETTINGS["quality_size_column"]
    size_scale = (
        config.SPS_GENERATION_SETTINGS["quality_size_scale"]
        if size_scale is None
        else float(size_scale)
    )
    nside_for_radec = int(config.SPS_GENERATION_SETTINGS["nside"])

    with h5py.File(path, "r") as h5:
        data = h5[dataset]
        names = set(data.dtype.names or ())
        required = {"e1", "e2", "int_mag", "sersic_n", size_column, "ra", "dec"}
        missing = sorted(required - names)
        if missing:
            return {
                "catalog_path": str(path),
                "catalog_dataset": dataset,
                "n_rows": int(data.shape[0]),
                "quality_columns_available": False,
                "missing_quality_columns": missing,
            }

        import healpy as hp

        post_cut_rows = 0
        occupied_chunks: list[np.ndarray] = []
        fields = list(required)
        for start in range(0, int(data.shape[0]), chunk_size):
            stop = min(start + chunk_size, int(data.shape[0]))
            rows = data.fields(fields)[start:stop]
            post_cut_rows += int(np.count_nonzero(quality_cut_mask(rows, size_column, size_scale)))
            pix = hp.ang2pix(
                nside_for_radec,
                rows["ra"].astype(float),
                rows["dec"].astype(float),
                lonlat=True,
            )
            occupied_chunks.append(np.unique(pix.astype(np.int64, copy=False)))

        occupied_pixels = (
            np.unique(np.concatenate(occupied_chunks))
            if occupied_chunks
            else np.empty(0, dtype=np.int64)
        )
        footprint = {
            "area_source": "catalog_ra_dec_occupied_pixels",
            "area_deg2": float(
                len(occupied_pixels) * hp.nside2pixarea(nside_for_radec, degrees=True)
            ),
            "nside": nside_for_radec,
            "n_pixels": int(len(occupied_pixels)),
        }

        n_rows = int(data.shape[0])

    area_arcmin2 = float(footprint["area_deg2"]) * 3600.0
    density = float(post_cut_rows / area_arcmin2) if area_arcmin2 > 0 else float("nan")
    return {
        "catalog_path": str(path),
        "catalog_dataset": dataset,
        "n_rows": n_rows,
        "post_quality_cut_rows": int(post_cut_rows),
        "quality_cut_fraction": float(post_cut_rows / n_rows) if n_rows > 0 else None,
        "post_quality_cut_density_arcmin2": density,
        "area_deg2": float(footprint["area_deg2"]),
        "area_arcmin2": area_arcmin2,
        "area_source": footprint["area_source"],
        "density_area_mode": "radec",
        "density_nside": footprint["nside"],
        "density_n_pixels": footprint["n_pixels"],
        "catalog_has_ra_dec": True,
        "quality_size_column": size_column,
        "quality_size_scale": float(size_scale),
    }


def _apply_basic_runtime_compatibility_shims() -> list[str]:
    notes: list[str] = []
    import torch  # noqa: F401

    python_bin = Path(sys.executable).resolve().parent
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(python_bin) not in path_parts[:1]:
        os.environ["PATH"] = os.pathsep.join([str(python_bin)] + path_parts)
        notes.append(f"prepended {python_bin} to PATH for sympy2c wrapper builds")

    if not hasattr(np, "trapezoid"):
        np.trapezoid = np.trapz
        notes.append("aliased numpy.trapezoid to numpy.trapz")
    return notes


def _apply_runtime_compatibility_shims() -> list[str]:
    notes = _apply_basic_runtime_compatibility_shims()

    import sklearn.exceptions as sklearn_exceptions

    if not hasattr(sklearn_exceptions, "InconsistentVersionWarning"):
        class InconsistentVersionWarning(UserWarning):
            pass

        sklearn_exceptions.InconsistentVersionWarning = InconsistentVersionWarning
        notes.append("added sklearn.exceptions.InconsistentVersionWarning shim")

    import sympy2c.vector as sympy2c_vector

    if not getattr(sympy2c_vector.VectorElement, "_skyvar_accepts_assumptions", False):
        orig_vector_element_new = sympy2c_vector.VectorElement.__new__
        orig_vector_new = sympy2c_vector.Vector.__new__

        def vector_element_new(cls, name, index=None, **assumptions):
            return orig_vector_element_new(cls, name, index)

        def vector_new(cls, name, size=None, **assumptions):
            return orig_vector_new(cls, name, size)

        sympy2c_vector.VectorElement.__new__ = staticmethod(vector_element_new)
        sympy2c_vector.Vector.__new__ = staticmethod(vector_new)
        sympy2c_vector.VectorElement._skyvar_accepts_assumptions = True
        sympy2c_vector.Vector._skyvar_accepts_assumptions = True
        notes.append("allowed sympy2c Vector/VectorElement to accept SymPy assumptions")

    import healpy as hp
    from ufig import coordinate_util
    import galsbi.ucat.galaxy_population_models.galaxy_position as galaxy_position

    if not getattr(galaxy_position, "_skyvar_robust_position_sampler", False):
        def sample_position_uniform(numobj, w, pixel_index, nside):
            numobj = int(numobj)
            if numobj <= 0:
                empty = np.empty(0, dtype=float)
                return empty, empty

            center = np.asarray(hp.pix2vec(int(nside), int(pixel_index)), dtype=float)
            radius = float(hp.max_pixrad(int(nside)) * 1.05)
            cos_radius = np.cos(radius)
            ref = np.array([0.0, 0.0, 1.0]) if abs(center[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            e1 = np.cross(ref, center)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(center, e1)

            theta_out = np.empty(numobj, dtype=float)
            phi_out = np.empty(numobj, dtype=float)
            filled = 0
            attempts = 0
            while filled < numobj:
                remaining = numobj - filled
                batch = max(1024, remaining * 3)
                cos_gamma = np.random.uniform(cos_radius, 1.0, size=batch)
                sin_gamma = np.sqrt(np.clip(1.0 - cos_gamma * cos_gamma, 0.0, None))
                alpha = np.random.uniform(0.0, 2.0 * np.pi, size=batch)
                vec = (
                    cos_gamma[:, None] * center[None, :]
                    + sin_gamma[:, None]
                    * (
                        np.cos(alpha)[:, None] * e1[None, :]
                        + np.sin(alpha)[:, None] * e2[None, :]
                    )
                )
                theta, phi = hp.vec2ang(vec)
                keep = hp.ang2pix(int(nside), theta, phi) == int(pixel_index)
                if np.any(keep):
                    take = min(int(np.count_nonzero(keep)), remaining)
                    theta_out[filled:filled + take] = theta[keep][:take]
                    phi_out[filled:filled + take] = phi[keep][:take]
                    filled += take
                attempts += 1
                if attempts > 1000:
                    raise RuntimeError(f"SkyVar robust GalSBI sampler stalled for pixel {pixel_index}.")

            if w is None:
                return coordinate_util.thetaphi2radec(theta_out, phi_out)
            return coordinate_util.thetaphi2xy(w, theta_out, phi_out)

        galaxy_position.sample_position_uniform = sample_position_uniform
        galaxy_position._skyvar_robust_position_sampler = True
        notes.append("patched GalSBI HEALPix position sampler for robust high-count source pools")
    return notes


def source_pixels_from_config(settings: dict[str, object] | None = None) -> list[int]:
    settings = dict(config.SPS_GENERATION_SETTINGS if settings is None else settings)
    nside = int(settings["nside"])
    count = int(settings["source_pixels"])
    n_pix = 12 * nside * nside

    if count <= 0:
        pixels = [int(settings["pixel"])]
    elif bool(settings.get("random_source_pixels", False)):
        if count > n_pix:
            raise ValueError(f"Requested {count} source pixels but nside={nside} has only {n_pix}.")
        rng = np.random.default_rng(int(settings["source_pixel_seed"]))
        pixels = [int(pixel) for pixel in np.sort(rng.choice(n_pix, size=count, replace=False))]
    else:
        if count > n_pix:
            raise ValueError(f"Requested {count} source pixels but nside={nside} has only {n_pix}.")
        start = int(settings["source_pixel_start"])
        if start < 0 or start >= n_pix:
            raise ValueError(f"source_pixel_start must be in [0, {n_pix}).")
        pixels = [int((start + offset) % n_pix) for offset in range(count)]

    bad = [pixel for pixel in pixels if pixel < 0 or pixel >= n_pix]
    if bad:
        raise ValueError(f"Pixels must be in [0, {n_pix}); bad values include {bad[:10]}.")
    return list(dict.fromkeys(pixels))


def galsbi_sps_expected_counts(
    nside: int,
    model_index: int = 0,
    posterior_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Return GalSBI-SPS intrinsic mean counts per HEALPix pixel."""

    _apply_basic_runtime_compatibility_shims()

    import importlib
    from types import SimpleNamespace

    import healpy as hp
    import PyCosmo
    from galsbi import load
    from galsbi.ucat_sps.galaxy_population_models.galaxy_stellar_mass_function import (
        initialize_stellar_mass_functions,
    )

    config_module = importlib.import_module(load.load_config("Tortorelli+25", "intrinsic"))
    values = {key: getattr(config_module, key) for key in dir(config_module) if not key.startswith("__")}

    posterior = (
        load_hdf5_column_table(posterior_path)
        if posterior_path not in (None, "")
        else load.load_abc_posterior("Tortorelli+25")
    )
    if int(model_index) < 0 or int(model_index) >= len(posterior):
        raise ValueError(f"model_index={model_index} is outside the GalSBI posterior range.")
    record = posterior[int(model_index)]
    values.update({column: record[column] for column in posterior.dtype.names})
    values["nside_sampling"] = int(nside)

    par = SimpleNamespace(**values)
    cosmo = PyCosmo.build()
    cosmo.set(h=par.h, omega_m=par.omega_m)
    pixarea_sr = hp.nside2pixarea(int(nside), degrees=False)
    pixarea_arcmin2 = hp.nside2pixarea(int(nside), degrees=True) * 3600.0

    sm_funcs_lowmass = initialize_stellar_mass_functions(
        par,
        cosmo=cosmo,
        pixarea=pixarea_sr,
        z_sm_intp=None,
        mass_key="lowmass",
    )
    sm_funcs_highmass = initialize_stellar_mass_functions(
        par,
        cosmo=cosmo,
        pixarea=pixarea_sr,
        z_sm_intp=None,
        mass_key="highmass",
    )

    components: dict[str, dict[str, int]] = {}
    total_per_pixel = 0
    for galaxy_type in par.galaxy_types:
        low = int(sm_funcs_lowmass[galaxy_type].n_gal_calc.n_gal_mean)
        high = int(sm_funcs_highmass[galaxy_type].n_gal_calc.n_gal_mean)
        components[str(galaxy_type)] = {
            "lowmass_mean_per_pixel": low,
            "highmass_mean_per_pixel": high,
            "total_mean_per_pixel": low + high,
        }
        total_per_pixel += low + high

    return {
        "model": "Tortorelli+25",
        "model_index": int(model_index),
        "posterior_path": "" if posterior_path in (None, "") else str(Path(posterior_path).expanduser()),
        "nside": int(nside),
        "pixel_area_arcmin2": float(pixarea_arcmin2),
        "components": components,
        "total_mean_per_pixel": int(total_per_pixel),
        "intrinsic_density_arcmin2": float(total_per_pixel / pixarea_arcmin2),
    }


def auto_sps_caps(
    expected_counts: dict[str, object],
    n_pixels: int,
    sigma: float,
) -> dict[str, object]:
    components = expected_counts["components"]
    caps: dict[str, int] = {}
    expected_by_type: dict[str, float] = {}
    for galaxy_type, values in components.items():
        mean = float(values["total_mean_per_pixel"]) * int(n_pixels)
        cap = int(math.ceil(mean + float(sigma) * math.sqrt(max(mean, 1.0))))
        caps[galaxy_type] = max(cap, int(math.ceil(mean)) + 10)
        expected_by_type[galaxy_type] = mean
    return {
        "cap_sigma": float(sigma),
        "expected_rows_by_type": expected_by_type,
        "auto_caps_by_type": caps,
        "expected_rows_total": float(sum(expected_by_type.values())),
        "auto_cap_total": int(sum(caps.values())),
    }


def resolve_sps_count_caps(
    settings: dict[str, object],
    pixels: list[int],
) -> tuple[dict[str, int], dict[str, object]]:
    expected_counts = galsbi_sps_expected_counts(
        int(settings["nside"]),
        model_index=int(settings["model_index"]),
        posterior_path=settings.get("posterior_path") or "",
    )
    auto_caps = auto_sps_caps(
        expected_counts,
        len(pixels),
        sigma=float(settings["auto_cap_sigma"]),
    )
    requested_caps = {
        "blue": None if settings["n_gal_max_blue"] is None else int(settings["n_gal_max_blue"]),
        "red": None if settings["n_gal_max_red"] is None else int(settings["n_gal_max_red"]),
    }
    resolved_caps = {
        galaxy_type: auto_caps["auto_caps_by_type"][galaxy_type]
        if requested_caps.get(galaxy_type) is None
        else int(requested_caps[galaxy_type])
        for galaxy_type in auto_caps["auto_caps_by_type"]
    }
    cap_limited_risk = {
        galaxy_type: requested_caps.get(galaxy_type) is not None
        and requested_caps[galaxy_type] < auto_caps["auto_caps_by_type"][galaxy_type]
        for galaxy_type in resolved_caps
    }

    expected_rows_total = float(auto_caps["expected_rows_total"])
    if (
        expected_rows_total > int(settings["max_expected_rows"])
        and not bool(settings["allow_large_density_run"])
    ):
        raise ValueError(
            "Density-faithful GalSBI-SPS generation would create about "
            f"{expected_rows_total:.3g} intrinsic rows over {len(pixels)} pixels. "
            f"The configured safety limit is {int(settings['max_expected_rows']):,}. "
            "Reduce SPS_GENERATION_SETTINGS['source_pixels'] or set "
            "SPS_GENERATION_SETTINGS['allow_large_density_run']=True in config.py."
        )
    if any(cap_limited_risk.values()):
        bad = {
            galaxy_type: {
                "requested": requested_caps[galaxy_type],
                "density_faithful_minimum": auto_caps["auto_caps_by_type"][galaxy_type],
            }
            for galaxy_type, is_bad in cap_limited_risk.items()
            if is_bad
        }
        raise ValueError(
            "Configured GalSBI n_gal_max caps would truncate the intrinsic density: "
            f"{bad}. Use None for SPS_GENERATION_SETTINGS['n_gal_max_blue'/'n_gal_max_red'] "
            "or raise the caps in config.py."
        )
    return resolved_caps, {
        "expected_counts": expected_counts,
        "auto_cap_plan": auto_caps,
        "requested_caps": requested_caps,
        "resolved_caps": resolved_caps,
        "cap_limited_risk": cap_limited_risk,
    }


def _stable_column_seed(column: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(column)) % (2**32)


def _streaming_finite_stats(
    data,
    columns: Sequence[str],
    chunk_size: int,
    sample_size: int,
) -> dict[str, dict[str, float | int | bool | None]]:
    n_rows = int(data.shape[0])
    states = {
        column: {
            "finite": 0,
            "min": math.inf,
            "max": -math.inf,
            "samples": [],
            "rng": np.random.default_rng(_stable_column_seed(column)),
        }
        for column in columns
    }

    for start in range(0, n_rows, int(chunk_size)):
        stop = min(start + int(chunk_size), n_rows)
        chunk_rows = stop - start
        for column, state in states.items():
            arr = np.asarray(data.fields(column)[start:stop])
            vals = arr[np.isfinite(arr)].astype(float, copy=False)
            if vals.size == 0:
                continue
            state["finite"] = int(state["finite"]) + int(vals.size)
            state["min"] = min(float(state["min"]), float(np.min(vals)))
            state["max"] = max(float(state["max"]), float(np.max(vals)))

            if n_rows <= sample_size:
                sample = vals
            else:
                target = max(1, int(math.ceil(float(sample_size) * float(chunk_rows) / float(n_rows))))
                if vals.size <= target:
                    sample = vals
                else:
                    rng = state["rng"]
                    sample = vals[rng.choice(vals.size, size=target, replace=False)]
            state["samples"].append(np.asarray(sample, dtype=float))

    output: dict[str, dict[str, float | int | bool | None]] = {}
    for column, state in states.items():
        finite = int(state["finite"])
        samples = state["samples"]
        if finite == 0 or not samples:
            output[column] = {
                "finite": 0,
                "min": None,
                "median": None,
                "max": None,
                "median_is_approximate": False,
                "median_sample_size": 0,
            }
            continue
        sample_values = np.concatenate(samples)
        output[column] = {
            "finite": finite,
            "min": float(state["min"]),
            "median": float(np.median(sample_values)),
            "max": float(state["max"]),
            "median_is_approximate": int(sample_values.size) < finite,
            "median_sample_size": int(sample_values.size),
        }
    return output


def inspect_generated_catalogs(
    prefix: Path,
    model_index: int,
    filters: Sequence[str],
    chunk_size: int,
    quality_size_column: str,
    quality_size_scale: float,
    sample_size: int,
) -> dict[str, object]:
    stat_columns = (
        "z",
        "log10_stellar_mass",
        "log10_surviv_stellar_mass",
        "mag",
        "int_mag",
        "abs_mag",
        "r50",
        "int_r50",
        "e1",
        "e2",
        "sersic_n",
        "ra",
        "dec",
        "galaxy_type",
        "excess_b_v",
    )
    required_columns = {
        "z",
        "int_mag",
        quality_size_column,
        "e1",
        "e2",
        "sersic_n",
        "ra",
        "dec",
        "log10_surviv_stellar_mass",
    }
    output: dict[str, object] = {}
    for band in filters:
        path = Path(f"{prefix}_{model_index}_{band}_ucat.gal.cat")
        if not path.exists():
            output[band] = {"path": str(path), "exists": False}
            continue
        with h5py.File(path, "r") as h5:
            data = h5["data"]
            columns = [column for column in stat_columns if column in (data.dtype.names or ())]
            band_summary = {
                "path": str(path),
                "exists": True,
                "n_rows": int(data.shape[0]),
                "columns": list(data.dtype.names or ()),
                "missing_required_columns": sorted(required_columns - set(data.dtype.names or ())),
                "stats": _streaming_finite_stats(
                    data,
                    columns,
                    chunk_size=chunk_size,
                    sample_size=sample_size,
                ),
            }
        if band == "r":
            band_summary["post_quality_cut_rows"] = quality_cut_count(
                path,
                chunk_size=chunk_size,
                size_column=quality_size_column,
                size_scale=quality_size_scale,
            )
        output[band] = band_summary
    return output


def generate_sps_catalog() -> dict[str, object]:
    """Generate a GalSBI-SPS intrinsic catalog from ``config.SPS_GENERATION_SETTINGS``."""

    settings = dict(config.SPS_GENERATION_SETTINGS)
    shim_notes = _apply_runtime_compatibility_shims()
    if bool(settings["rebuild_wrapper"]):
        os.environ["REBUILD_WRAPPER"] = "1"
        shim_notes.append("set REBUILD_WRAPPER=1")

    import healpy as hp
    from galsbi import GalSBI

    output_prefix = Path(str(settings["output_prefix"]))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pixels = source_pixels_from_config(settings)
    nside = int(settings["nside"])
    n_pix = 12 * nside * nside
    pixels_arr = np.asarray(pixels, dtype=np.int64)
    pixel_ordering = str(settings["pixel_ordering"]).strip().lower()
    if pixel_ordering == "ring":
        map_pixels = pixels_arr
    elif pixel_ordering == "nest":
        map_pixels = hp.nest2ring(nside, pixels_arr)
    else:
        raise ValueError("SPS_GENERATION_SETTINGS['pixel_ordering'] must be 'ring' or 'nest'.")

    healpix_map = np.zeros(n_pix, dtype=np.float32)
    healpix_map[map_pixels] = 1.0
    resolved_caps, count_plan = resolve_sps_count_caps(settings, pixels)

    kwargs = {
        "mode": "intrinsic",
        "model_index": int(settings["model_index"]),
        "file_name": str(output_prefix),
        "nside_sampling": nside,
        "healpix_map": healpix_map,
        "n_gal_max_blue": resolved_caps["blue"],
        "n_gal_max_red": resolved_caps["red"],
        "raise_max_num_gal_error": True,
    }
    if settings["seed"] is not None:
        kwargs["seed"] = int(settings["seed"])

    posterior_path = str(settings.get("posterior_path", "")).strip()
    if posterior_path:
        posterior_path = str(Path(posterior_path).expanduser())

    model = GalSBI("Tortorelli+25", verbosity=str(settings["verbosity"]))
    with use_custom_galsbi_posterior("Tortorelli+25", posterior_path) as custom_posterior:
        if custom_posterior is not None and int(settings["model_index"]) >= len(custom_posterior):
            raise ValueError(
                f"model_index={settings['model_index']} is outside custom posterior "
                f"{posterior_path} with {len(custom_posterior)} rows."
            )
        model(**kwargs)

    summary_path = Path(f"{output_prefix}_{settings['model_index']}_generation_summary.json")
    summary: dict[str, object] = {
        "model": "Tortorelli+25",
        "mode": "intrinsic",
        "model_index": int(settings["model_index"]),
        "posterior_source": "custom" if posterior_path else "galsbi_default",
        "posterior_path": posterior_path,
        "output_prefix": str(output_prefix),
        "nside": nside,
        "pixel": pixels[0] if len(pixels) == 1 else None,
        "pixels": pixels if len(pixels) <= 2048 else None,
        "pixels_preview": pixels[:20],
        "pixels_tail": pixels[-20:] if len(pixels) > 20 else [],
        "n_pixels": len(pixels),
        "pixel_source": "source_pixels_random" if bool(settings["random_source_pixels"]) else "source_pixels",
        "input_pixel_ordering": pixel_ordering,
        "galsbi_map_ordering": "ring",
        "healpix_area_deg2": float(4.0 * np.pi * (180.0 / np.pi) ** 2 / n_pix),
        "healpix_area_deg2_total": float(len(pixels) * 4.0 * np.pi * (180.0 / np.pi) ** 2 / n_pix),
        "n_gal_max_blue": int(resolved_caps["blue"]),
        "n_gal_max_red": int(resolved_caps["red"]),
        "count_generation_mode": "density_faithful_auto_cap",
        "density_faithful": True,
        "galsbi_expected_counts": count_plan["expected_counts"],
        "galsbi_auto_cap_plan": count_plan["auto_cap_plan"],
        "galsbi_requested_caps": count_plan["requested_caps"],
        "galsbi_resolved_caps": count_plan["resolved_caps"],
        "quality_size_column": settings["quality_size_column"],
        "quality_size_scale": float(settings["quality_size_scale"]),
        "source_pixels": int(settings["source_pixels"]),
        "source_pixel_start": int(settings["source_pixel_start"]),
        "source_pixel_seed": int(settings["source_pixel_seed"]),
        "seed_override": settings["seed"],
        "filters": list(model.filters),
        "runtime_compatibility_notes": shim_notes,
        "catalogs": inspect_generated_catalogs(
            output_prefix,
            int(settings["model_index"]),
            list(model.filters),
            chunk_size=int(settings["summary_chunk_size"]),
            quality_size_column=str(settings["quality_size_column"]),
            quality_size_scale=float(settings["quality_size_scale"]),
            sample_size=int(settings["summary_stat_sample_size"]),
        ),
        "summary_path": str(summary_path),
    }

    r_catalog_path = Path(f"{output_prefix}_{settings['model_index']}_r_ucat.gal.cat")
    if r_catalog_path.exists():
        summary["post_quality_density"] = quality_cut_density(
            r_catalog_path,
            dataset=config.SOURCE_SETTINGS["dataset"],
            chunk_size=int(settings["summary_chunk_size"]),
            size_column=str(settings["quality_size_column"]),
            size_scale=float(settings["quality_size_scale"]),
        )
    utils.write_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        raise SystemExit(
            "bias_variation.catalog no longer accepts CLI subcommands or flags. "
            "Edit bias_variation/config.py and run `python -m bias_variation.catalog`."
        )
    print(json.dumps(generate_sps_catalog(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
