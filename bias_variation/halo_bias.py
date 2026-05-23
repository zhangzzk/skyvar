"""Colossus/Tinker halo-bias adapter."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, shmr, utils


SOURCE_COLUMNS = (
    "z",
    "int_mag",
    "int_r50_arcsec",
    "abs_mag",
    "sersic_n",
    "e1",
    "e2",
    "ra",
    "dec",
    "galaxy_type",
    "log10_surviv_stellar_mass",
    "log10_stellar_mass",
    "log10_mstar",
)

BIAS_SOURCE_COLUMNS = (
    "sersic_n",
    "r",
    "Re",
    "redshift",
    "BA",
    "angle",
    "bias_tinker10",
)

BIAS_SOURCE_DIAGNOSTIC_COLUMNS = (
    "Re_semimajor",
    "log10_surviv_stellar_mass",
    "log10_stellar_mass",
    "log10_mstar",
    "log10_mhalo_median",
    "log10_mhalo",
    "mhalo_msun",
)


@dataclass(frozen=True)
class BiasAssignmentConfig:
    """Configuration for the GalSBI-SPS to halo-mass and bias relation."""

    redshift_column: str = "redshift"
    mass_source: str = "sps_surviving"
    log_stellar_mass_column: str | None = None
    stellar_mass_column: str | None = None
    abs_mag_column: str = "abs_mag"
    solar_abs_mag_r: float = 4.65
    mass_to_light: float = 1.0
    scatter_dex: float = 0.15
    seed: int = 20260516
    inverse_mode: str = "interp"
    interp_z_nodes: int = 80
    interp_halo_grid: int = 2048
    skip_colossus: bool = False
    bias_mode: str = "interp"
    bias_z_nodes: int = 80
    bias_mass_grid: int = 2048
    mdef: str = "200c"
    h: float = float(config.BIAS_ASSIGNMENT_SETTINGS["h"])
    bias_column: str = "bias_tinker10"


def choose_stellar_mass(df: pd.DataFrame, config: BiasAssignmentConfig) -> tuple[np.ndarray, str]:
    """Return stellar mass in Msun and a recorded source label."""

    if config.mass_source == "sps_surviving":
        column = "log10_surviv_stellar_mass"
        if column not in df.columns:
            raise KeyError(
                f"Requested SPS surviving stellar-mass column {column!r} not found. "
                "Use a GalSBI-SPS catalog or explicitly pass --mass-source luminosity_proxy "
                "for diagnostic proxy runs."
            )
        log_mstar = df[column].to_numpy(dtype=float)
        return np.power(10.0, log_mstar), f"log10_column:{column}"

    if config.mass_source == "sps_formed":
        column = "log10_stellar_mass"
        if column not in df.columns:
            raise KeyError(
                f"Requested SPS formed stellar-mass column {column!r} not found. "
                "Use a GalSBI-SPS catalog or choose another explicit --mass-source."
            )
        log_mstar = df[column].to_numpy(dtype=float)
        return np.power(10.0, log_mstar), f"log10_column:{column}"

    if config.mass_source == "log_column":
        if not config.log_stellar_mass_column:
            raise ValueError("--mass-source log_column requires --log-stellar-mass-column.")
        if config.log_stellar_mass_column not in df.columns:
            raise KeyError(
                f"Requested log-stellar-mass column {config.log_stellar_mass_column!r} not found."
            )
        log_mstar = df[config.log_stellar_mass_column].to_numpy(dtype=float)
        return np.power(10.0, log_mstar), f"log10_column:{config.log_stellar_mass_column}"

    if config.mass_source == "linear_column":
        if not config.stellar_mass_column:
            raise ValueError("--mass-source linear_column requires --stellar-mass-column.")
        if config.stellar_mass_column not in df.columns:
            raise KeyError(f"Requested stellar-mass column {config.stellar_mass_column!r} not found.")
        return df[config.stellar_mass_column].to_numpy(dtype=float), f"column:{config.stellar_mass_column}"

    if config.mass_source == "luminosity_proxy":
        if config.abs_mag_column not in df.columns:
            raise KeyError(
                "Explicit luminosity-proxy mass source requested, but absolute-magnitude column "
                f"{config.abs_mag_column!r} is unavailable."
            )
        mstar = shmr.stellar_mass_proxy_from_absolute_magnitude(
            df[config.abs_mag_column].to_numpy(dtype=float),
            solar_abs_mag_r=config.solar_abs_mag_r,
            mass_to_light=config.mass_to_light,
        )
        return mstar, f"luminosity_proxy:{config.abs_mag_column},M/L={config.mass_to_light}"

    raise ValueError(
        "mass_source must be one of: sps_surviving, sps_formed, log_column, "
        "linear_column, luminosity_proxy."
    )


def assignment_metadata(config: BiasAssignmentConfig, mass_source: str) -> dict[str, object]:
    """Return JSON-serializable metadata for a bias assignment run."""

    return {
        "bias_assignment_config": asdict(config),
        "mass_source": mass_source,
        "scatter_interpretation": "additive Gaussian scatter in log10 inferred halo mass",
        "scatter_dex": float(config.scatter_dex),
        "inverse_mode": config.inverse_mode,
        "interp_z_nodes": None if config.inverse_mode == "exact" else int(config.interp_z_nodes),
        "interp_halo_grid": None if config.inverse_mode == "exact" else int(config.interp_halo_grid),
        "colossus_used": not bool(config.skip_colossus),
        "colossus_mdef": None if config.skip_colossus else config.mdef,
        "colossus_mass_conversion": (
            None if config.skip_colossus else "M_colossus[Msun/h] = M_halo[Msun] * h"
        ),
        "bias_mode": None if config.skip_colossus else config.bias_mode,
        "bias_z_nodes": (
            None if config.skip_colossus or config.bias_mode == "exact" else int(config.bias_z_nodes)
        ),
        "bias_mass_grid": (
            None if config.skip_colossus or config.bias_mode == "exact" else int(config.bias_mass_grid)
        ),
    }


def _default_cosmo_params() -> dict[str, float]:
    cosmo = config.COSMOLOGY_SETTINGS
    return {
        "flat": True,
        "H0": 100.0 * float(cosmo["h"]),
        "Om0": float(cosmo["Omega_c"]) + float(cosmo["Omega_b"]),
        "Ob0": float(cosmo["Omega_b"]),
        "sigma8": float(cosmo["sigma8"]),
        "ns": float(cosmo["n_s"]),
    }


def configure_colossus_cosmology(name: str = "skyvar_bias_variation") -> None:
    """Register and activate the SkyVar fiducial cosmology in Colossus."""

    from colossus.cosmology import cosmology

    params = _default_cosmo_params()
    existing = cosmology.cosmologies
    if name not in existing:
        cosmology.addCosmology(name, params)
    cosmology.setCosmology(name)


def tinker10_halo_bias(
    halo_mass_msun: np.ndarray,
    redshift: np.ndarray,
    h: float = float(config.BIAS_ASSIGNMENT_SETTINGS["h"]),
    mdef: str = "200c",
    model: str = "tinker10",
) -> np.ndarray:
    """Evaluate halo bias with ``colossus.lss.bias.haloBias``.

    Colossus expects halo mass in ``Msun / h``.  The prototype SHMR functions
    return numeric solar masses, so this adapter multiplies by ``h`` before
    calling Colossus.  The default mass definition is ``200c`` to match the
    Moster+13 virial-mass convention used by the baseline SHMR.
    """

    try:
        from colossus.lss import bias
    except ImportError as exc:
        raise ImportError(
            "Colossus is required for Tinker+10 halo bias. Install it with "
            "`python -m pip install colossus` in the Python environment used "
            "for this project."
        ) from exc

    configure_colossus_cosmology()
    halo_mass_msun = np.asarray(halo_mass_msun, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if halo_mass_msun.shape != redshift.shape:
        raise ValueError("halo_mass_msun and redshift must have the same shape.")

    out = np.full(halo_mass_msun.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(halo_mass_msun)
        & (halo_mass_msun > 0)
        & np.isfinite(redshift)
        & (redshift >= 0)
    )
    if not np.any(valid):
        return out
    mass_msun_h = halo_mass_msun[valid] * float(h)
    out[valid] = bias.haloBias(mass_msun_h, z=redshift[valid], mdef=mdef, model=model)
    return out


def tinker10_halo_bias_interpolated(
    halo_mass_msun: np.ndarray,
    redshift: np.ndarray,
    h: float = float(config.BIAS_ASSIGNMENT_SETTINGS["h"]),
    mdef: str = "200c",
    model: str = "tinker10",
    n_z_nodes: int = 80,
    n_mass_grid: int = 2048,
    log_mass_min: float | None = None,
    log_mass_max: float | None = None,
) -> np.ndarray:
    """Fast Tinker+10 bias using Colossus on a redshift/mass grid."""

    try:
        from colossus.lss import bias
    except ImportError as exc:
        raise ImportError(
            "Colossus is required for Tinker+10 halo bias. Install it with "
            "`python -m pip install colossus` in the Python environment used "
            "for this project."
        ) from exc

    configure_colossus_cosmology()
    halo_mass_msun = np.asarray(halo_mass_msun, dtype=float)
    redshift = np.asarray(redshift, dtype=float)
    if halo_mass_msun.shape != redshift.shape:
        raise ValueError("halo_mass_msun and redshift must have the same shape.")

    out = np.full(halo_mass_msun.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(halo_mass_msun)
        & (halo_mass_msun > 0)
        & np.isfinite(redshift)
        & (redshift >= 0)
    )
    if not np.any(valid):
        return out

    log_mass = np.log10(halo_mass_msun[valid])
    z_valid = redshift[valid]
    z_min = float(np.min(z_valid))
    z_max = float(np.max(z_valid))
    if z_min == z_max:
        z_nodes = np.array([z_min])
    else:
        z_nodes = np.linspace(z_min, z_max, max(2, int(n_z_nodes)))

    if log_mass_min is None:
        log_mass_min = max(6.0, float(np.nanmin(log_mass)) - 0.1)
    if log_mass_max is None:
        log_mass_max = min(17.5, float(np.nanmax(log_mass)) + 0.1)
    log_mass_grid = np.linspace(float(log_mass_min), float(log_mass_max), int(n_mass_grid))
    mass_grid_msun_h = (10.0**log_mass_grid) * float(h)

    z_index = np.abs(z_valid[:, None] - z_nodes[None, :]).argmin(axis=1)
    out_valid = np.full(log_mass.shape, np.nan, dtype=float)
    for i, z_node in enumerate(z_nodes):
        use = z_index == i
        if not np.any(use):
            continue
        bias_grid = bias.haloBias(mass_grid_msun_h, z=float(z_node), mdef=mdef, model=model)
        good = np.isfinite(bias_grid)
        out_valid[use] = np.interp(
            log_mass[use],
            log_mass_grid[good],
            bias_grid[good],
            left=np.nan,
            right=np.nan,
        )

    out[valid] = out_valid
    return out


def assign_galaxy_bias(
    df: pd.DataFrame,
    config: BiasAssignmentConfig,
    copy: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Add stellar-mass, halo-mass, and optional Tinker+10 bias columns."""

    if config.redshift_column not in df.columns:
        raise KeyError(f"Redshift column {config.redshift_column!r} not found.")
    if config.inverse_mode not in {"interp", "exact"}:
        raise ValueError("inverse_mode must be 'interp' or 'exact'.")
    if config.bias_mode not in {"interp", "exact"}:
        raise ValueError("bias_mode must be 'interp' or 'exact'.")

    out = df.copy() if copy else df
    redshift = out[config.redshift_column].to_numpy(dtype=float)
    mstar, mass_source = choose_stellar_mass(out, config)

    out["log10_mstar"] = np.where(mstar > 0, np.log10(mstar), np.nan)
    out["mstar_source"] = mass_source

    if config.inverse_mode == "exact":
        log_mhalo_median = shmr.inverse_moster13_halo_mass(mstar, redshift)
    else:
        log_mhalo_median = shmr.inverse_moster13_halo_mass_interpolated(
            mstar,
            redshift,
            n_z_nodes=config.interp_z_nodes,
            n_halo_grid=config.interp_halo_grid,
        )
    out["log10_mhalo_median"] = log_mhalo_median
    out["log10_mhalo"] = shmr.apply_log_normal_scatter_to_log_mass(
        log_mhalo_median,
        scatter_dex=config.scatter_dex,
        seed=config.seed,
    )
    out["mhalo_msun"] = 10.0 ** out["log10_mhalo"]

    if not config.skip_colossus:
        if config.bias_mode == "exact":
            out[config.bias_column] = tinker10_halo_bias(
                out["mhalo_msun"].to_numpy(dtype=float),
                redshift,
                h=config.h,
                mdef=config.mdef,
                model="tinker10",
            )
        else:
            out[config.bias_column] = tinker10_halo_bias_interpolated(
                out["mhalo_msun"].to_numpy(dtype=float),
                redshift,
                h=config.h,
                mdef=config.mdef,
                model="tinker10",
                n_z_nodes=config.bias_z_nodes,
                n_mass_grid=config.bias_mass_grid,
            )

    return out, assignment_metadata(config, mass_source)


def summarize_halo_convention(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"finite": 0}
    return {
        "finite": int(finite.size),
        "fraction_lt_1": float(np.mean(finite < 1.0)),
        "p01": float(np.percentile(finite, 1)),
        "p50": float(np.percentile(finite, 50)),
        "p99": float(np.percentile(finite, 99)),
    }


def bias_source_table_columns(df: pd.DataFrame) -> list[str]:
    """Return the compact source-table columns consumed by selection."""

    missing = sorted(set(BIAS_SOURCE_COLUMNS) - set(df.columns))
    if missing:
        raise KeyError(f"Bias-assigned source table is missing required columns: {missing}")
    optional = [col for col in BIAS_SOURCE_DIAGNOSTIC_COLUMNS if col in df.columns]
    return list(dict.fromkeys([*BIAS_SOURCE_COLUMNS, *optional]))


def build_bias_source_from_galsbi(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a GalSBI/SPS catalog and add halo mass plus Tinker+10 bias."""

    from . import catalog

    source_cfg = config.SOURCE_SETTINGS
    bias_cfg = config.BIAS_ASSIGNMENT_SETTINGS

    cat_path = Path(args.catalog)
    info = catalog.inspect_hdf5_catalog(cat_path, source_cfg["dataset"])
    catalog.validate_bias_source_schema(args, info.columns)

    columns = [name for name in SOURCE_COLUMNS if name in info.columns]
    for name in (
        args.log_stellar_mass_column,
        args.stellar_mass_column,
        bias_cfg["abs_mag_column"],
        args.size_column,
    ):
        if name and name in info.columns:
            columns.append(name)
    columns = list(dict.fromkeys(columns))

    df = catalog.load_galsbi_subset(
        path=cat_path,
        dataset=source_cfg["dataset"],
        n_rows=source_cfg["source_rows"],
        seed=source_cfg["seed"],
        random=bool(args.random_source_rows),
        columns=columns,
    )
    df = catalog.standardize_galsbi_columns(
        df,
        size_column=args.size_column,
        size_scale=float(source_cfg["size_scale"]),
    )
    rows_before_cuts = len(df)
    df = catalog.apply_selection_quality_cuts(df)

    assignment_config = BiasAssignmentConfig(
        mass_source=args.mass_source,
        log_stellar_mass_column=args.log_stellar_mass_column,
        stellar_mass_column=args.stellar_mass_column,
        abs_mag_column=bias_cfg["abs_mag_column"],
        solar_abs_mag_r=float(bias_cfg["solar_abs_mag_r"]),
        mass_to_light=float(bias_cfg["mass_to_light"]),
        scatter_dex=float(bias_cfg["scatter_dex"]),
        seed=int(source_cfg["seed"]),
        inverse_mode=args.inverse_mode,
        interp_z_nodes=int(bias_cfg["interp_z_nodes"]),
        interp_halo_grid=int(bias_cfg["interp_halo_grid"]),
        skip_colossus=args.skip_colossus,
        bias_mode=args.bias_mode,
        bias_z_nodes=int(bias_cfg["bias_z_nodes"]),
        bias_mass_grid=int(bias_cfg["bias_mass_grid"]),
        mdef=str(bias_cfg["mdef"]),
        h=float(bias_cfg["h"]),
    )
    df, assignment_meta = assign_galaxy_bias(df, assignment_config, copy=False)
    re_values = df["Re_semimajor"] if "Re_semimajor" in df.columns else df["Re"]
    source = pd.DataFrame(
        {
            "sersic_n": df["sersic_n"].to_numpy(np.float32),
            "r": df["r"].to_numpy(np.float32),
            "Re": re_values.to_numpy(np.float32),
            "redshift": df["redshift"].to_numpy(np.float32),
            "BA": df["BA"].to_numpy(np.float32),
            "angle": df["angle"].to_numpy(np.float32),
            "bias_tinker10": df["bias_tinker10"].to_numpy(np.float32),
        }
    )
    for col in BIAS_SOURCE_DIAGNOSTIC_COLUMNS:
        if col in df.columns and col not in source.columns:
            source[col] = df[col].to_numpy()
    bias_source_table_columns(source)

    density_summary = catalog.quality_cut_density(
        cat_path,
        dataset=source_cfg["dataset"],
        chunk_size=int(config.SPS_GENERATION_SETTINGS["summary_chunk_size"]),
        size_column=args.size_column or config.SPS_GENERATION_SETTINGS["quality_size_column"],
        size_scale=float(source_cfg["size_scale"]),
    )
    computed_density = density_summary.get("post_quality_cut_density_arcmin2")

    summary: dict[str, object] = {
        "catalog_path": str(cat_path),
        "catalog_dataset": source_cfg["dataset"],
        "catalog_rows_total": int(info.n_rows),
        "source_rows_requested": int(source_cfg["source_rows"]),
        "source_rows_loaded_before_cuts": int(rows_before_cuts),
        "source_rows_after_quality_cuts": int(len(source)),
        "random_source_rows": bool(args.random_source_rows),
        "seed": int(source_cfg["seed"]),
        "size_column": args.size_column or "int_r50_arcsec_if_present",
        "size_scale": float(source_cfg["size_scale"]),
        "source_density_arcmin2": None if computed_density is None else float(computed_density),
        "source_density_source": "catalog_post_quality_cut_density",
        "source_density_catalog_diagnostic": density_summary,
        "source_columns": list(source.columns),
        "bias_chain": utils.summarize_bias_chain(df, scatter_dex=float(bias_cfg["scatter_dex"])),
    }
    summary.update(assignment_meta)
    return source, summary


def build_assign_source_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a bias-assigned source table for bias_variation.selection."
    )
    parser.add_argument(
        "--catalog",
        default=config.SOURCE_SETTINGS.get("catalog") or None,
        help="GalSBI/SPS HDF5 catalog path. Defaults to SOURCE_SETTINGS['catalog'].",
    )
    parser.add_argument(
        "--output-table",
        "--output",
        dest="output_table",
        default=config.SOURCE_SETTINGS.get("bias_source_table") or None,
        help=(
            "Output Feather source table consumed by bias_variation.selection. "
            "Defaults to SOURCE_SETTINGS['bias_source_table']."
        ),
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Optional JSON summary path. Defaults to <output-table>_summary.json.",
    )
    parser.add_argument("--random-source-rows", action="store_true", help="Draw source rows randomly.")
    parser.add_argument("--size-column", default=config.SOURCE_SETTINGS["size_column"])
    parser.add_argument(
        "--mass-source",
        choices=("sps_surviving", "sps_formed", "log_column", "linear_column", "luminosity_proxy"),
        default=config.BIAS_ASSIGNMENT_SETTINGS["mass_source"],
    )
    parser.add_argument("--log-stellar-mass-column", default=None)
    parser.add_argument("--stellar-mass-column", default=None)
    parser.add_argument("--inverse-mode", choices=("interp", "exact"), default="interp")
    parser.add_argument("--bias-mode", choices=("interp", "exact"), default="interp")
    parser.add_argument("--skip-colossus", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def assign_source_table(args: argparse.Namespace) -> dict[str, object]:
    if not args.catalog:
        raise ValueError("No input catalog configured. Pass --catalog or set SOURCE_SETTINGS['catalog'].")
    if not args.output_table:
        raise ValueError(
            "No output source table configured. Pass --output-table or set "
            "SOURCE_SETTINGS['bias_source_table']."
        )
    output_table = Path(args.output_table)
    if output_table.exists() and not args.overwrite:
        raise FileExistsError(f"Output source table already exists: {output_table}")
    output_table.parent.mkdir(parents=True, exist_ok=True)

    source, summary = build_bias_source_from_galsbi(args)
    source.to_feather(output_table)

    summary_path = (
        Path(args.summary)
        if args.summary
        else output_table.with_name(f"{output_table.stem}_summary.json")
    )
    summary["output_table"] = str(output_table)
    summary["output_summary"] = str(summary_path)
    utils.write_json(summary_path, summary)
    return summary


def build_diagnose_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Colossus halo-bias mass-definition choices.")
    parser.add_argument("table", help="Bias source Feather table.")
    parser.add_argument(
        "--mdefs",
        nargs="+",
        default=("200c", "vir", "200m", "500c"),
        help="Colossus mass definitions to compare.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to <table>_halo_convention.json.",
    )
    return parser


def diagnose_halo_convention(args: argparse.Namespace) -> dict:
    table = Path(args.table)
    df = pd.read_feather(table)
    required = {"mhalo_msun", "redshift"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns in {table}: {missing}")

    mhalo = df["mhalo_msun"].to_numpy(dtype=float)
    redshift = df["redshift"].to_numpy(dtype=float)
    h = float(config.BIAS_ASSIGNMENT_SETTINGS["h"])
    output: dict[str, object] = {
        "table": str(table),
        "n_rows": int(len(df)),
        "mass_unit_input": "Msun",
        "colossus_mass_conversion": "M_colossus[Msun/h] = M_halo[Msun] * h",
        "h": h,
        "mdefs": {},
    }
    for mdef in args.mdefs:
        bias_values = tinker10_halo_bias(
            mhalo,
            redshift,
            h=h,
            mdef=mdef,
            model="tinker10",
        )
        output["mdefs"][mdef] = summarize_halo_convention(bias_values)

    output_path = Path(args.output) if args.output else table.with_name(f"{table.stem}_halo_convention.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output["output"] = str(output_path)
    utils.write_json(output_path, output)
    return output


_COMMANDS = {
    "assign-source": (build_assign_source_parser, assign_source_table),
    "diagnose": (build_diagnose_parser, diagnose_halo_convention),
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
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
