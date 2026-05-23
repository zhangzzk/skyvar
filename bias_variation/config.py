"""Configuration for the bias-variation workflows.

Runtime choices live in this file.  Jobs should call the Python entry points
directly; edit this file to change run settings.
"""

from __future__ import annotations

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = _PACKAGE_DIR.parent
OUTPUT_DIR = BASE_DIR / "bias_variation" / "output"
PROJECT_GALSBI_DIR = Path("/project/ls-gruen/users/zekang.zhang/cats/galsbi")

SPS_POSTERIOR_PATH = BASE_DIR / "notebooks" / "abc_posterior_best_fit.h5"



# ==============================================================================
# 1. PATHS & I/O
# ==============================================================================
PATHS = {
    "output_dir": str(OUTPUT_DIR),
    "model_json": "",
    "blendemu_dir": "",
}


# ==============================================================================
# 2. BIAS ASSIGNMENT
# ==============================================================================
COSMOLOGY_SETTINGS = {
    "Omega_c": 0.2645,
    "Omega_b": 0.0493,
    "h": 0.6736,
    "sigma8": 0.8111,
    "n_s": 0.9649,
}

BIAS_ASSIGNMENT_SETTINGS = {
    "mass_source": "sps_surviving",
    "abs_mag_column": "abs_mag",
    "solar_abs_mag_r": 4.65,
    "mass_to_light": 1.0,
    "scatter_dex": 0.15,
    "interp_z_nodes": 80,
    "interp_halo_grid": 2048,
    "bias_z_nodes": 80,
    "bias_mass_grid": 2048,
    "mdef": "200c",
    "h": COSMOLOGY_SETTINGS["h"],
}


# ==============================================================================
# 3. SOURCE POOLS & CLASSIFICATION
# ==============================================================================
SOURCE_SETTINGS = {
    "catalog": str(PROJECT_GALSBI_DIR / "galsbi_sps_bestfit_1m_0_r_ucat.gal.cat"),
    "bias_source_table": str(PROJECT_GALSBI_DIR / "galsbi_sps_bestfit_1m_0_r_ucat_bias_source.feather"),
    "target_density_arcmin2": 182.73576027397755,
    "dataset": "data",
    "source_rows": 0,
    "seed": 20260516,
    "size_column": "r50",
    "size_scale": 0.168,
    "diagnostic_points": 120_000,
}

QUALITY_CUT_SETTINGS = {
    "ba_min": 0.05,
    "ba_max": 1.0,
    "mag_min": 0.0,
    "mag_max": 28.0,
    "re_min": 0.01,
    "re_max": 5.0,
    "sersic_min": 0.5,
    "sersic_max": 6.0,
}


# ==============================================================================
# 4. FULL SKYVAR-STYLE RUNS
# ==============================================================================
FULL_WORKFLOW_SETTINGS = {
    "max_pixels": None,
    "z_min": None,
    "z_max": None,
    "z_bins": None,
    "pixel_bz_count": 0,
    "pixel_bz_z_bins": 15,
}

SPS_GENERATION_SETTINGS = {
    "output_prefix": str(PROJECT_GALSBI_DIR / "galsbi_sps_bestfit_1m"),
    "nside": 512,
    "pixel": 0,
    "source_pixels": 308,
    "source_pixel_start": 0,
    "source_pixel_seed": SOURCE_SETTINGS["seed"],
    "random_source_pixels": True,
    "pixel_ordering": "ring",
    "model_index": 0,
    "posterior_path": str(SPS_POSTERIOR_PATH),
    "n_gal_max_blue": None,
    "n_gal_max_red": None,
    "auto_cap_sigma": 10.0,
    "max_expected_rows": 20_000_000,
    "allow_large_density_run": False,
    "quality_size_column": "r50",
    "quality_size_scale": 0.168,
    "summary_chunk_size": 1_000_000,
    "summary_stat_sample_size": 200_000,
    "rebuild_wrapper": False,
    "verbosity": "info",
    "seed": None,
}
