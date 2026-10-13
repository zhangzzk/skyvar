from pathlib import Path


# ==============================================================================
# 1. PATHS & I/O
# ==============================================================================
_SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = _SRC_DIR.parent
PROJECT_DATA_DIR = Path("/project/ls-gruen/users/zekang.zhang")
BLENDEMU_DIR = "/home/z/Zekang.Zhang/blending_emulator"
TIAOGENG_DIR = "/home/z/Zekang.Zhang/tiaogeng"

PATHS = {
    "gal_cat": str(PROJECT_DATA_DIR / "cats/galsbi/f24_0_r_ucat.gal.cat"),
    "model_json": "/home/z/Zekang.Zhang/optuna_study/models/classification_model_f24_rescaled_neighbor.json",
    "boundary_npy": "/home/z/Zekang.Zhang/optuna_study/models/train_boundary_f24_cla_neighbor.npy",
    "data_dir": str(BASE_DIR / "data"),
    "sys_preds_dir": str(PROJECT_DATA_DIR / "proj2_sims/sys_preds"),
}


# ==============================================================================
# 2. SIMULATION (selection.py -> simulation stage)
# ==============================================================================
SIM_SETTINGS = {
    # HEALPix resolution.
    "sys_nside_sim": 1024,
    "sys_nside_stats": 256,

    # Population.
    "n_pop_sample": 50_000,
    "chunk_size": 500_000_000,
    "n_jobs": -1,

    # Detection threshold shared by selection.py and density_variation.py.
    "detection_threshold": 0.1,

    "log_level": "INFO",
}


# ==============================================================================
# 3. SYSTEMATICS MAPS (systematics.py)
# ==============================================================================
TILE_SIZE = 1.0
SYSTEMATICS_CONFIG = {
    "footprint": {
        "ra_range": [150, 220],
        "dec_range": [-9, 9],
    },
    "tiles": {
        "size_deg": TILE_SIZE,
    },
    "noise": {
        "mu0": 6.0,
        "sigma_corr": 0.3,
        "l_corr": 4.0,
        "sigma_uncorr": 0.5,
        "sigma_pix": 0.1,
    },
    "psf": {
        "mu0": 0.65,
        "sigma_corr": 0.0,
        "l_corr": 4.0,
        "sigma_uncorr": 0.07,
        "intra_scale": TILE_SIZE,
        "sigma_pix": 0.01,
    },
}


# ==============================================================================
# 4. INPUT CATALOG & OBSERVING CONDITIONS (selection.py)
# ==============================================================================
CATALOG_SETTINGS = {
    "mag_min": 0,
    "mag_max": 28,
    "re_min": 0.01,
    "re_max": 5.0,
    "ba_min": 0.05,
    "ba_max": 1.0,
    "sersic_min": 0.5,
    "sersic_max": 6.0,
    "n_arcmin2": None,
}

OBS_CONDITIONS = {
    "pixel_size": 0.2,
    "zero_point": 30.0,
    "psf_fwhm_nominal": 0.6,
    "moffat_beta": 2.4,
    "pixel_rms_nominal": 6.0,
    "detec_mag_bound": 26.0,
}


# ==============================================================================
# 5. GALAXY SELECTION (shared: selection.py + density_variation.py)
# ==============================================================================
PHOTOZ_PARAMS = {
    "sigma_pho": 0.0376,
    "sigma_int": 0.0212,
    "alpha": 0.4,
    "m_ref": 21,
    "rms_ref": 6.0,
    "psf_fwhm_ref": 0.7,

    # MagLim selection.
    "maglim0": 4,
    "maglim1": 18.7,
    "maglim2": 17,

    # SNR selection.
    "snr_min": 5,
}


# ==============================================================================
# 6. ANALYSIS (selection.py -> summary statistics stage)
# ==============================================================================
ANALYSIS_SETTINGS = {
    "z_bins": 60,
    "z_min": 0.0,
    "z_max": 2.0,

    "tomo_bin_edges": [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05],

    "smoothing_sigma_dz": 0.05,
}

STATS_PARAMS = {
    "min_count": 100,
}


# ==============================================================================
# 7. COSMOLOGY & THEORY (clustering.py, density_variation.py)
# ==============================================================================
COSMO_PARAMS = {
    "Omega_c": 0.2645,
    "Omega_b": 0.0493,
    "h": 0.6736,
    "sigma8": 0.8111,
    "n_s": 0.9649,
}

CLUSTERING_SETTINGS = {
    "ell_min": 2,
    "ell_max": 3000,

    "theta_min_deg": 0.01,
    "theta_max_deg": 10.0,
    "theta_bins": 30,

    "auto_only": True,
    "flat_global": False,
    # Also compute the harmonic-space (C_ell) enhancement in nz_variation.
    "compute_cl": True,

    "nbins": 20,
    "min_sep_arcmin": 5.0,
    "max_sep_arcmin": 250.0,
}


# ==============================================================================
# 8. DENSITY VARIATION (density_variation.py)
# ==============================================================================
DENSITY_SETTINGS = {
    "external": {
        "tiaogeng_path": TIAOGENG_DIR,
        "gls_filename": "mock_gls.npy",
    },

    "mock": {
        "gaussian_mean": 0.6,
        "gaussian_sigma": 0.25,
        "glass_nside": 1024,
        "glass_z_nbins": 201,
        "n_arcmin2": 10.0,
        "bias": 1.0,
        "normalize_detection_by_max": True,
    },

    "treecorr": {
        "gal_nside": 1024,
        "n_patches": 30,
        "n_threads": 20,
        "var_method": "bootstrap",
        "cross_patch_weight": "geom",
    },

    "cache": {
        "reuse_mock_catalogs": False,
    },
}
