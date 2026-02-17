import os
import numpy as np

# ==============================================================================
# 1. GLOBAL SETTINGS & PATHS
# ==============================================================================
# Compute resources and HEALPix resolution.
SIM_SETTINGS = {
    'sys_nside_sim': 64,     # High-res nside for simulated maps.
    'sys_nside_stats': 64,   # Nside used when grouping patch-level statistics.
    'n_pop_sample': 500_000,  # Number of simulated galaxies per sim pixel.
    'chunk_size': 250_000_000,
    'n_jobs': -1,  # Parallel workers (-1 uses all available).
    'detection_threshold': 0.2, # Detection-probability threshold.
}

# External library paths.
BLENDING_EMULATOR_DIR = '/home/z/Zekang.Zhang/blending_emulator'

# I/O directories.
BASE_DIR = '/home/z/Zekang.Zhang/skyvar/'
PROJECT_DATA_DIR = '/project/ls-gruen/users/zekang.zhang/'

PATHS = {
    'gal_cat': os.path.join(PROJECT_DATA_DIR, "cats/galsbi/f24_0_r_ucat.gal.cat"),
    'mock_sys_map': os.path.join(BASE_DIR, f"data/mock_sys_map_{SIM_SETTINGS['sys_nside_sim']}.fits"),
    'model_json': "/home/z/Zekang.Zhang/optuna_study/models/classification_model_f24_rescaled_neighbor.json",
    'boundary_npy': "/home/z/Zekang.Zhang/optuna_study/models/train_boundary_f24_cla_neighbor.npy",
    'output_preds': os.path.join(PROJECT_DATA_DIR, f"proj2_sims/sys_preds/mock_sys_preds_sim{SIM_SETTINGS['sys_nside_sim']}_full_{SIM_SETTINGS['n_pop_sample']}.feather"),
    'output_fits_template': os.path.join(PROJECT_DATA_DIR, "proj2_sims/sys_preds/sys_{}_{}_nz_bins/{}.fits"),
    'w_theta_fits': os.path.join(PROJECT_DATA_DIR, f"proj2_sims/sys_preds/w_theta_stats{SIM_SETTINGS['sys_nside_stats']}.fits"),
}

# ==============================================================================
# 2. SYSTEMATICS CONFIGURATION (systematics.py)
# ==============================================================================
# Parameters used to generate mock systematics maps.
SYSTEMATICS_CONFIG = {
    'footprint': {
        'ra_range': [140, 240],
        'dec_range': [-6, 6],
    },
    'tiles': {
        'size_deg': 2.0,
    },
    'noise': {
        'mu0': 6.0,        # Global mean pixel noise.
        'sigma_tile': 0.8, # Tile-to-tile scatter.
        'sigma_pix': 0.05   # Small pixel-level jitter.
    },
    'psf': {
        'mu0': 0.7,           # Global baseline (median seeing).
        'sigma_tile': 0.07,   # Scatter in mean PSF across tiles.
        'Abar': -0.08,        # Mean intra-tile amplitude.
        'sigma_A': 0.04,      # Scatter of intra-tile amplitude.
        'sigma_pix': 0.02,    # Small pixel-level noise.
        'xmean_fluc': 0.15,
        'ymean_fluc': 0.15,
        'covxx_mean': 10,
        'covyy_mean': 10,
        'covxx_fluc': 2,
        'covyy_fluc': 2,
        'covxy_fluc': 4,
    },
}

# ==============================================================================
# 3. SELECTION & PHOTO-Z CONFIGURATION (selection.py)
# ==============================================================================
# Baseline input-catalog filters.
CATALOG_SETTINGS = {
    'mag_min': 0,
    'mag_max': 28,
    're_min': 0.01,
    're_max': 5.0,
    'ba_min': 0.05,
    'ba_max': 1.0,
    'sersic_min': 0.5,
    'sersic_max': 6.0,
}

# Nominal observation conditions (baseline values).
OBS_CONDITIONS = {
    'pixel_size': 0.2,
    'zero_point': 30.0,
    'psf_fwhm_nominal': 0.6,
    'moffat_beta': 2.4,
    'pixel_rms_nominal': 6.0,
    'detec_mag_bound': 26.0, # Only brighter galaxies are used as primaries.
}

# Photo-z model and tomographic binning.
PHOTOZ_PARAMS = {
    'sigma_pho': 0.0376,       # Base scatter at m_ref.
    'sigma_int': 0.0212,     # Minimum scatter floor.
    'alpha': 0.4,          # Magnitude-scaling power.
    'm_ref': 21,           # Reference magnitude.
    'rms_ref': 6.0,       # Reference noise level.
    'psf_fwhm_ref': 0.7,   # Reference PSF seeing.
    'maglim0': 4,          # MagLim slope.
    'maglim1': 19,         # MagLim intercept.
    'maglim2': 17,         # Bright-end MagLim cut.
    'snr_min': 0,          # Minimum SNR for lens selection.
}

# General analysis and smoothing settings.
ANALYSIS_SETTINGS = {
    'z_bins': 60,
    'z_min': 0.0,
    'z_max': 2.0,
    'tomo_bin_edges': [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05],
    'smoothing_sigma_dz': 0.05,
    'smooth_nz': False,      # Smooth n(z) distributions.
    'load_preds': False,    # Load cached predictions instead of re-simulating.
}

STATS_PARAMS = {
    'min_count': 100,       # Minimum galaxies per pixel for valid stats.
}

# ==============================================================================
# 4. CLUSTERING & COSMOLOGY CONFIGURATION
# ==============================================================================
COSMO_PARAMS = {
    'Omega_c': 0.25,
    'Omega_b': 0.05,
    'h': 0.67,
    'sigma8': 0.8,
    'n_s': 0.965,
}

CLUSTERING_SETTINGS = {
    'ell_max': 3000,
    'ell_min': 2,
    'theta_min_deg': 0.01,
    'theta_max_deg': 10.0,
    'theta_bins': 30,
}
