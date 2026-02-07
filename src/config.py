import os
import numpy as np

# ==============================================================================
# 1. GLOBAL SETTINGS & PATHS
# ==============================================================================
# Computational resources and Healpix resolution
SIM_SETTINGS = {
    'sys_nside': 96,
    'n_pop_sample': 500_000,
    'chunk_size': 100_000_000,
    'n_jobs': -1,  # Parallel processing threads (-1 uses all available)
    'detection_threshold': 0.2, # Probability threshold for galaxy detection
}

# External library paths
CODE_SRC = '/path/to/external/codes/src'
BLENDING_EMULATOR_DIR = '/path/to/blending_emulator'

# IO Directories
BASE_DIR = './' 
PROJECT_DATA_DIR = '/path/to/project/data/'

PATHS = {
    'gal_cat': "/path/to/catalogue/f24_0_r_ucat.gal.cat",
    'mock_sys_map': os.path.join(BASE_DIR, f"data/mock_sys_map_{SIM_SETTINGS['sys_nside']}.fits"),
    'model_json': "/path/to/models/classification_model.json",
    'boundary_npy': "/path/to/models/train_boundary.npy",
    'output_preds': f"/path/to/output/mock_sys_preds_{SIM_SETTINGS['sys_nside']}_full_{SIM_SETTINGS['n_pop_sample']}.feather",
    'output_fits_template': "/path/to/output/sys_{}_{}_nz_bins/{}.fits",
}

# ==============================================================================
# 2. SYSTEMATICS CONFIGURATION (systematics.py)
# ==============================================================================
# Parameters for generating mock systematic maps (Noise, PSF, Galactic etc.)
SYSTEMATICS_CONFIG = {
    'footprint': {
        'ra_range': [140, 240],
        'dec_range': [-5, 5],
    },
    'tiles': {
        'size_deg': 1.0,
    },
    'noise': {
        'mu0': 6.0,        # global mean pixel noise
        'sigma_tile': 0.8, # tile-to-tile scatter
        'sigma_pix': 0.0   # small pixel-level jitter
    },
    'psf': {
        'mu0': 0.7,           # global baseline (median seeing)
        'sigma_tile': 0.07,   # scatter in mean PSF across tiles
        'Abar': -0.08,        # mean intra-tile amplitude
        'sigma_A': 0.04,      # scatter of amplitude across tiles
        'sigma_pix': 0.00,    # small pixel-level noise
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
# Initial input catalog filters
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

# Nominal observation conditions (used as baseline)
OBS_CONDITIONS = {
    'pixel_size': 0.2,
    'zero_point': 30.0,
    'psf_fwhm_nominal': 0.6,
    'moffat_beta': 2.4,
    'pixel_rms_nominal': 6.0,
    'detec_mag_bound': 27.0,
}

# Photo-z modeling and Tomographic Binning
PHOTOZ_PARAMS = {
    'sigma0': 0.038,       # Base scatter at m_ref
    'sigma_min': 0.02,     # Minimum scatter floor
    'alpha': 0.4,          # Magnitude scaling power
    'm_ref': 21,           # Reference magnitude
    'rms_ref': 12.0,       # Reference noise level
    'psf_fwhm_ref': 0.6,   # Reference PSF seeing
    'maglim0': 4,          # MagLim slope
    'maglim1': 19,       # MagLim intercept
    'maglim2': 16,         # Bright-end MagLim cut
    'snr_min': 0,          # Minimum SNR for len selection
}

# General Analysis & Smoothing
ANALYSIS_SETTINGS = {
    'z_bins': 100,
    'z_min': 0.0,
    'z_max': 2.0,
    'tomo_bin_edges': [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05],
    'smoothing_sigma_dz': 0.1,
    'smooth_nz': True,      # Whether to smooth n(z) distributions
    'load_preds': False,     # If True, load existing feather file instead of re-simulating
}

STATS_PARAMS = {
    'min_count': 100,       # Minimum galaxies per pixel for valid statistics
}

# ==============================================================================
# 4. CLUSTERING & COSMOLOGY CONFIGURATION (clustering.py / variation.py)
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
