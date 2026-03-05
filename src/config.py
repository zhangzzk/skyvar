import os
import numpy as np

# ==============================================================================
# 1. GLOBAL SETTINGS & PATHS
# ==============================================================================
# Compute resources and HEALPix resolution.
SIM_SETTINGS = {
    'sys_nside_sim': 128,     # High-res nside for simulated maps.
    'sys_nside_stats': 128,   # Nside used when grouping pixel-level statistics.
    'n_pop_sample': 10_000,  # Number of simulated galaxies per sim pixel.
    'chunk_size': 250_000_000,
    'n_jobs': -1,  # Parallel workers (-1 uses all available).
    'detection_threshold': 0, # Detection-probability threshold.
    'log_level': 'INFO',       # Logging level: DEBUG, INFO, WARNING, ERROR.
}

# External library paths.
BLENDING_EMULATOR_DIR = '/home/z/Zekang.Zhang/blending_emulator'

# I/O directories.
BASE_DIR = '/home/z/Zekang.Zhang/skyvar/'
PROJECT_DATA_DIR = '/project/ls-gruen/users/zekang.zhang/'

PATHS = {
    'gal_cat': os.path.join(PROJECT_DATA_DIR, "cats/galsbi/f24_0_r_ucat.gal.cat"),
    'model_json': "/home/z/Zekang.Zhang/optuna_study/models/classification_model_f24_rescaled_neighbor.json",
    'boundary_npy': "/home/z/Zekang.Zhang/optuna_study/models/train_boundary_f24_cla_neighbor.npy",
    'data_dir': os.path.join(BASE_DIR, "data"),
    'sys_preds_dir': os.path.join(PROJECT_DATA_DIR, "proj2_sims/sys_preds"),
}

# ==============================================================================
# 2. SYSTEMATICS CONFIGURATION (systematics.py)
# ==============================================================================
# Parameters used to generate mock systematics maps.
TILE_SIZE = 1.0
SYSTEMATICS_CONFIG = {
    'footprint': {
        'ra_range': [155, 233],
        'dec_range': [-9, 9],
    },
    'tiles': {
        'size_deg': TILE_SIZE,
    },
    'noise': {
        'mu0': 6.0,            # Global mean pixel noise.
        'sigma_corr': 0.3,     # Correlated large-scale scatter.  
        'l_corr': 6.0,        # Correlation length [deg].        
        'sigma_uncorr': 0.6,   # Uncorrelated tile-to-tile scatter.  
        'sigma_pix': 0.1,      # Small pixel-level jitter.
    },
    'psf': {
        'mu0': 0.7,            # Global baseline (median seeing).
        'sigma_corr': 0.03,    # Correlated large-scale scatter.  
        'l_corr': 6.0,        # Correlation length [deg].        
        'sigma_uncorr': 0.08,  # Uncorrelated tile-to-tile scatter.  
        'intra_scale': TILE_SIZE,    # Intra-tile Gaussian bump scale [deg].  
        'sigma_pix': 0.01,     # Small pixel-level noise.
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
    'n_arcmin2': None,
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
    'pixel_rms_ref': 6.0, # Alias for plotting.py
    'psf_fwhm_ref': 0.7,   # Reference PSF seeing.
    'maglim0': 4,          # MagLim slope.
    'maglim1': 18.7,         # MagLim intercept.
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
    'post_detection_mode': 'skip_cuts', # 'standard' or 'skip_cuts'
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
    'auto_only': True,
    'flat_global': True,
}

# ==============================================================================
# 5. DENSITY VARIATION CONFIGURATION (density_variation.py)
# ==============================================================================
# TODO: verify all values below — reconstructed from code signatures.
DENSITY_SETTINGS = {
    'external': {
        'tiaogeng_path': '/home/z/Zekang.Zhang/tiaogeng',
        'gls_filename': 'mock_gls.npy',
    },
    'mock': {
        'nz_source': 'toy',
        'gaussian_mean': 0.6,
        'gaussian_sigma': 0.23,
        'glass_nside': 1024,
        'n_arcmin2': 10.0,
        'bias': 1.0,
        'normalize_detection_by_max': True,
        'selection_mode': 'snr',  # 'snr' or 'maglim'
        'snr_min': 0.0,
    },
    'treecorr': {
        'gal_nside': 1024,
        'n_patches': 30,
        'n_threads': 20,
        'nbins': 20,
        'min_sep_arcmin': 5.0,
        'max_sep_arcmin': 250.0,
        'var_method': 'bootstrap',
        'cross_patch_weight': 'geom',
    },
    'catalog': {
        'seed': 42,
    },
    'cache': {
        'reuse_mock_catalogs': True,
    }
}
