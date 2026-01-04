import os
import numpy as np

# --- 0. Project Paths ---
CODE_SRC = '/home/z/Zekang.Zhang/tiaogeng/codes/src'
BLENDING_EMULATOR_DIR = '/home/z/Zekang.Zhang/blending_emulator'

# --- 1. Global Simulation Settings ---
SIM_SETTINGS = {
    'sys_nside': 128,
    'n_pop_sample': 50_000,
    'chunk_size': 10_000_000,
    'n_jobs': 128,  # Parallel processing threads
}

# --- 2. Paths & Directories ---
BASE_DIR = '/home/z/Zekang.Zhang/spatial_variation/'
PROJECT_DATA_DIR = '/project/ls-gruen/users/zekang.zhang/'

PATHS = {
    'gal_cat': os.path.join(PROJECT_DATA_DIR, "cats/galsbi/f24_0_r_ucat.gal.cat"),
    'mock_sys_map': os.path.join(BASE_DIR, f"data/mock_sys_map_{SIM_SETTINGS['sys_nside']}.fits"),
    'model_json': "/home/z/Zekang.Zhang/optuna_study/models/classification_model_f24_rescaled_neighbor.json",
    'boundary_npy': "/home/z/Zekang.Zhang/optuna_study/models/train_boundary_f24_cla_neighbor.npy",
    'output_preds': os.path.join(PROJECT_DATA_DIR, f"proj2_sims/sys_preds/mock_sys_preds_{SIM_SETTINGS['sys_nside']}_full_{SIM_SETTINGS['n_pop_sample']}.feather"),
    'output_fits_template': os.path.join(PROJECT_DATA_DIR, "proj2_sims/sys_preds/sys_{}_{}_nz_magbin{}/{}.fits"),
}

# --- 3. Observation Conditions (for Mock Simulation) ---
OBS_CONDITIONS = {
    'pixel_size': 0.214,
    'zero_point': 30.0,
    'psf_fwhm_nominal': 0.6,
    'moffat_beta': 2.4,
    'pixel_rms_nominal': 6.0,
    'detec_mag_bound': 26.0,
}

# --- 4. Photo-z Model (from math/clustering_enhance.md) ---
PHOTOZ_PARAMS = {
    'b0': 0.0, 'b1': 0.0, 'bm': 0.0, 'bc': 0.0,
    'sigma0': 0.05, 'alpha': 0.1, 
    'm_ref': 24.0, 
    'pixel_rms_ref': 6.0  # Reference noise level
}

# --- 5. Redshift Binning & Analysis ---
ANALYSIS_SETTINGS = {
    'z_max': 2.0,
    'z_bins': 60,
    'tomo_bin_edges': [0.0, 0.5, 1.0, 1.5, 2.0],
    'smoothing_sigma_dz': 0.2,
}

# --- 6. Cosmology ---
COSMO_PARAMS = {
    'Omega_c': 0.25,
    'Omega_b': 0.05,
    'h': 0.67,
    'sigma8': 0.8,
    'n_s': 0.965,
}

# --- 7. Clustering Analysis ---
CLUSTERING_SETTINGS = {
    'ell_max': 2048,
    'ell_min': 2,
    'theta_min_deg': 0.01,
    'theta_max_deg': 10.0,
    'theta_bins': 30,
}

# --- 8. Systematics Map Generation ---
SYSTEMATICS_CONFIG = {
    'footprint': {
        'ra_range': [140, 240],
        'dec_range': [-5, 5],
    },
    'tiles': {
        'size_deg': 1.0,
    },
    'noise': {
        'mu0': 6.0,
        'sigma_tile': 0.5,
        'sigma_pix': 0.05
    },
    'psf': {
        'mu0': 0.75,
        'sigma_tile': 0.07,
        'Abar': -0.08,
        'sigma_A': 0.04,
        'sigma_pix': 0.005,
        'xmean_fluc': 0.15,
        'ymean_fluc': 0.15,
        'covxx_mean': 10,
        'covyy_mean': 10,
        'covxx_fluc': 2,
        'covyy_fluc': 2,
        'covxy_fluc': 4,
    }
}
