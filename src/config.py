import os
import numpy as np

# --- 0. Project Paths ---
CODE_SRC = '/home/z/Zekang.Zhang/tiaogeng/codes/src'
BLENDING_EMULATOR_DIR = '/home/z/Zekang.Zhang/blending_emulator'

# --- 1. Global Simulation Settings ---
SIM_SETTINGS = {
    'sys_nside': 32,
    'n_pop_sample': 2_000_000,
    'chunk_size': 100_000_000,
    'n_jobs': -1,  # Parallel processing threads
    'detection_threshold': 0.5,
}

# --- 2. Paths & Directories ---
BASE_DIR = '/home/z/Zekang.Zhang/nz_variation/'
PROJECT_DATA_DIR = '/project/ls-gruen/users/zekang.zhang/'

PATHS = {
    'gal_cat': os.path.join(PROJECT_DATA_DIR, "cats/galsbi/f24_0_r_ucat.gal.cat"),
    'mock_sys_map': os.path.join(BASE_DIR, f"data/mock_sys_map_{SIM_SETTINGS['sys_nside']}.fits"),
    'model_json': "/home/z/Zekang.Zhang/optuna_study/models/classification_model_f24_rescaled_neighbor.json",
    'boundary_npy': "/home/z/Zekang.Zhang/optuna_study/models/train_boundary_f24_cla_neighbor.npy",
    'output_preds': os.path.join(PROJECT_DATA_DIR, f"proj2_sims/sys_preds/mock_sys_preds_{SIM_SETTINGS['sys_nside']}_full_{SIM_SETTINGS['n_pop_sample']}.feather"),
    'output_fits_template': os.path.join(PROJECT_DATA_DIR, "proj2_sims/sys_preds/sys_{}_{}_nz_bins/{}.fits"),
}

# --- 3. Observation Conditions (for Mock Simulation) ---
# This does not really matter
OBS_CONDITIONS = {
    'pixel_size': 0.2,
    'zero_point': 30.0,
    'psf_fwhm_nominal': 0.6,
    'moffat_beta': 2.4,
    'pixel_rms_nominal': 6.0,
    'detec_mag_bound': 26.0,
}

# --- 4. Photo-z Model (from math/clustering_enhance.md) ---
PHOTOZ_PARAMS = {
    'sigma0': 0.0376,
    'sigma_min': 0,
    'alpha': 0.4,
    'm_ref': 22,
    'rms_ref': 6.0,
    'psf_fwhm_ref': 0.6,
    
    # 'sigma_int_ref': 0.0212 , 'sigma_pho_ref': (0.0376**2-0.0212**2)**0.5, 
    # 'snr_ref': 70, 
    'maglim0': 4, 'maglim1': 19.5, 'maglim2': 16, 'snr_min': 0,
    
    # Systematic Bias (z_pho shift ∝ 1/SNR)
    # 'bias_base': 0.0,
    # 'bias_snr_slope': -0.20,
}

STATS_PARAMS = {
    'min_count': 100,
}


# --- 5. Redshift Binning & Analysis ---
ANALYSIS_SETTINGS = {
    'z_bins': 60,
    'z_min': 0.0,
    'z_max': 2.0,
    'tomo_bin_edges': [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05],
    'smoothing_sigma_dz': 0.1,
    'smooth_nz': False,
    'load_preds': True,
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
    'ell_max': 3000,
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
    'mu0': 6.0,        # global mean pixel noise
    'sigma_tile': 0.8, # tile-to-tile scatter
    'sigma_pix': 0.0  # small pixel-level jitter
    },

    'psf': {
    # global baseline (median seeing)
    'mu0': 0.7,

    # inter-tile fluctuations
    'sigma_tile': 0.07,   # scatter in mean PSF across tiles

    # amplitude distribution
    'Abar': -0.08,        # mean intra-tile amplitude
    'sigma_A': 0.04,      # scatter of amplitude across tiles

    # small pixel-level noise
    'sigma_pix': 0.00,

    # Gaussian shape inside tiles
    'xmean_fluc': 0.15,
    'ymean_fluc': 0.15,
    'covxx_mean': 10,
    'covyy_mean': 10,
    'covxx_fluc': 2,
    'covyy_fluc': 2,
    'covxy_fluc': 4,
},
}

# --- 9. Catalog Selection Settings ---
CATALOG_SETTINGS = {
    'mag_min': 0,
    'mag_max': 25,
    're_min': 0.01,
    're_max': 5.0,
    'ba_min': 0.05,
    'ba_max': 1.0,
    'sersic_min': 0.5,
    'sersic_max': 6.0,
    'cat_area': 5.97, # sq. degree
}
