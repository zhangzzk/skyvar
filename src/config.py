import os
import numpy as np

# ==============================================================================
# 1. PATHS & I/O
# ==============================================================================
BLENDING_EMULATOR_DIR = '/home/z/Zekang.Zhang/blending_emulator'

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
# 2. SIMULATION (selection.py → simulation stage)
# ==============================================================================
# Controls the Monte-Carlo simulation that populates HEALPix pixels with
# galaxies, runs the XGBoost classifier, and writes the predictions file.
SIM_SETTINGS = {
    # --- HEALPix resolution ---
    'sys_nside_sim': 128,        # Nside at which systematics maps are generated.
    'sys_nside_stats': 128,      # Nside for grouping pixel-level statistics.

    # --- Population ---
    'n_pop_sample': 10_000,      # Galaxies drawn per sim pixel.
    'chunk_size': 250_000_000,   # Max rows per simulation chunk.
    'n_jobs': -1,                # Parallel workers (-1 = all CPUs).

    # --- Detection threshold (shared: selection.py + density_variation.py) ---
    # Galaxies with detection probability ≤ this value are discarded.
    # Used by apply_galaxy_selection() — the single entry-point for both scripts.
    'detection_threshold': 0,

    'log_level': 'INFO',
}


# ==============================================================================
# 3. SYSTEMATICS MAPS (systematics.py)
# ==============================================================================
# Mock observing-condition maps (PSF, noise) over the survey footprint.
TILE_SIZE = 1.0
SYSTEMATICS_CONFIG = {
    'footprint': {
        'ra_range': [155, 233],        # [deg]
        'dec_range': [-9, 9],          # [deg]
    },
    'tiles': {
        'size_deg': TILE_SIZE,
    },
    'noise': {
        'mu0': 6.0,                    # Global mean pixel noise.
        'sigma_corr': 0.3,            # Correlated large-scale scatter.
        'l_corr': 6.0,                # Correlation length [deg].
        'sigma_uncorr': 0.6,          # Uncorrelated tile-to-tile scatter.
        'sigma_pix': 0.1,             # Small pixel-level jitter.
    },
    'psf': {
        'mu0': 0.7,                    # Global baseline (median seeing) [arcsec].
        'sigma_corr': 0.03,           # Correlated large-scale scatter.
        'l_corr': 6.0,                # Correlation length [deg].
        'sigma_uncorr': 0.08,         # Uncorrelated tile-to-tile scatter.
        'intra_scale': TILE_SIZE,     # Intra-tile Gaussian bump scale [deg].
        'sigma_pix': 0.01,            # Small pixel-level noise.
    },
}


# ==============================================================================
# 4. INPUT CATALOG & OBSERVING CONDITIONS (selection.py)
# ==============================================================================
# Baseline quality cuts applied to the galaxy catalog before simulation.
CATALOG_SETTINGS = {
    'mag_min': 0,
    'mag_max': 28,
    're_min': 0.01,
    're_max': 5.0,
    'ba_min': 0.05,
    'ba_max': 1.0,
    'sersic_min': 0.5,
    'sersic_max': 6.0,
    'n_arcmin2': None,       # Override catalog density [gal/arcmin²] (None = auto).
}

# Baseline telescope / observation parameters for the simulation.
OBS_CONDITIONS = {
    'pixel_size': 0.2,           # Detector pixel scale [arcsec/pix].
    'zero_point': 30.0,          # Photometric zero-point [mag].
    'psf_fwhm_nominal': 0.6,    # Nominal PSF FWHM [arcsec].
    'moffat_beta': 2.4,         # Moffat profile β.
    'pixel_rms_nominal': 6.0,   # Nominal pixel RMS noise.
    'detec_mag_bound': 26.0,    # Faint limit for primary galaxies [mag].
}


# ==============================================================================
# 5. GALAXY SELECTION (shared: selection.py + density_variation.py)
# ==============================================================================
# apply_galaxy_selection(df, mode) reads from here.
#   mode = 'snr'    → keeps galaxies with SNR > snr_min.
#   mode = 'maglim' → applies MagLim cut using maglim0/1/2 below.
#   mode = None      → detection threshold only (from SIM_SETTINGS).

# Photo-z scatter model parameters.
#   Also supplies the MagLim and SNR cut thresholds.
PHOTOZ_PARAMS = {
    # --- Photo-z scatter model ---
    'sigma_pho': 0.0376,         # Base photo-z scatter at m_ref.
    'sigma_int': 0.0212,         # Irreducible scatter floor.
    'alpha': 0.4,                # Magnitude-scaling exponent.
    'm_ref': 21,                 # Reference magnitude.
    'rms_ref': 6.0,              # Reference noise level.
    'pixel_rms_ref': 6.0,        # Alias used by plotting.py.
    'psf_fwhm_ref': 0.7,         # Reference PSF seeing [arcsec].

    # --- MagLim selection (mode='maglim') ---
    'maglim0': 4,                # Faint-end slope:  r_obs < maglim0 * z_pho + maglim1.
    'maglim1': 18.7,             # Faint-end intercept.
    'maglim2': 17,               # Bright-end magnitude cut.

    # --- SNR selection (mode='snr') ---
    'snr_min': 0,                # Minimum SNR for lens selection.
}


# ==============================================================================
# 6. ANALYSIS (selection.py → summary statistics stage)
# ==============================================================================
# Controls what happens *after* the predictions are loaded / generated:
# galaxy selection → obs-property computation → tomo binning → n(z) summaries.
ANALYSIS_SETTINGS = {
    # --- Redshift grid (shared with density_variation.py) ---
    'z_bins': 60,
    'z_min': 0.0,
    'z_max': 2.0,

    # --- Tomographic binning ---
    'tomo_bin_edges': [0.2, 0.4, 0.55, 0.7, 0.85, 0.95, 1.05],

    # --- n(z) smoothing ---
    'smoothing_sigma_dz': 0.05,
    'smooth_nz': False,

    # --- Workflow control ---
    'load_preds': False,         # True = load cached predictions, False = re-simulate.
    # Selection mode for selection.py — passed directly to apply_galaxy_selection(mode=...).
    #   'snr'    → SNR cut (snr_min from §5)
    #   'maglim' → MagLim cut (maglim0/1/2 from §5)
    #   None     → detection threshold only
    'selection_mode': None,
}

STATS_PARAMS = {
    'min_count': 100,            # Minimum galaxies per pixel for valid statistics.
}


# ==============================================================================
# 7. COSMOLOGY & THEORY (clustering.py, density_variation.py)
# ==============================================================================
COSMO_PARAMS = {
    'Omega_c': 0.25,
    'Omega_b': 0.05,
    'h': 0.67,
    'sigma8': 0.8,
    'n_s': 0.965,
}

# Angular power spectrum / correlation function settings.
#   Used by clustering.py (enhancement) and density_variation.py (theory curve).
CLUSTERING_SETTINGS = {
    # --- Harmonic-space range ---
    'ell_min': 2,
    'ell_max': 3000,

    # --- Real-space binning (clustering.py enhancement) ---
    'theta_min_deg': 0.01,
    'theta_max_deg': 10.0,
    'theta_bins': 30,

    'auto_only': True,
    'flat_global': True,

    # --- TreeCorr binning (density_variation.py NN correlation) ---
    'nbins': 20,
    'min_sep_arcmin': 5.0,
    'max_sep_arcmin': 250.0,
}


# ==============================================================================
# 8. DENSITY VARIATION (density_variation.py)
# ==============================================================================
# End-to-end w(θ) measurement: GLASS mock catalogs + selection map + TreeCorr.
DENSITY_SETTINGS = {
    # --- External code paths ---
    'external': {
        'tiaogeng_path': '/home/z/Zekang.Zhang/tiaogeng',
        'gls_filename': 'mock_gls.npy',
    },

    # --- Mock galaxy catalog generation ---
    'mock': {
        # n(z) source for GLASS mock:  'toy' = Gaussian, 'measured' = from predictions.
        'nz_source': 'toy',

        # Toy Gaussian n(z) parameters (only used when nz_source='toy').
        'gaussian_mean': 0.6,
        'gaussian_sigma': 0.23,

        # GLASS grid & population.
        'glass_nside': 1024,
        'glass_z_nbins': 201,          # z-grid points for GLASS (finer than z_bins).
        'n_arcmin2': 5.0,              # Galaxy density for mock catalog.
        'bias': 1.0,                   # Galaxy bias (also used for theory curve).

        # Detection map post-processing.
        'normalize_detection_by_max': True,

        # Selection mode for _compute_measured_nz() when nz_source='measured'.
        # Uses apply_galaxy_selection(mode=...) — see §5.
        'selection_mode': 'snr',       # 'snr', 'maglim', or 'none'
    },

    # --- TreeCorr NN correlation ---
    'treecorr': {
        'gal_nside': 1024,             # HEALPix nside for galaxy pixelization.
        'n_patches': 30,               # Bootstrap / jackknife patch count.
        'n_threads': 20,
        'var_method': 'bootstrap',
        'cross_patch_weight': 'geom',
    },

    # --- Caching ---
    'cache': {
        'reuse_mock_catalogs': True,   # False = regenerate even if files exist.
    },
}
