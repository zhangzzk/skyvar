import os
os.environ["NUMEXPR_MAX_THREADS"] = "128"
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
import xgboost as xgb
import gc
import psutil

def get_memory_usage():
    """Return memory usage of current process in GB."""
    return psutil.Process().memory_info().rss / 1e9

try:
    from . import utils
    from . import config
except ImportError:
    import utils
    import config

# Add paths to sys.path
CODE_SRC = config.CODE_SRC
BLENDING_EMULATOR_DIR = config.BLENDING_EMULATOR_DIR
for path in [CODE_SRC, BLENDING_EMULATOR_DIR]:
    if path not in sys.path:
        sys.path.append(path)

# Custom imports after sys.path update
try:
    import glass_mock
    import generate_mocksys
    import nz_utils
    from cosmic_toolbox import arraytools as at
except ImportError as e:
    print(f"Warning: Could not import some custom modules: {e}")

# Constants from config
SYS_NSIDE = config.SIM_SETTINGS['sys_nside']
N_POP_SAMPLE = config.SIM_SETTINGS['n_pop_sample']
CHUNK_SIZE = config.SIM_SETTINGS['chunk_size']
N_JOBS = config.SIM_SETTINGS['n_jobs']
GAL_CAT_PATH = config.PATHS['gal_cat']
MOCK_SYS_MAP_PATH = config.PATHS['mock_sys_map']
MODEL_JSON = config.PATHS['model_json']
OUTPUT_PREDS = config.PATHS['output_preds']
OUTPUT_FITS_TEMPLATE = config.PATHS['output_fits_template']


def groupby_dndz(sys_cat, z, post_cut=None, weight_col=None):
    """Compute per-pixel normalized n(z) and sum_num using vectorized operations."""
    z = np.asarray(z)
    if z.size < 2:
        raise ValueError("Need at least two z centers to define bin edges.")

    edges = np.empty(z.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    dz = np.diff(edges)

    # Filter and Prepare Weights
    if post_cut is not None:
        df_cut = sys_cat.loc[post_cut(sys_cat)].copy()
    else:
        df_cut = sys_cat.copy()

    if weight_col is None:
        df_cut["_w"] = df_cut["detection"]
    else:
        df_cut["_w"] = df_cut["detection"] * df_cut[weight_col]

    # Vectorized 2D Histogram
    pix_idx = df_cut["pix_idx_input_p"].values
    z_vals = df_cut["redshift_input_p"].values
    z_bins = np.digitize(z_vals, edges) - 1
    
    n_pix = int(sys_cat["pix_idx_input_p"].max() + 1)
    n_z = len(z)
    mask_valid = (z_bins >= 0) & (z_bins < n_z) & (pix_idx >= 0) & (pix_idx < n_pix)
    
    flat_idx = pix_idx[mask_valid] * n_z + z_bins[mask_valid]
    counts_flat = np.bincount(flat_idx, weights=df_cut["_w"].values[mask_valid], minlength=n_pix * n_z)
    pixel_counts = counts_flat.reshape(n_pix, n_z)
    
    sum_num = pixel_counts.sum(axis=1)
    hist_raw = np.zeros_like(pixel_counts)
    active_pix = sum_num > 0
    hist_raw[active_pix] = pixel_counts[active_pix] / (sum_num[active_pix][:, None] * dz)
    
    out = pd.DataFrame(hist_raw, index=np.arange(n_pix))
    out["sum_num"] = sum_num

    dndz_in = np.histogram(sys_cat["redshift_input_p"], bins=edges, density=True)[0]
    num_in = sys_cat.shape[0]

    dndz_det = np.histogram(df_cut["redshift_input_p"], bins=edges, density=True, weights=df_cut["_w"])[0]
    num_det = df_cut["_w"].sum()

    out.loc["total_input"] = list(dndz_in) + [num_in]
    out.loc["total_detected"] = list(dndz_det) + [num_det]
    
    return out


def smooth_nz_preserve_moments(z, nz, sigma_dz=0.05):
    """Vectorized version of moment-preserving smoothing."""
    z = np.asarray(z)
    nz = np.asarray(nz).astype(float)
    is_1d = nz.ndim == 1
    if is_1d:
        nz = nz[np.newaxis, :]
        
    dz = np.diff(z).mean() if z.size > 1 else 1.0
    sigma_bins = sigma_dz / dz
    
    f = gaussian_filter1d(nz, sigma=sigma_bins, axis=1, mode="nearest")
    f = np.clip(f, 0, None)

    I0 = np.trapezoid(nz, z, axis=1)
    mask0 = I0 > 0
    mu0 = np.zeros(nz.shape[0])
    var0 = np.zeros(nz.shape[0])
    mu0[mask0] = np.trapezoid(z * nz[mask0], z, axis=1) / I0[mask0]
    var0[mask0] = np.trapezoid((z - mu0[mask0, None])**2 * nz[mask0], z, axis=1) / I0[mask0]

    If = np.trapezoid(f, z, axis=1)
    maskf = (If > 0)
    muf = np.zeros(nz.shape[0])
    varf = np.zeros(nz.shape[0])
    m4 = np.zeros(nz.shape[0])
    
    muf[maskf] = np.trapezoid(z * f[maskf], z, axis=1) / If[maskf]
    varf[maskf] = np.trapezoid((z - muf[maskf, None])**2 * f[maskf], z, axis=1) / If[maskf]
    m4[maskf] = np.trapezoid((z - muf[maskf, None])**4 * f[maskf], z, axis=1) / If[maskf]

    mask_corr = maskf & (varf > 0) & (m4 > 0) & mask0
    
    a = np.zeros(nz.shape[0])
    b = np.zeros(nz.shape[0])
    a[mask_corr] = (mu0[mask_corr] - muf[mask_corr]) / varf[mask_corr]
    b[mask_corr] = (var0[mask_corr] - varf[mask_corr]) / (2 * m4[mask_corr])

    g = f.copy()
    z_shifted = z - muf[:, None]
    poly = (1 + a[:, None] * z_shifted + b[:, None] * (z_shifted**2 - varf[:, None]))
    g[mask_corr] *= poly[mask_corr]
    g = np.clip(g, 0, None)

    Ig = np.trapezoid(g, z, axis=1)
    mask_g = Ig > 0
    g[mask_g] *= (I0[mask_g] / Ig[mask_g])[:, None]

    return g[0] if is_1d else g


def load_and_filter_catalog():
    """Load and process the input galaxy catalog."""
    print(f"Loading catalog from {GAL_CAT_PATH}...")
    gal_cat = at.rec2pd(at.load_hdf(GAL_CAT_PATH))
    gal_cat = gal_cat[['sersic_n', 'int_mag', 'int_r50_arcsec', 'z', 'e1', 'e2']].rename(columns={
        'int_mag': 'r',
        'int_r50_arcsec': 'Re',
        'z': 'redshift',
    })

    BA, angle = utils.e1e2_to_q_phi(gal_cat['e1'], gal_cat['e2'])
    gal_cat['BA'] = BA
    gal_cat['angle'] = angle / np.pi * 180 + 90
    gal_cat['Re'] /= np.sqrt(gal_cat['BA'])

    gal_cat = gal_cat.loc[
        (gal_cat['BA'] > 0.05) & (gal_cat['BA'] < 1.0) &
        (gal_cat['r'] < 28) & (gal_cat['r'] > 0) &
        (gal_cat['Re'] < 5.) & (gal_cat['Re'] > 0.01) &
        (gal_cat['sersic_n'] < 6.) & (gal_cat['sersic_n'] > 0.5)
    ].reset_index(drop=True).astype(np.float64)

    cat_area = 5.968310365946076
    n_degree2 = gal_cat.shape[0] / cat_area
    print(f"Number density: {n_degree2 / 60**2:.2f} gal/arcmin^2")
    return gal_cat, n_degree2


def load_system_maps():
    """Load system maps and return maps and SEEN_idx."""
    print(f"Loading system maps from {MOCK_SYS_MAP_PATH}...")
    maps = hp.read_map(MOCK_SYS_MAP_PATH, field=None)
    psf_hp_map = maps[0]
    SEEN_idx = np.where(~np.isnan(psf_hp_map))[0]
    return maps, SEEN_idx


def galaxy_snr_from_mag_size(mag, r_half, seeing_fwhm, sigma_pix, zeropoint=30.0, pixscale=0.2):
    """Approximate galaxy SNR."""
    flux = 10.0 ** (-0.4 * (mag - zeropoint))
    sigma_gal = r_half / 1.678
    sigma_psf = seeing_fwhm / 2.355
    sigma_eff2 = sigma_gal**2 + sigma_psf**2
    n_eff = 4.0 * np.pi * sigma_eff2 / pixscale**2
    snr = flux / (sigma_pix * np.sqrt(n_eff))
    return snr


def process_one(i, idx, icat, conditions, gal_num, psf_hp_map, noise_hp_map, galactic_hp_map, detec_mag_bound):
    """Worker function for parallel processing."""
    rng_local = np.random.default_rng(i)
    randi = rng_local.integers(0, icat.shape[0], size=gal_num)
    subset = icat.iloc[randi].copy()
    subset['pix_idx'] = idx

    conds = conditions.copy()
    conds['psf_fwhm'] = psf_hp_map[idx]
    conds['pixel_rms'] = noise_hp_map[idx]
    
    subset.loc[:, 'r'] += galactic_hp_map[idx]
    subset.loc[subset['r'] > detec_mag_bound, 'r'] = detec_mag_bound

    for key, value in conds.items():
        subset[key] = value

    subset['snr'] = galaxy_snr_from_mag_size(
        subset['r'],
        subset['Re'],
        subset['psf_fwhm'],
        subset['pixel_rms'],
        zeropoint=subset['zero_point'],
        pixscale=subset['pixel_size'],
    )
    return subset


def simulate_and_classify_chunked(gal_cat, n_degree2):
    """
    Memory-efficient simulation and classification. 
    Filters non-detections immediately to save 90%+ memory.
    """
    import gc
    maps, SEEN_idx = load_system_maps()
    psf_hp_map, noise_hp_map, galactic_hp_map = maps
    
    conditions = {
        "pixel_size": config.OBS_CONDITIONS['pixel_size'],
        "zero_point": config.OBS_CONDITIONS['zero_point'],
        "psf_fwhm": config.OBS_CONDITIONS['psf_fwhm_nominal'],
        "moffat_beta": config.OBS_CONDITIONS['moffat_beta'],
        "pixel_rms": config.OBS_CONDITIONS['pixel_rms_nominal'],
    }
    detec_mag_bound = config.OBS_CONDITIONS['detec_mag_bound']
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    z_max = config.ANALYSIS_SETTINGS['z_max']
    z_bins_n = config.ANALYSIS_SETTINGS['z_bins']
    z = np.linspace(0, z_max, z_bins_n)
    edges = np.empty(z.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])

    print("Loading XGBoost model...")
    bst_cla = xgb.Booster({'device': 'cuda', 'n_jobs': -1})
    bst_cla.load_model(MODEL_JSON)
    
    temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Global Input Statistics Accumulators
    global_hist_in = np.zeros(z_bins_n)
    global_num_in = 0
    
    pixels_per_chunk = max(1, CHUNK_SIZE // N_POP_SAMPLE)
    n_pixels = len(SEEN_idx)
    chunk_files = []
    
    print(f"Starting chunked processing: {n_pixels} pixels in groups of {pixels_per_chunk}")
    
    for start_p in range(0, n_pixels, pixels_per_chunk):
        end_p = min(start_p + pixels_per_chunk, n_pixels)
        block_indices = SEEN_idx[start_p:end_p]
        
        print(f"  Chunk {start_p//pixels_per_chunk + 1}/{(n_pixels-1)//pixels_per_chunk + 1} ({len(block_indices)} pixels)")
        
        # 1. Simulate this block
        results = Parallel(n_jobs=N_JOBS, backend="threading")(
            delayed(process_one)(start_p + i, idx, gal_cat, conditions, N_POP_SAMPLE,
                                 psf_hp_map, noise_hp_map, galactic_hp_map, detec_mag_bound)
            for i, idx in enumerate(block_indices)
        )
        block_fullset = pd.concat(results, ignore_index=True)
        results = None 
        
        # Accumulate input statistics before classification/filtering
        h_in = np.histogram(block_fullset["redshift"], bins=edges)[0]
        global_hist_in += h_in
        global_num_in += len(block_fullset)
        
        # 2. Add spatial coordinates (required by icat2cla)
        block_tot_num = block_fullset.shape[0]
        area_sq = block_tot_num / n_degree2
        side = np.sqrt(area_sq)
        block_fullset["RA"] = np.random.uniform(0, side, size=block_tot_num)
        block_fullset["DEC"] = np.random.uniform(0, side, size=block_tot_num)

        # 3. Classify and Filter
        try:
            block_cla = nz_utils.icat2cla_v2(block_fullset, block_fullset, bst_cla, predict=True)
            # CRITICAL: Keep only detections to save memory
            block_cla = block_cla[block_cla['detection'] > 0.5].copy()
            
            if not block_cla.empty:
                # Add Photo-z weights for detected objects early
                p_weights = utils.get_photoz_weights(block_cla, tomo_bin_edges)
                for i in range(len(tomo_bin_edges) - 1):
                    block_cla[f"tomo_p_{i}"] = p_weights[:, i]
                
                temp_path = os.path.join(temp_dir, f"cla_chunk_det_{start_p}.feather")
                block_cla.to_feather(temp_path)
                chunk_files.append(temp_path)
        except Exception as e:
            print(f"Error processing block {start_p}: {e}")
        
        block_fullset = None
        block_cla = None
        gc.collect()
        print(f"    Memory usage: {get_memory_usage():.2f} GB")

    # Pre-calculated density and num
    dz = np.diff(edges)
    dndz_in_total = global_hist_in / (global_num_in * dz) if global_num_in > 0 else global_hist_in
    
    # Pack global simulation truth
    sim_truth = {
        'dndz_in': dndz_in_total,
        'num_in': global_num_in,
        'edges': edges,
        'dz': dz,
        'z': z
    }
        
    return chunk_files, psf_hp_map, SEEN_idx, sim_truth


def generate_summary_statistics_from_cat(cla_cat, psf_hp_map, SEEN_idx, output_dir):
    """Compute detection maps and dN/dz distributions from a catalog."""
    mean_p = np.full(psf_hp_map.shape, hp.UNSEEN)
    
    # Map pixels to active indices to avoid out-of-bounds
    pixel_counts = np.bincount(cla_cat['pix_idx_input_p'], minlength=len(psf_hp_map))
    mean_p[SEEN_idx] = pixel_counts[SEEN_idx] / N_POP_SAMPLE
    
    utils.plt_map(mean_p, SYS_NSIDE, np.where(~np.isnan(psf_hp_map)), 
            save_path=os.path.join(output_dir, "detection_rate_map.png"))
    
    z_max = config.ANALYSIS_SETTINGS['z_max']
    z_bins_n = config.ANALYSIS_SETTINGS['z_bins']
    z = np.linspace(0, z_max, z_bins_n)
    
    # Reconstruction of sim_truth based on global expectation if loading from file
    # (Since loading file loses the exact input sample used, we assume uniform)
    # Or we can just calculate from redshift_input_p which is still in the file
    edges = np.empty(z.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = z[0] - 0.5 * (z[1] - z[0])
    edges[-1] = z[-1] + 0.5 * (z[-1] - z[-2])
    dz = np.diff(edges)
    
    global_hist_in = np.histogram(cla_cat["redshift_input_p"], bins=edges)[0]
    # NOTE: This 'num_in' is not quite right because cla_cat only has detections
    # But for dndz_in shape it's fine.
    dndz_in = global_hist_in / (global_hist_in.sum() * dz)
    
    results = {}
    print("Calculating dN/dz for full sample...")
    # Clean catalogs often have indices beyond current nside range if just switched
    mask_in_footprint = (cla_cat['pix_idx_input_p'] < len(psf_hp_map))
    df_filtered = cla_cat[mask_in_footprint].copy()
    
    sys_res_full = groupby_dndz(df_filtered, z, post_cut=None)
    metadata_rows = sys_res_full.loc[["total_input", "total_detected"]].copy()
    
    # Re-index to SEEN_idx to ensure shape matches later correlation calls
    sys_res_data = sys_res_full.reindex(SEEN_idx).fillna(0)
    # Add metadata back
    sys_res_final = pd.concat([sys_res_data, metadata_rows])
    
    # Overwrite the total_input with proper normalized shape from global reconstruction
    sys_res_final.loc["total_input", sys_res_final.columns[:-1]] = dndz_in
    # Roughly guess num_in if missing (assume 10% detection rate for scale)
    sys_res_final.loc["total_input", "sum_num"] = cla_cat.shape[0] / 0.1
    
    results['full'] = process_stats(sys_res_final, z, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
    
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    for i in range(len(tomo_bin_edges)-1):
        if f"tomo_p_{i}" in cla_cat.columns:
            print(f"Calculating dN/dz for tomo_{i}...")
            sys_res_i = groupby_dndz(df_filtered, z, post_cut=None, weight_col=f"tomo_p_{i}")
            meta_i = sys_res_i.loc[["total_input", "total_detected"]].copy()
            sys_res_i_data = sys_res_i.reindex(SEEN_idx).fillna(0)
            sys_res_i_final = pd.concat([sys_res_i_data, meta_i])
            
            # Use same shared input distribution shape
            sys_res_i_final.loc["total_input"] = sys_res_final.loc["total_input"]
            results[f'tomo_{i}'] = process_stats(sys_res_i_final, z, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
            
    return results


def process_stats(sys_res, z, SEEN_idx, smooth=False):
    """Auxiliary to package dndz results."""
    dndzs = sys_res.drop(["sum_num"], axis=1)
    if "total_input" in dndzs.index:
        dndzs = dndzs.drop(["total_input", "total_detected"])
    
    dndzs = dndzs.to_numpy()
    sum_num = sys_res["sum_num"].drop(["total_input", "total_detected"]).values
    
    mean_num = sys_res.loc["total_detected", "sum_num"] / len(SEEN_idx)
    frac_pix = (sum_num / mean_num) if mean_num > 0 else np.ones_like(sum_num)
    frac = sys_res.loc["total_detected", "sum_num"] / sys_res.loc["total_input", "sum_num"]
    
    dndz_in = sys_res.drop(["sum_num"], axis=1).loc["total_input"].to_numpy().astype(float)
    dndz_det = sys_res.drop(["sum_num"], axis=1).loc["total_detected"].to_numpy().astype(float)

    if smooth:
        print(f"Smoothing {dndzs.shape[0]} distributions...")
        sigma_dz = config.ANALYSIS_SETTINGS['smoothing_sigma_dz']
        sm_dndzs = smooth_nz_preserve_moments(z, dndzs, sigma_dz=sigma_dz)
        sm_dndz_in = smooth_nz_preserve_moments(z, dndz_in, sigma_dz=sigma_dz)
        sm_dndz_det = smooth_nz_preserve_moments(z, dndz_det, sigma_dz=sigma_dz)
    
        return {
            'z': z, 'dndzs': dndzs, 'dndz_in': dndz_in, 'dndz_det': dndz_det,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'sm_dndzs': sm_dndzs, 'sm_dndz_in': sm_dndz_in, 'sm_dndz_det': sm_dndz_det
        }
    else:
        return {
            'z': z, 'dndzs': dndzs, 'dndz_in': dndz_in, 'dndz_det': dndz_det,
            'frac': frac, 'frac_pix': frac_pix, 'SEEN_idx': SEEN_idx,
            'sm_dndzs': dndzs, 'sm_dndz_in': dndz_in, 'sm_dndz_det': dndz_det
        }


def generate_summary_statistics_incremental(chunk_files, sim_truth, psf_hp_map, SEEN_idx, output_dir):
    """Memory-efficient incremental statistics generation."""
    import gc
    z = sim_truth['z']
    edges = sim_truth['edges']
    dz = sim_truth['dz']
    n_z = z.size
    n_pix_total = hp.nside2npix(SYS_NSIDE)
    n_pix_active = len(SEEN_idx)
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    n_tomo = len(tomo_bin_edges) - 1

    # Accumulators
    det_counts_map = np.zeros(n_pix_total)
    
    # We use dictionaries to store histograms for 'full' and each tomo bin
    # Key 'full' plus tomo_0, tomo_1...
    keys = ['full'] + [f'tomo_{i}' for i in range(n_tomo)]
    accumulators = {k: np.zeros((n_pix_total, n_z)) for k in keys}
    total_det_nums = {k: 0.0 for k in keys}
    total_det_hists = {k: np.zeros(n_z) for k in keys}

    print("Incremental accumulation from chunks...")
    for f in chunk_files:
        df = pd.read_feather(f)
        if df.empty: continue
        
        pixel_indices = df["pix_idx_input_p"].values
        z_vals = df["redshift_input_p"].values
        z_bins = np.digitize(z_vals, edges) - 1
        mask_z = (z_bins >= 0) & (z_bins < n_z)
        
        # Detection rate map (using detected count only, denominator is constant N_POP_SAMPLE)
        pixel_counts = np.bincount(pixel_indices, minlength=n_pix_total)
        det_counts_map += pixel_counts
        
        for k in keys:
            if k == 'full':
                weights = np.ones(len(df))
            else:
                idx = int(k.split('_')[1])
                weights = df[f'tomo_p_{idx}'].values
                
            total_det_nums[k] += weights.sum()
            total_det_hists[k] += np.histogram(z_vals, bins=edges, weights=weights)[0]
            
            # Per-pixel histogram accumulation
            flat_idx = pixel_indices[mask_z] * n_z + z_bins[mask_z]
            counts_flat = np.bincount(flat_idx, weights=weights[mask_z], minlength=n_pix_total * n_z)
            accumulators[k] += counts_flat.reshape(n_pix_total, n_z)
            
        df = None
        gc.collect()

    # Post-process and packaging
    final_results = {}
    
    # 1. Detection Rate Map
    mean_p = np.full(psf_hp_map.shape, hp.UNSEEN)
    # Correcting mean_p: since only detections are in chunks, count/N_POP is the rate
    mean_p[SEEN_idx] = det_counts_map[SEEN_idx] / N_POP_SAMPLE
    utils.plt_map(mean_p, SYS_NSIDE, np.where(~np.isnan(psf_hp_map)), 
            save_path=os.path.join(output_dir, "detection_rate_map.png"))

    for k in keys:
        pixel_counts = accumulators[k]
        sum_num = pixel_counts.sum(axis=1)
        
        # Filter to SEEN_idx and normalize
        active_counts = pixel_counts[SEEN_idx]
        active_sum_num = sum_num[SEEN_idx]
        
        hist_raw = np.zeros_like(active_counts)
        mask_v = active_sum_num > 0
        hist_raw[mask_v] = active_counts[mask_v] / (active_sum_num[mask_v][:, None] * dz)
        
        df_stats = pd.DataFrame(hist_raw)
        df_stats["sum_num"] = active_sum_num
        
        # dndz totals
        d_det = total_det_hists[k] / (total_det_nums[k] * dz) if total_det_nums[k] > 0 else total_det_hists[k]
        
        # Mocking the groupby_dndz structure for process_stats
        df_stats.loc["total_input"] = list(sim_truth['dndz_in']) + [sim_truth['num_in']]
        df_stats.loc["total_detected"] = list(d_det) + [total_det_nums[k]]
        
        final_results[k] = process_stats(df_stats, z, SEEN_idx, smooth=config.ANALYSIS_SETTINGS['smooth_nz'])
    
    return final_results


def save_diagnostic_plots(results, output_dir, key='full'):
    """Generate and save distributions plots with pixel variations."""
    stats = results[key]
    z = stats['z']
    dndzs = stats['dndzs']
    
    plt.figure(figsize=(10, 6))
    
    # Plot individual pixel variations as thin light lines
    n_pixels = dndzs.shape[0]
    n_plot = min(n_pixels, 150)
    step = max(1, n_pixels // n_plot)
    for i in range(0, n_pixels, step):
        plt.plot(z, dndzs[i], color='gray', alpha=0.05, lw=0.5)

    plt.plot(z, stats['dndz_det'], 'r-', lw=2, label='Mean Detected')
    plt.plot(z, stats['dndz_in'], 'k--', lw=1.5, label='Input (Truth)')
    
    plt.xlabel('Redshift $z$')
    plt.ylabel('$n(z)$')
    plt.legend()
    plt.title(f"Distribution: {key} (including pixel variations)")
    plt.savefig(os.path.join(output_dir, f"pixel_nz_variations.png"))
    plt.close()


def plot_tomographic_bins(results, output_dir):
    """Plot the global n(z) for all tomographic bins on one plot with variations."""
    plt.figure(figsize=(10, 6))
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    
    # Colors for the bins (matching the reference image's vibrant aesthetic)
    standard_colors = ['#3f51b5', '#e91e63', '#4caf50', '#ff9800', '#9c27b0', '#607d8b']
    if len(tomo_keys) <= len(standard_colors):
        colors = standard_colors[:len(tomo_keys)]
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, len(tomo_keys)))
    
    for i, key in enumerate(tomo_keys):
        stats = results[key]
        z = stats['z']
        dndzs = stats['dndzs']
        color = colors[i]
        
        # Plot pixel variations as extremely thin, light lines
        n_pixels = dndzs.shape[0]
        n_plot = min(n_pixels, 150)
        step = max(1, n_pixels // n_plot)
        for j in range(0, n_pixels, step):
            plt.plot(z, dndzs[j], color=color, alpha=0.2, lw=0.5)

        # Plot mean
        plt.plot(z, stats['dndz_det'], color=color, lw=2., label=f"Bin {i}")
        # Add the 'Truth' (input) for each bin as a dashed line
        # plt.plot(z, stats['dndz_in'], color=color, ls='--', lw=1.2, alpha=0.7)

    plt.xlabel('Redshift $z$')
    plt.ylabel('$n(z)$')
    plt.title('Tomographic Bin Redshift Distributions (with spatial variations)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "tomographic_bins_nz.png"))
    plt.close()


def save_fits_output(stats, bin_idx=4):
    """Store final results in a multi-HDU FITS file."""
    # Using magbin=1 as default for consistency with notebook template
    output_fits_path = OUTPUT_FITS_TEMPLATE.format(SYS_NSIDE, N_POP_SAMPLE, bin_idx)
    os.makedirs(os.path.dirname(output_fits_path), exist_ok=True)
    
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(stats['z'], name='Z'),
        fits.ImageHDU(stats['dndzs'], name='DNDZS'),
        fits.ImageHDU(stats['dndz_in'], name='DNDZ_IN'),
        fits.ImageHDU(stats['dndz_det'], name='DNDZ_DET'),
        fits.ImageHDU([stats['frac']], name='FRAC'),
        fits.ImageHDU(stats['frac_pix'], name='FRAC_PIX'),
        fits.ImageHDU(stats['sm_dndzs'], name='SM_DNDZS'),
        fits.ImageHDU(stats['sm_dndz_in'], name='SM_DNDZ_IN'),
        fits.ImageHDU(stats['sm_dndz_det'], name='SM_DNDZ_DET'),
        fits.ImageHDU(stats['SEEN_idx'], name='SEEN_IDX')
    ]
    fits.HDUList(hdus).writeto(output_fits_path, overwrite=True)
    print(f"Results saved to {output_fits_path}")


def load_fits_output(bin_idx=4):
    """Load results from a multi-HDU FITS file."""
    output_fits_path = OUTPUT_FITS_TEMPLATE.format(SYS_NSIDE, N_POP_SAMPLE, bin_idx)
    if not os.path.exists(output_fits_path):
        print(f"Warning: FITS file {output_fits_path} not found.")
        return None
    
    with fits.open(output_fits_path) as hdul:
        stats = {
            'z': hdul['Z'].data,
            'dndzs': hdul['DNDZS'].data,
            'dndz_in': hdul['DNDZ_IN'].data,
            'dndz_det': hdul['DNDZ_DET'].data,
            'frac': hdul['FRAC'].data[0],
            'frac_pix': hdul['FRAC_PIX'].data,
            'sm_dndzs': hdul['SM_DNDZS'].data,
            'sm_dndz_in': hdul['SM_DNDZ_IN'].data,
            'sm_dndz_det': hdul['SM_DNDZ_DET'].data,
            'SEEN_idx': hdul['SEEN_IDX'].data
        }
    return stats


def main():
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if config.ANALYSIS_SETTINGS.get('load_preds', True) and os.path.exists(OUTPUT_PREDS):
        print(f"Loading existing predictions from {OUTPUT_PREDS}...")
        cla_cat = pd.read_feather(OUTPUT_PREDS)
        maps, SEEN_idx = load_system_maps()
        psf_hp_map = maps[0]
        results = generate_summary_statistics_from_cat(cla_cat, psf_hp_map, SEEN_idx, output_dir)
    else:
        if not os.path.exists(OUTPUT_PREDS):
             print(f"Predictions file {OUTPUT_PREDS} not found. Running simulation...")
        gal_cat, n_degree2 = load_and_filter_catalog()
        
        chunk_files, psf_hp_map, SEEN_idx, sim_truth = simulate_and_classify_chunked(gal_cat, n_degree2)
        
        results = generate_summary_statistics_incremental(chunk_files, sim_truth, psf_hp_map, SEEN_idx, output_dir)
        
        # Consolidation of detected galaxies (ONLY detected, so much smaller)
        print(f"Re-assembling {len(chunk_files)} detected-only chunks...")
        cla_cat = pd.concat([pd.read_feather(f) for f in chunk_files], ignore_index=True)
        print(f"    Memory usage after re-assembly: {get_memory_usage():.2f} GB")
        
        # Cleanup temporary chunks
        for f in chunk_files:
            os.remove(f)
        temp_dir = os.path.join(os.path.dirname(OUTPUT_PREDS), "temp_chunks")
        try:
            os.rmdir(temp_dir)
        except:
            pass
            
        print(f"Saving detected catalog to {OUTPUT_PREDS}...")
        os.makedirs(os.path.dirname(OUTPUT_PREDS), exist_ok=True)
        cla_cat.to_feather(OUTPUT_PREDS)
        print(f"    Memory usage after saving: {get_memory_usage():.2f} GB")
    
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    # results already generated incrementally in the else block
    
    save_diagnostic_plots(results, output_dir)
    plot_tomographic_bins(results, output_dir)
    
    # Save full sample and tomographic bins
    # Using bin_idx=99 for full sample to avoid overlap with tomo bins
    save_fits_output(results['full'], bin_idx=99) 
    for i in range(len(tomo_bin_edges) - 1):
        if f'tomo_{i}' in results:
            save_fits_output(results[f'tomo_{i}'], bin_idx=i)

    print("\n--- Enhancement Factor Results ---")
    for key, stats in results.items():
        enhancement = utils.calculate_geometric_enhancement(
            stats['z'], stats['dndzs'], stats['dndz_det'], frac_pix=None,
        )
        print(f"Geometric Enhancement Factor ({key:10s}): {enhancement:.6f}")


if __name__ == "__main__":
    main()
