import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter1d
import xgboost as xgb
import argparse

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
    ].reset_index(drop=True)

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
    subset['pix_idx'] = i

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


def simulate_pixel_observables(gal_cat, n_degree2):
    """Run parallel simulation of observables per pixel."""
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

    print(f"Running parallel processing on {len(SEEN_idx)} pixels...")
    results = Parallel(n_jobs=N_JOBS, backend="threading")(
        delayed(process_one)(i, idx, gal_cat, conditions, N_POP_SAMPLE,
                             psf_hp_map, noise_hp_map, galactic_hp_map, detec_mag_bound)
        for i, idx in enumerate(SEEN_idx)
    )
    fullset = pd.concat(results, ignore_index=True)

    tot_num = fullset.shape[0]
    area_sq = tot_num / n_degree2
    side = np.sqrt(area_sq)
    print(f'Simulated Area: {area_sq:.2f} degree^2')
    fullset["RA"] = np.random.uniform(0, side, size=tot_num)
    fullset["DEC"] = np.random.uniform(0, side, size=tot_num)
    
    return fullset, psf_hp_map, SEEN_idx


def run_xgb_classification(fullset):
    """Run XGBoost classification on the simulated catalog."""
    print("Loading XGBoost model and predicting...")
    bst_cla = xgb.Booster({'device': 'cuda', 'n_jobs': -1})
    bst_cla.load_model(MODEL_JSON)

    tot_num = len(fullset)
    chunks = range(0, tot_num, CHUNK_SIZE)
    cla_cat_list = []
    for i, start in enumerate(chunks):
        end = min(start + CHUNK_SIZE, tot_num)
        print(f"  Chunk {i + 1}/{len(chunks)}")
        chunk = fullset.iloc[start:end].copy()
        try:
            cla_chunk = nz_utils.icat2cla_v2(chunk, chunk, bst_cla, predict=True)
            cla_cat_list.append(cla_chunk)
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
    
    cla_cat = pd.concat(cla_cat_list, ignore_index=True)
    
    print(f"Saving predictions to {OUTPUT_PREDS}...")
    os.makedirs(os.path.dirname(OUTPUT_PREDS), exist_ok=True)
    cla_cat.to_feather(OUTPUT_PREDS)
    
    return cla_cat


def process_stats(sys_res, z, SEEN_idx):
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


def generate_summary_statistics(cla_cat, psf_hp_map, SEEN_idx, output_dir, tomo_bin_edges=None):
    """Compute detection maps and dN/dz distributions."""
    mean_p = np.full(psf_hp_map.shape, hp.UNSEEN)
    counts = np.bincount(cla_cat['pix_idx_input_p'])
    weighted_counts = np.bincount(cla_cat['pix_idx_input_p'], weights=cla_cat['detection'])
    
    max_idx = len(SEEN_idx)
    detection_rate = np.zeros(max_idx)
    mask_valid = counts[:max_idx] > 0
    detection_rate[mask_valid] = weighted_counts[:max_idx][mask_valid] / counts[:max_idx][mask_valid]
    mean_p[SEEN_idx] = detection_rate
    
    utils.plt_map(mean_p, SYS_NSIDE, np.where(~np.isnan(psf_hp_map)), 
            s=20, save_path=os.path.join(output_dir, "detection_rate_map.png"))
    
    z = np.linspace(0, config.ANALYSIS_SETTINGS['z_max'], config.ANALYSIS_SETTINGS['z_bins'])
    results = {}
    
    print("Calculating dN/dz for full sample...")
    sys_res = groupby_dndz(cla_cat, z, post_cut=None)
    stats_full = process_stats(sys_res, z, SEEN_idx)
    results['full'] = stats_full
    
    if tomo_bin_edges is not None:
        print(f"Calculating dN/dz for {len(tomo_bin_edges)-1} tomographic bins...")
        p_weights = utils.get_photoz_weights(cla_cat, tomo_bin_edges)
        for i in range(len(tomo_bin_edges)-1):
            w_col = f"tomo_p_{i}"
            cla_cat[w_col] = p_weights[:, i]
            sys_res_i = groupby_dndz(cla_cat, z, post_cut=None, weight_col=w_col)
            results[f'tomo_{i}'] = process_stats(sys_res_i, z, SEEN_idx)
            del cla_cat[w_col]
            
    return results


def save_diagnostic_plots(results, output_dir):
    """Generate and save distributions plots."""
    for key, stats in results.items():
        if key == 'full' or key.startswith('tomo_'):
            plt.figure(figsize=(10, 6))
            plt.plot(stats['z'], stats['dndz_det'], 'r-', label='Detected')
            plt.plot(stats['z'], stats['dndz_in'], 'k--', label='Input')
            plt.legend()
            plt.title(f"Distribution: {key}")
            plt.savefig(os.path.join(output_dir, f"pixel_nz_variations_{key}.png"))
            plt.close()


def plot_tomographic_bins(results, output_dir):
    """Plot the global n(z) for all tomographic bins on one plot."""
    plt.figure(figsize=(10, 6))
    tomo_keys = sorted([k for k in results.keys() if k.startswith('tomo_')], 
                       key=lambda x: int(x.split('_')[1]))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(tomo_keys)))
    for i, key in enumerate(tomo_keys):
        stats = results[key]
        plt.plot(stats['z'], stats['dndz_det'], color=colors[i], lw=2, label=f"Bin {i}")
        plt.fill_between(stats['z'], 0, stats['dndz_det'], color=colors[i], alpha=0.2)

    plt.xlabel('Redshift $z$')
    plt.ylabel('$n(z)$')
    plt.title('Tomographic Bin Redshift Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "tomographic_bins_nz.png"))
    plt.close()


def save_fits_output(stats):
    """Store final results in a multi-HDU FITS file."""
    output_fits_path = OUTPUT_FITS_TEMPLATE.format(SYS_NSIDE, N_POP_SAMPLE, None, 4)
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


def main():
    parser = argparse.ArgumentParser(description="Run selection simulation and analysis.")
    parser.add_argument("--load_preds", action="store_true", help="Load existing predictions.")
    args = parser.parse_args()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    if args.load_preds:
        print(f"Loading existing predictions from {OUTPUT_PREDS}...")
        cla_cat = pd.read_feather(OUTPUT_PREDS)
        maps, SEEN_idx = load_system_maps()
        psf_hp_map = maps[0]
    else:
        gal_cat, n_degree2 = load_and_filter_catalog()
        fullset, psf_hp_map, SEEN_idx = simulate_pixel_observables(gal_cat, n_degree2)
        cla_cat = run_xgb_classification(fullset)
    
    tomo_bin_edges = config.ANALYSIS_SETTINGS['tomo_bin_edges']
    results = generate_summary_statistics(cla_cat, psf_hp_map, SEEN_idx, output_dir, tomo_bin_edges=tomo_bin_edges)
    
    save_diagnostic_plots(results, output_dir)
    plot_tomographic_bins(results, output_dir)
    save_fits_output(results['full'])

    print("\n--- Enhancement Factor Results ---")
    for key, stats in results.items():
        enhancement = utils.calculate_geometric_enhancement(
            stats['z'], stats['dndzs'], stats['dndz_det'], frac_pix=stats['frac_pix']
        )
        print(f"Geometric Enhancement Factor ({key:10s}): {enhancement:.6f}")


if __name__ == "__main__":
    main()
