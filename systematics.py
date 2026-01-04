"""
Systematics Simulation Module
-----------------------------
This module provides classes and functions for generating mock systematic maps.

Copyright Claim:
The core geometric functions (gaussian_2d, lon_diff, lon_sum) and the 'tiles' class 
are adapted from the 'tiaogeng' project (tiaogeng/codes/src/generate_mocksys.py) 
by Zekang Zhang and collaborators.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from tqdm import tqdm
from dustmaps.sfd import SFDQuery
from astropy.coordinates import SkyCoord
from astropy.io import fits

try:
    from . import utils
    from . import config
except ImportError:
    import utils
    import config

# Add paths to sys.path
if config.CODE_SRC not in sys.path:
    sys.path.append(config.CODE_SRC)

try:
    from generate_mocksys import lon_diff, gaussian_2d, tiles as Tiles
except ImportError as e:
    print(f"Warning: Could not import generate_mocksys: {e}")

# --- Core Functions (from generate_mocksys.py) ---

def gaussian_2d_inv(x, y, cov=np.eye(2)*5, xmean=0, ymean=0, amp=1, shift=0):
    '''
    2-D Gaussian-like function with explicitly inverted covariance.
    '''
    dx = x - xmean
    dy = y - ymean
    dxdy = np.vstack([dx, dy])
    chi2 = np.sum(dxdy * (np.linalg.inv(cov) @ dxdy), axis=0)
    return np.exp(-chi2/2) * amp + shift

# Note: Using gaussian_2d from generate_mocksys if available, 
# otherwise use local implementation.
if 'gaussian_2d' not in locals():
    def gaussian_2d(x, y, cov=np.eye(2)*5, xmean=0, ymean=0, amp=1, shift=0):
        return gaussian_2d_inv(x, y, cov, xmean, ymean, amp, shift)

# --- Systematic Classes ---

class SystematicBase:
    """Base class for tile-based systematics."""
    def __init__(self, tiles_obj, config_dict):
        self.tiles = tiles_obj
        self.config = config_dict

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids):
        raise NotImplementedError

    def __call__(self, pix_lons, pix_lats, pix_tile_ids=None):
        return self.eval_sys(pix_lons, pix_lats, pix_tile_ids)

class PSFSystematic(SystematicBase):
    """PSF systematic with tile-level and intra-tile variations."""
    def __init__(self, tiles_obj, config_dict):
        super().__init__(tiles_obj, config_dict)
        self.syscovs = np.zeros((self.tiles.n_tiles, 2, 2))
        self.xmeans = np.zeros(self.tiles.n_tiles)
        self.ymeans = np.zeros(self.tiles.n_tiles)

        self.delta_tiles = np.random.normal(0, self.config['sigma_tile'], size=self.tiles.n_tiles)
        self.amp_tiles = np.random.normal(self.config['Abar'], self.config['sigma_A'], size=self.tiles.n_tiles)

        for t in range(self.tiles.n_tiles):
            cov_xx = self.config['covxx_mean'] + (np.random.rand() - 0.5) * 2 * self.config['covxx_fluc']
            cov_yy = self.config['covyy_mean'] + (np.random.rand() - 0.5) * 2 * self.config['covyy_fluc']
            cov_xy = (np.random.rand() - 0.5) * 2 * self.config['covxy_fluc']
            self.syscovs[t] = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])
            self.xmeans[t] = np.random.normal(scale=self.config['xmean_fluc'])
            self.ymeans[t] = np.random.normal(scale=self.config['ymean_fluc'])

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids):
        source_sys = np.zeros_like(pix_tile_ids, dtype=float)
        for t in tqdm(range(self.tiles.n_tiles), desc="Evaluating PSF Systematic"):
            mask = (pix_tile_ids == t)
            if not np.any(mask):
                continue

            x = lon_diff(pix_lons[mask], self.tiles.tile_centers[t][0]) * np.cos(
                np.radians(self.tiles.tile_centers[t][1])
            )
            y = pix_lats[mask] - self.tiles.tile_centers[t][1]

            G = gaussian_2d(x, y, cov=self.syscovs[t], xmean=self.xmeans[t], ymean=self.ymeans[t])
            noise = np.random.normal(0, self.config['sigma_pix'], size=np.sum(mask))
            source_sys[mask] = self.config['mu0'] + self.delta_tiles[t] + self.amp_tiles[t] * G + noise

        source_sys[pix_tile_ids == -1] = np.nan
        return source_sys

class PixelNoiseSystematic(SystematicBase):
    """Pixel noise systematic with tile-level scatter."""
    def __init__(self, tiles_obj, config_dict):
        super().__init__(tiles_obj, config_dict)
        self.tile_noise = np.random.normal(loc=self.config['mu0'], scale=self.config['sigma_tile'], size=self.tiles.n_tiles)

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids):
        source_sys = np.zeros_like(pix_tile_ids, dtype=float)
        for t in range(self.tiles.n_tiles):
            mask = (pix_tile_ids == t)
            if not np.any(mask):
                continue
            noise_val = self.tile_noise[t]
            jitter = np.random.normal(0, self.config['sigma_pix'], size=np.sum(mask))
            source_sys[mask] = noise_val + jitter

        source_sys[pix_tile_ids == -1] = np.nan
        return source_sys

class GalacticSystematic(SystematicBase):
    """Galactic extinction based on SFD dust maps."""
    def __init__(self, tiles_obj=None, config_dict=None):
        super().__init__(tiles_obj, config_dict)
        self.sfd = SFDQuery()

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids=None):
        coords = SkyCoord(pix_lons, pix_lats, unit='deg', frame='icrs')
        Ebv = self.sfd(coords)
        Ar = 2.285 * Ebv
        return Ar

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate mock systematic maps.")
    parser.add_argument("--nside", type=int, default=config.SIM_SETTINGS['sys_nside'], help="HEALPix nside.")
    parser.add_argument("--output", type=str, default=None, help="Output FITS path.")
    args = parser.parse_args()

    nside = args.nside
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_pix, dec_pix = np.degrees(phi), np.degrees(0.5 * np.pi - theta)

    # Footprint from config
    RA_MIN, RA_MAX = config.SYSTEMATICS_CONFIG['footprint']['ra_range']
    DEC_MIN, DEC_MAX = config.SYSTEMATICS_CONFIG['footprint']['dec_range']
    mask_footprint = (ra_pix > RA_MIN) & (ra_pix < RA_MAX) & (dec_pix > DEC_MIN) & (dec_pix < DEC_MAX)
    
    # Initialize Tiles
    print("Initializing tiles...")
    dx = config.SYSTEMATICS_CONFIG['tiles']['size_deg']
    dy = dx
    nlon = int((RA_MAX - RA_MIN) / dx)
    nlat = int((DEC_MAX - DEC_MIN) / dy)
    test_tiles = Tiles(RA_MIN, DEC_MIN, dx, dy, nlon, nlat)
    
    pix_tileind = np.full(npix, -1, dtype=int)
    pix_tileind[mask_footprint] = test_tiles.get_tileind_fast(ra_pix[mask_footprint], dec_pix[mask_footprint])

    # Systematics
    sys_noise = PixelNoiseSystematic(test_tiles, config.SYSTEMATICS_CONFIG['noise'])
    sys_psf = PSFSystematic(test_tiles, config.SYSTEMATICS_CONFIG['psf'])
    sys_galactic = GalacticSystematic()

    print("Evaluating systematics...")
    pix_sys_noise = sys_noise(ra_pix, dec_pix, pix_tileind)
    pix_sys_psf = sys_psf(ra_pix, dec_pix, pix_tileind)
    pix_sys_galactic = sys_galactic(ra_pix, dec_pix)

    output_path = args.output or config.PATHS['mock_sys_map']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving maps to {output_path}...")
    hp.write_map(output_path, [pix_sys_psf, pix_sys_noise, pix_sys_galactic], overwrite=True, dtype=np.float32)

    # Plotting
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    utils.plt_map(pix_sys_psf, nside, mask_footprint, label="PSF FWHM", save_path=os.path.join(output_dir, "sys_map_psf.png"))
    utils.plt_map(pix_sys_noise, nside, mask_footprint, label="Pixel RMS", save_path=os.path.join(output_dir, "sys_map_noise.png"))
    utils.plt_map(pix_sys_galactic, nside, mask_footprint, label="Extinction Ar", save_path=os.path.join(output_dir, "sys_map_galactic.png"))

if __name__ == "__main__":
    main()