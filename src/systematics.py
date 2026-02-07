"""
Systematics Simulation Module
-----------------------------
This module provides classes and functions for generating mock systematic maps.

Copyright Claim:
The core geometric functions (gaussian_2d, lon_diff, lon_sum) and the 'tiles' class 
are adapted from the 'tiaogeng' project (tiaogeng/codes/src/generate_mocksys.py) 
by project contributors.
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
    from . import plotting as plt_nz
except ImportError:
    import utils
    import config
    import plotting as plt_nz

# Add paths to sys.path
if config.CODE_SRC not in sys.path:
    sys.path.append(config.CODE_SRC)

# --- Core Functions (from generate_mocksys.py) ---

def lon_diff(lon1, lon2):
    """Calculate the minor arc difference between two longitudes."""
    dlon = lon1 - lon2
    dlon[(dlon) > 180] -= 360
    dlon[(dlon) < -180] += 360
    return dlon

def lon_sum(lon, dlon):
    """Calculate the longitude by adding a difference to another longitude."""
    lon = np.atleast_1d(lon)
    dlon = np.atleast_1d(dlon)
    aux_sign = np.zeros((lon + dlon).shape)
    aux_sign[(lon+dlon < 360) * (lon+dlon > 0)] = 0
    aux_sign[lon+dlon > 360] = -1
    aux_sign[lon+dlon < 0] = 1
    return lon + dlon + 360 * aux_sign

def gaussian_2d(x, y, cov=np.eye(2)*5, xmean=0, ymean=0, amp=1, shift=0):
    """2-D Gaussian-like function."""
    dx = x - xmean
    dy = y - ymean
    dxdy = np.vstack([dx, dy])
    chi2 = np.sum(dxdy * (cov @ dxdy), axis=0)
    return np.exp(-chi2/2) * amp + shift

class Tiles:
    """Class that defines a group of tiles in the sky."""
    def __init__(self, start_lon, start_lat, dx, dy, nlon, nlat):
        self.n_tiles = nlon * nlat
        lonind = np.arange(nlon)
        latind = np.arange(nlat)
        lonind, latind = np.meshgrid(lonind, latind)

        lonind = lonind.reshape(-1)
        latind = latind.reshape(-1)
        center_lats = start_lat + latind * dy
        center_lons = lon_sum(start_lon, dx * lonind / np.cos(np.radians(center_lats)))
        self.tile_centers = np.vstack([center_lons, center_lats]).T
        self.corner_lon_w = lon_diff(self.tile_centers.T[0], dx/np.cos(np.radians(self.tile_centers.T[1]))/2)
        self.corner_lon_e = lon_sum(self.tile_centers.T[0], dx/np.cos(np.radians(self.tile_centers.T[1]))/2)
        self.corner_lat_n = self.tile_centers.T[1]+dy/2
        self.corner_lat_s = self.tile_centers.T[1]-dy/2
        self.dlats = self.corner_lat_n - self.corner_lat_s
        self.dlons = lon_diff(self.corner_lon_e, self.corner_lon_w)
        
    def get_tileind_fast(self, lon, lat):
        """Vectorized version: find which tile each (lon, lat) belongs to."""
        lon = np.asarray(lon)
        lat = np.asarray(lat)
        tile_inds = np.full(lon.shape, -1, dtype=int)

        lon_min = (self.tile_centers[:, 0] - self.dlons / 2 + 360) % 360
        lon_max = (self.tile_centers[:, 0] + self.dlons / 2 + 360) % 360
        lat_min = self.tile_centers[:, 1] - self.dlats / 2
        lat_max = self.tile_centers[:, 1] + self.dlats / 2

        lon = (lon + 360) % 360

        for i, (lmin, lmax, bmin, bmax) in tqdm(enumerate(zip(lon_min, lon_max, lat_min, lat_max))):
            in_lon = (lon >= lmin) & (lon <= lmax) if lmin <= lmax else ((lon >= lmin) | (lon <= lmax))
            in_lat = (lat >= bmin) & (lat <= bmax)
            mask = in_lon & in_lat & (tile_inds == -1)
            tile_inds[mask] = i

        return tile_inds
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
    # Ensure output directories exist
    os.makedirs(os.path.join(config.BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "output"), exist_ok=True)

    nside = config.SIM_SETTINGS['sys_nside']
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_pix, dec_pix = np.degrees(phi), np.degrees(0.5 * np.pi - theta)

    # Footprint from config
    RA_MIN, RA_MAX = config.SYSTEMATICS_CONFIG['footprint']['ra_range']
    DEC_MIN, DEC_MAX = config.SYSTEMATICS_CONFIG['footprint']['dec_range']
    
    # Initialize Tiles
    print("Initializing tiles...")
    dx = config.SYSTEMATICS_CONFIG['tiles']['size_deg']
    dy = dx
    nlon = int((RA_MAX - RA_MIN) / dx)
    nlat = int((DEC_MAX - DEC_MIN) / dy)
    test_tiles = Tiles(RA_MIN, DEC_MIN, dx, dy, nlon, nlat)
    
    pix_tileind = test_tiles.get_tileind_fast(ra_pix, dec_pix)
    mask_footprint = pix_tileind != -1

    # Systematics
    sys_noise = PixelNoiseSystematic(test_tiles, config.SYSTEMATICS_CONFIG['noise'])
    sys_psf = PSFSystematic(test_tiles, config.SYSTEMATICS_CONFIG['psf'])
    sys_galactic = GalacticSystematic()

    print("Evaluating systematics...")
    pix_sys_noise = sys_noise(ra_pix, dec_pix, pix_tileind)
    pix_sys_psf = sys_psf(ra_pix, dec_pix, pix_tileind)
    pix_sys_galactic = sys_galactic(ra_pix, dec_pix)

    output_path = config.PATHS['mock_sys_map']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving maps to {output_path}...")
    hp.write_map(output_path, [pix_sys_psf, pix_sys_noise, pix_sys_galactic], overwrite=True, dtype=np.float32)

    # Combined Plotting (Maps and Histograms)
    print("Generating consolidated overview plots...")
    plt_nz.plot_systematics_overview(
        [pix_sys_psf, pix_sys_noise, pix_sys_galactic],
        ["PSF FWHM", "Pixel RMS", "Extinction Ar"],
        nside, mask_footprint, 
        os.path.join("output", "sys_combined.png")
    )

if __name__ == "__main__":
    main()