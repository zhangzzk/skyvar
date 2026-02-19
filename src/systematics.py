"""
Build mock observing-condition maps (PSF, noise, Galactic extinction).

Parts of the geometry utilities and tile logic are adapted from
`tiaogeng/codes/src/generate_mocksys.py`.
"""

import os
import sys
import logging
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

logger = logging.getLogger(__name__)

# Core geometry helpers (adapted from generate_mocksys.py).

def lon_diff(lon1, lon2):
    """Return the wrapped longitude difference in degrees."""
    dlon = lon1 - lon2
    dlon[(dlon) > 180] -= 360
    dlon[(dlon) < -180] += 360
    return dlon

def lon_sum(lon, dlon):
    """Add a longitude offset and wrap into the valid range."""
    lon = np.atleast_1d(lon)
    dlon = np.atleast_1d(dlon)
    aux_sign = np.zeros((lon + dlon).shape)
    aux_sign[(lon+dlon < 360) * (lon+dlon > 0)] = 0
    aux_sign[lon+dlon > 360] = -1
    aux_sign[lon+dlon < 0] = 1
    return lon + dlon + 360 * aux_sign

def gaussian_2d_iso(x, y, scale, amp=1.0):
    """Evaluate an isotropic 2D Gaussian centred at the origin."""
    r2 = x**2 + y**2
    return amp * np.exp(-r2 / (2.0 * scale**2))


def normalize_footprint(ra_range, dec_range):
    """Return normalized footprint bounds and spans."""
    ra_min = float(ra_range[0]) % 360.0
    ra_max = float(ra_range[1]) % 360.0
    dec_lo = min(float(dec_range[0]), float(dec_range[1]))
    dec_hi = max(float(dec_range[0]), float(dec_range[1]))

    ra_span = (ra_max - ra_min) % 360.0
    if np.isclose(ra_span, 0.0):
        ra_span = 360.0
    dec_span = dec_hi - dec_lo
    return ra_min, ra_max, dec_lo, dec_hi, ra_span, dec_span


def in_ra_range(ra, ra_min, ra_max):
    """Check whether longitudes are inside [ra_min, ra_max], allowing wrap."""
    ra = np.asarray(ra) % 360.0
    if ra_min <= ra_max:
        return (ra >= ra_min) & (ra <= ra_max)
    return (ra >= ra_min) | (ra <= ra_max)


def build_tiles_from_footprint(ra_range, dec_range, size_deg):
    """Build tile grid robustly from configured footprint and tile size."""
    if size_deg <= 0:
        raise ValueError(f"tiles.size_deg must be > 0, got {size_deg}")

    ra_min, ra_max, dec_lo, dec_hi, ra_span, dec_span = normalize_footprint(ra_range, dec_range)
    nlon = max(1, int(np.ceil(ra_span / size_deg)))
    nlat = max(1, int(np.ceil(dec_span / size_deg)))

    # Use tile centres as grid origins so nominal tile footprint is centered on config bounds.
    start_lon = (ra_min + 0.5 * size_deg) % 360.0
    start_lat = dec_lo + 0.5 * size_deg
    tiles = Tiles(start_lon, start_lat, size_deg, size_deg, nlon, nlat)
    return tiles, (ra_min, ra_max, dec_lo, dec_hi)

class Tiles:
    """Represent a regular grid of sky tiles."""
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
        """Vectorized lookup of tile indices for input sky positions."""
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
# Systematic-map model classes.

class SystematicBase:
    """Base class for tile-based systematic models."""
    def __init__(self, tiles_obj, config_dict):
        self.tiles = tiles_obj
        self.config = config_dict

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids):
        raise NotImplementedError

    def __call__(self, pix_lons, pix_lats, pix_tile_ids=None):
        return self.eval_sys(pix_lons, pix_lats, pix_tile_ids)

    def _correlated_tile_draws(self, mean, sigma, l_corr):
        """Draw spatially-correlated tile values using an RBF covariance kernel.
        """
        centers = self.tiles.tile_centers                     # (N, 2)
        cos_dec = np.cos(np.radians(
            0.5 * (centers[:, None, 1] + centers[None, :, 1])
        ))
        dlon = lon_diff(centers[:, None, 0], centers[None, :, 0]) * cos_dec
        dlat = centers[:, None, 1] - centers[None, :, 1]
        D2 = dlon**2 + dlat**2
        C = sigma**2 * np.exp(-D2 / (2.0 * l_corr**2))
        C += np.eye(len(C)) * (1e-10 * sigma**2)             # numerical jitter
        return np.random.multivariate_normal(
            np.full(len(C), mean), C
        )


class SeeingSystematic(SystematicBase):
    """PSF seeing model: correlated + uncorrelated inter-tile + intra-tile bump."""
    def __init__(self, tiles_obj, config_dict):
        super().__init__(tiles_obj, config_dict)
        cfg = self.config
        # Smooth large-scale field
        self.tile_seeing = self._correlated_tile_draws(
            cfg['mu0'], cfg['sigma_corr'], cfg['l_corr']
        )
        # Independent tile-to-tile scatter on top
        self.tile_seeing += np.random.normal(0, cfg['sigma_uncorr'],
                                             size=self.tiles.n_tiles)

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids):
        cfg = self.config
        source_sys = np.full_like(pix_tile_ids, np.nan, dtype=float)
        for t in tqdm(range(self.tiles.n_tiles), desc="Evaluating seeing"):
            mask = pix_tile_ids == t
            if not np.any(mask):
                continue
            # Local coordinates relative to tile centre [deg]
            x = lon_diff(pix_lons[mask], self.tiles.tile_centers[t][0]) * np.cos(
                np.radians(self.tiles.tile_centers[t][1])
            )
            y = pix_lats[mask] - self.tiles.tile_centers[t][1]
            G = gaussian_2d_iso(x, y, scale=cfg['intra_scale'], amp=1.0)
            # Seeing degrades toward tile edges: PSF = s_t * (2 - G)
            noise = np.random.normal(0, cfg['sigma_pix'], size=np.sum(mask))
            source_sys[mask] = self.tile_seeing[t] * (2.0 - G) + noise
        return source_sys


class NoiseSystematic(SystematicBase):
    """Pixel-noise (RMS) model: correlated + uncorrelated inter-tile components."""
    def __init__(self, tiles_obj, config_dict):
        super().__init__(tiles_obj, config_dict)
        cfg = self.config
        # Smooth large-scale field
        self.tile_noise = self._correlated_tile_draws(
            cfg['mu0'], cfg['sigma_corr'], cfg['l_corr']
        )
        # Independent tile-to-tile scatter on top
        self.tile_noise += np.random.normal(0, cfg['sigma_uncorr'],
                                            size=self.tiles.n_tiles)

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids):
        cfg = self.config
        source_sys = np.full_like(pix_tile_ids, np.nan, dtype=float)
        for t in range(self.tiles.n_tiles):
            mask = pix_tile_ids == t
            if not np.any(mask):
                continue
            jitter = np.random.normal(0, cfg['sigma_pix'], size=np.sum(mask))
            source_sys[mask] = self.tile_noise[t] + jitter
        return source_sys

class GalacticSystematic(SystematicBase):
    """Galactic extinction model based on SFD dust maps."""
    def __init__(self, tiles_obj=None, config_dict=None):
        super().__init__(tiles_obj, config_dict)
        self.sfd = SFDQuery()

    def eval_sys(self, pix_lons, pix_lats, pix_tile_ids=None):
        coords = SkyCoord(pix_lons, pix_lats, unit='deg', frame='icrs')
        Ebv = self.sfd(coords)
        Ar = 2.285 * Ebv
        return Ar


def main():
    # Create output directories if needed.
    os.makedirs(os.path.join(config.BASE_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "output"), exist_ok=True)

    nside = config.SIM_SETTINGS['sys_nside_sim']
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra_pix, dec_pix = np.degrees(phi), np.degrees(0.5 * np.pi - theta)

    # Sky footprint from config.
    RA_MIN, RA_MAX = config.SYSTEMATICS_CONFIG['footprint']['ra_range']
    DEC_MIN, DEC_MAX = config.SYSTEMATICS_CONFIG['footprint']['dec_range']
    
    # Initialize tile grid.
    logger.info("Initializing tiles...")
    dx = config.SYSTEMATICS_CONFIG['tiles']['size_deg']
    test_tiles, (ra_min_n, ra_max_n, dec_lo_n, dec_hi_n) = build_tiles_from_footprint(
        (RA_MIN, RA_MAX), (DEC_MIN, DEC_MAX), dx
    )
    
    pix_tileind = test_tiles.get_tileind_fast(ra_pix, dec_pix)
    exact_footprint = in_ra_range(ra_pix, ra_min_n, ra_max_n) & (dec_pix >= dec_lo_n) & (dec_pix <= dec_hi_n)
    pix_tileind[~exact_footprint] = -1
    mask_footprint = pix_tileind != -1

    # Build systematics models.
    sys_noise = NoiseSystematic(test_tiles, config.SYSTEMATICS_CONFIG['noise'])
    sys_psf = SeeingSystematic(test_tiles, config.SYSTEMATICS_CONFIG['psf'])
    sys_galactic = GalacticSystematic()

    logger.info("Evaluating systematics...")
    pix_sys_noise = sys_noise(ra_pix, dec_pix, pix_tileind)
    pix_sys_psf = sys_psf(ra_pix, dec_pix, pix_tileind)
    pix_sys_galactic = sys_galactic(ra_pix, dec_pix)
    pix_sys_galactic[~mask_footprint] = np.nan

    output_path = utils.get_output_path("mock_sys_map")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Saving maps to %s...", output_path)
    hp.write_map(output_path, [pix_sys_psf, pix_sys_noise, pix_sys_galactic], overwrite=True, dtype=np.float32)

    # Combined plotting (maps + histograms).
    logger.info("Generating consolidated overview plots...")
    plt_nz.plot_systematics_overview(
        [pix_sys_psf, pix_sys_noise, pix_sys_galactic],
        ["PSF FWHM", "Pixel RMS", "Extinction Ar"],
        nside, mask_footprint,
        os.path.join("output", "sys_combined.png"),
        ra_range=(RA_MIN, RA_MAX), dec_range=(DEC_MIN, DEC_MAX),
        hist_vlims=[None, None, (0, 0.5)],
    )

if __name__ == "__main__":
    logging.basicConfig(level=getattr(logging, config.SIM_SETTINGS.get('log_level', 'INFO')),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    main()
