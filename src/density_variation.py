import os
import sys
import logging
from dataclasses import dataclass
from importlib import import_module

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import treecorr
from astropy.io import fits

try:
    from . import config
    from . import selection
    from . import utils
except ImportError:
    import config
    import selection
    import utils


logger = logging.getLogger(__name__)

SIM_CFG = config.SIM_SETTINGS
DEN_CFG = config.DENSITY_SETTINGS
TREECORR_CFG = DEN_CFG["treecorr"]
MOCK_CFG = DEN_CFG["mock"]

SYS_NSIDE = int(SIM_CFG["sys_nside_stats"])


@dataclass(frozen=True)
class DensityWThetaResult:
    theta_orig: np.ndarray
    theta_ur: np.ndarray
    theta_or: np.ndarray
    w_orig: np.ndarray
    w_ur: np.ndarray
    w_or: np.ndarray
    cov_orig: np.ndarray
    cov_ur: np.ndarray
    cov_or: np.ndarray


def _density_cache_tag() -> str:
    fp = config.SYSTEMATICS_CONFIG["footprint"]
    size_deg = float(config.SYSTEMATICS_CONFIG["tiles"]["size_deg"])
    ra0, ra1 = float(fp["ra_range"][0]) % 360.0, float(fp["ra_range"][1]) % 360.0
    dec0, dec1 = float(fp["dec_range"][0]), float(fp["dec_range"][1])
    nside = int(MOCK_CFG["glass_nside"])
    n_arcmin2 = float(MOCK_CFG["n_arcmin2"])
    z0 = float(MOCK_CFG.get("gaussian_mean", 0.6))
    sig = float(MOCK_CFG.get("gaussian_sigma", 0.23))
    source = str(MOCK_CFG.get("nz_source", "toy")).strip().lower()

    return (
        f"nside{nside}_narcmin{str(n_arcmin2).replace('.', 'p')}"
        f"_tile{str(size_deg).replace('.', 'p')}"
        f"_ra{str(ra0).replace('.', 'p')}_{str(ra1).replace('.', 'p')}"
        f"_dec{str(dec0).replace('.', 'p')}_{str(dec1).replace('.', 'p')}"
        f"_nz{source}_m{str(z0).replace('.', 'p')}_s{str(sig).replace('.', 'p')}"
    )


def _load_generate_mocksys():
    tiaogeng_path = DEN_CFG["external"]["tiaogeng_path"]
    src_path = os.path.join(tiaogeng_path, "codes", "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    return import_module("generate_mocksys")


def build_mean_p_map() -> np.ndarray:
    """Notebook-consistent mean_p from cached predictions."""
    preds_path = utils.get_output_path("output_preds")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    df = pd.read_feather(preds_path)
    required = {"pix_idx_input_p", "detection"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Predictions file missing required columns: {missing}")

    mock_sys_map_path = utils.get_output_path("mock_sys_map")
    psf_hp_map = hp.read_map(mock_sys_map_path, field=0)
    mean_p = np.full(psf_hp_map.shape, hp.UNSEEN)

    _, seen_idx_stats, _ = selection.load_system_maps(return_sim_idx=True)

    pix_idx_stats = selection.map_pix_sim_to_stats(
        df["pix_idx_input_p"].to_numpy(),
        nside_sim=SIM_CFG["sys_nside_sim"],
        nside_stats=SYS_NSIDE,
    )

    num = np.bincount(pix_idx_stats, weights=df["detection"], minlength=mean_p.size)
    den = np.bincount(pix_idx_stats, minlength=mean_p.size)
    frac_pix = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    mean_p[seen_idx_stats] = frac_pix[seen_idx_stats]

    if bool(MOCK_CFG.get("normalize_detection_by_max", True)):
        vmax = np.max(mean_p[seen_idx_stats]) if len(seen_idx_stats) > 0 else 0.0
        if vmax > 0:
            mean_p[seen_idx_stats] /= vmax

    logger.info("Built mean_p map from %s", preds_path)
    return mean_p


def load_mock_catalogs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load glass data/random catalogs from configured cache path and tag."""
    tag = _density_cache_tag()
    base = config.PATHS["sys_preds_dir"]
    glasscat = os.path.join(base, f"glass_testgalcat_density_{tag}.fits")
    glasscat_rand = os.path.join(base, f"glass_testgalcat_rand_density_{tag}.fits")

    if not os.path.exists(glasscat):
        raise FileNotFoundError(f"Missing data catalog: {glasscat}")
    if not os.path.exists(glasscat_rand):
        raise FileNotFoundError(f"Missing random catalog: {glasscat_rand}")

    with fits.open(glasscat) as h:
        lon = h[1].data["RA"]
        lat = h[1].data["Dec"]

    with fits.open(glasscat_rand) as h:
        lon_rand = h[1].data["RA"]
        lat_rand = h[1].data["Dec"]

    return lon, lat, lon_rand, lat_rand


def apply_tile_filter(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_rand: np.ndarray,
    lat_rand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only sources inside valid tiles (notebook style)."""
    gm = _load_generate_mocksys()

    fp = config.SYSTEMATICS_CONFIG["footprint"]
    size_deg = float(config.SYSTEMATICS_CONFIG["tiles"]["size_deg"])
    start_lon = float(fp["ra_range"][0])
    start_lat = float(fp["dec_range"][0])
    nlon = int(round((float(fp["ra_range"][1]) - start_lon) / size_deg))
    nlat = int(round((float(fp["dec_range"][1]) - start_lat) / size_deg))

    tiles = gm.tiles(start_lon, start_lat, size_deg, size_deg, nlon, nlat)

    tileind = tiles.get_tileind_regular(lon, lat)
    tileind_rand = tiles.get_tileind_regular(lon_rand, lat_rand)

    keep = tileind != -1
    keep_rand = tileind_rand != -1

    return lon[keep], lat[keep], lon_rand[keep_rand], lat_rand[keep_rand]


def pix2galaxy(lon: np.ndarray, lat: np.ndarray, hp_vals: np.ndarray, val_nside: int) -> np.ndarray:
    hp_ind = hp.ang2pix(val_nside, lon, lat, lonlat=True)
    return hp_vals[hp_ind]


def cat_to_hpcat(
    lon: np.ndarray,
    lat: np.ndarray,
    Ns: int,
    keep: np.ndarray | None = None,
    frac: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    patch_centers=None,
):
    if frac is not None and keep is not None:
        raise ValueError("Cannot specify both frac and keep!")

    if frac is not None:
        rand = np.random.random(size=frac.size)
        keep = (rand < np.clip(frac, 0.0, 1.0)).astype(int)
    elif keep is None:
        keep = np.ones(lon.size)

    hp_ind = hp.ang2pix(Ns, lon, lat, lonlat=True)
    gmap = np.zeros(hp.nside2npix(Ns))
    np.add.at(gmap, hp_ind, keep)

    if mask is None:
        mask = np.ones_like(gmap)
    gmap *= mask

    hp_ind_unmasked = np.arange(mask.size)[(mask != 0) * (gmap > 0)]
    hp_lon, hp_lat = hp.pix2ang(Ns, hp_ind_unmasked, lonlat=True)
    delta = gmap[hp_ind_unmasked]

    if patch_centers is None:
        return treecorr.Catalog(
            ra=hp_lon,
            dec=hp_lat,
            ra_units="deg",
            dec_units="deg",
            w=delta,
            npatch=int(TREECORR_CFG.get("n_patches", 30)),
        )

    return treecorr.Catalog(
        ra=hp_lon,
        dec=hp_lat,
        ra_units="deg",
        dec_units="deg",
        w=delta,
        patch_centers=patch_centers,
    )


def treecorr_nncor(
    catd,
    catr,
    min_sep=5,
    max_sep=250,
    nbins=20,
    bin_slop=0.0,
    sep_units="arcmin",
    var_method="bootstrap",
):
    kwargs = dict(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        bin_slop=bin_slop,
        sep_units=sep_units,
        var_method=var_method,
        cross_patch_weight="geom",
    )
    gg = treecorr.NNCorrelation(**kwargs)
    rr = treecorr.NNCorrelation(**kwargs)
    dr = treecorr.NNCorrelation(**kwargs)
    rd = treecorr.NNCorrelation(**kwargs)

    gg.process(catd)
    rr.process(catr)
    dr.process(catd, catr)
    rd.process(catr, catd)

    w, _ = gg.calculateXi(rr=rr, dr=dr, rd=rd)
    return gg.meanr, w, gg.cov


def measure_wtheta(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_rand: np.ndarray,
    lat_rand: np.ndarray,
    mean_p: np.ndarray,
) -> DensityWThetaResult:
    frac_per_galaxy = pix2galaxy(lon, lat, mean_p, SYS_NSIDE)
    frac_per_galaxy_rand = pix2galaxy(lon_rand, lat_rand, mean_p, SYS_NSIDE)

    gal_nside = int(TREECORR_CFG.get("gal_nside", 1024))

    catd_selec = cat_to_hpcat(lon, lat, gal_nside, frac=frac_per_galaxy)
    catd_orig = cat_to_hpcat(lon, lat, gal_nside, patch_centers=catd_selec.patch_centers)
    catr_or = cat_to_hpcat(lon_rand, lat_rand, gal_nside, frac=frac_per_galaxy_rand, patch_centers=catd_selec.patch_centers)
    catr_ur = cat_to_hpcat(lon_rand, lat_rand, gal_nside, patch_centers=catd_selec.patch_centers)

    treecorr.config.thread_count = int(TREECORR_CFG.get("n_threads", 20))

    nbins = int(TREECORR_CFG.get("nbins", 20))
    min_sep = float(TREECORR_CFG.get("min_sep_arcmin", 5.0))
    max_sep = float(TREECORR_CFG.get("max_sep_arcmin", 250.0))
    var_method = str(TREECORR_CFG.get("var_method", "bootstrap"))

    theta_ur, w_ur, cov_ur = treecorr_nncor(catd_selec, catr_ur, nbins=nbins, max_sep=max_sep, min_sep=min_sep, var_method=var_method)
    theta_or, w_or, cov_or = treecorr_nncor(catd_selec, catr_or, nbins=nbins, max_sep=max_sep, min_sep=min_sep, var_method=var_method)
    theta, w, cov = treecorr_nncor(catd_orig, catr_ur, nbins=nbins, max_sep=max_sep, min_sep=min_sep, var_method=var_method)

    return DensityWThetaResult(
        theta_orig=theta,
        theta_ur=theta_ur,
        theta_or=theta_or,
        w_orig=w,
        w_ur=w_ur,
        w_or=w_or,
        cov_orig=cov,
        cov_ur=cov_ur,
        cov_or=cov_or,
    )


def plot_wtheta(result: DensityWThetaResult, save_path: str):
    plt.figure(figsize=(8, 6))
    plt.errorbar(
        result.theta_orig,
        result.theta_orig * result.w_orig,
        yerr=result.theta_orig * np.sqrt(np.maximum(np.diag(result.cov_orig), 0.0)),
        fmt=".",
        label="no selection",
    )
    plt.errorbar(
        result.theta_ur,
        result.theta_ur * result.w_ur,
        yerr=result.theta_ur * np.sqrt(np.maximum(np.diag(result.cov_ur), 0.0)),
        fmt=".",
        label="uniform random",
    )
    plt.errorbar(
        result.theta_or,
        result.theta_or * result.w_or,
        yerr=result.theta_or * np.sqrt(np.maximum(np.diag(result.cov_or), 0.0)),
        fmt=".",
        label="organized random",
    )
    plt.xscale("log")
    plt.xlabel(r"$\theta$ [arcmin]")
    plt.ylabel(r"$\theta \cdot w(\theta)$")
    plt.title("Angular Correlation: Density Variation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, SIM_CFG.get("log_level", "INFO")),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    mean_p = build_mean_p_map()

    lon, lat, lon_rand, lat_rand = load_mock_catalogs()
    lon, lat, lon_rand, lat_rand = apply_tile_filter(lon, lat, lon_rand, lat_rand)
    logger.info("Tile-filtered mock catalog sizes: data=%d random=%d", lon.size, lon_rand.size)

    result = measure_wtheta(lon, lat, lon_rand, lat_rand, mean_p)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "wtheta_density_variation.png")

    plot_wtheta(result, out_path)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
