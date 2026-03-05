import os
import sys
import logging
from dataclasses import dataclass
from importlib import import_module

import healpy as hp
import numpy as np
import pandas as pd
import treecorr
from astropy.io import fits

try:
    from . import config
    from . import clustering
    from . import selection
    from . import plotting
    from . import utils
except ImportError:
    import config
    import clustering
    import selection
    import plotting
    import utils


logger = logging.getLogger(__name__)

SIM_CFG = config.SIM_SETTINGS
ANA_CFG = config.ANALYSIS_SETTINGS
DEN_CFG = config.DENSITY_SETTINGS
TREECORR_CFG = DEN_CFG["treecorr"]
MOCK_CFG = DEN_CFG["mock"]

SYS_NSIDE = int(SIM_CFG["sys_nside_stats"])

# Module-level cache for the measured n(z), populated by main() before
# _build_nz_for_glass is called with nz_source='measured'.
_MEASURED_NZ_CACHE: dict = {}


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
    theta_theory: np.ndarray = None
    w_theory: np.ndarray = None


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


def _load_glass_mock():
    tiaogeng_path = DEN_CFG["external"]["tiaogeng_path"]
    src_path = os.path.join(tiaogeng_path, "codes", "src")
    if src_path not in sys.path:
        sys.path.append(src_path)
    return import_module("glass_mock")


def _load_gls(gls_path: str):
    """Load GLASS spectrum from either numpy-binary or text formats."""
    try:
        return np.load(gls_path)
    except Exception:
        return np.loadtxt(gls_path)


def _compute_measured_nz(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute the global detected n(z) from a pre-selected predictions catalog.

    Mirrors selection.py's dndz_det: histogram(redshift_input_p, weights=detection, density=True).
    """
    # Redshift grid from config (same as selection.py).
    z_centers, z_edges = utils.get_redshift_bins(None)

    # Detection-weighted histogram, density=True → integrates to 1.
    weights = pd.to_numeric(df["detection"], errors="coerce").fillna(0.0).to_numpy()
    dndz, _ = np.histogram(
        df["redshift_input_p"].to_numpy(), bins=z_edges,
        density=True, weights=weights,
    )
    return z_centers, dndz


def _build_nz_for_glass(z_glass: np.ndarray) -> np.ndarray:
    """Build the n(z) to feed GLASS, dispatching on MOCK_CFG['nz_source'].

    Supported modes
    ---------------
    'toy'      : Gaussian with mean/sigma from config (default).
    'measured' : Computed from the predictions file with detection threshold
                 and SNR cut applied, interpolated onto *z_glass*.
    """
    source = str(MOCK_CFG.get("nz_source", "toy")).strip().lower()

    if source == "toy":
        dz = (z_glass[-1] - z_glass[0]) / z_glass.size
        z0 = float(MOCK_CFG.get("gaussian_mean", 0.6))
        sig = float(MOCK_CFG.get("gaussian_sigma", 0.23))
        dndz = np.exp(-(z_glass - z0) ** 2 / (2 * sig ** 2))
        dndz /= (dndz.sum() * dz)
        logger.info("nz_source=toy  (z0=%.3f, sigma=%.3f)", z0, sig)
        return dndz

    if source == "measured":
        if "z" not in _MEASURED_NZ_CACHE or "dndz" not in _MEASURED_NZ_CACHE:
            raise RuntimeError(
                "nz_source='measured' but _MEASURED_NZ_CACHE is empty. "
                "main() must call _compute_measured_nz(df) first."
            )
        z_sel = _MEASURED_NZ_CACHE["z"]
        dndz_sel = _MEASURED_NZ_CACHE["dndz"]

        # Interpolate onto the GLASS z-grid (zero outside measured range).
        dndz = np.interp(z_glass, z_sel, dndz_sel, left=0.0, right=0.0)
        # Re-normalize on the glass grid.
        dz = (z_glass[-1] - z_glass[0]) / z_glass.size
        norm = dndz.sum() * dz
        if norm > 0:
            dndz /= norm
        logger.info(
            "nz_source=measured  (z_sel=[%.3f,%.3f], %d bins → %d bins)",
            z_sel[0], z_sel[-1], len(z_sel), len(z_glass),
        )
        return dndz

    raise ValueError(f"Unknown nz_source='{source}'. Choose 'toy' or 'measured'.")


def _build_mock_catalogs(glasscat: str, glasscat_rand: str) -> None:
    """Generate GLASS data/random catalogs from current config."""
    nside = int(MOCK_CFG["glass_nside"])
    n_arcmin2 = float(MOCK_CFG["n_arcmin2"])
    bias = float(MOCK_CFG.get("bias", 1.0))

    h = float(config.COSMO_PARAMS["h"])
    Oc = float(config.COSMO_PARAMS["Omega_c"])
    Ob = float(config.COSMO_PARAMS["Omega_b"])

    z_min = float(ANA_CFG.get("z_min", 0.0))
    z_max = float(ANA_CFG.get("z_max", 2.0))
    glass_z_nbins = int(MOCK_CFG.get("glass_z_nbins", 201))
    z = np.linspace(z_min, z_max, glass_z_nbins)
    dndz = _build_nz_for_glass(z)

    tiaogeng_path = DEN_CFG["external"]["tiaogeng_path"]
    gls_filename = DEN_CFG["external"]["gls_filename"]
    gls_path = os.path.join(tiaogeng_path, "data", gls_filename)
    if not os.path.exists(gls_path):
        raise FileNotFoundError(f"Missing gls file: {gls_path}")
    gls = _load_gls(gls_path)

    fp = config.SYSTEMATICS_CONFIG["footprint"]
    size_deg = float(config.SYSTEMATICS_CONFIG["tiles"]["size_deg"])
    start_lon = float(fp["ra_range"][0])
    start_lat = float(fp["dec_range"][0])
    nlon = int(round((float(fp["ra_range"][1]) - start_lon) / size_deg))
    nlat = int(round((float(fp["dec_range"][1]) - start_lat) / size_deg))

    vec_vertices = hp.ang2vec(
        np.array([start_lon, start_lon, start_lon + nlon * size_deg, start_lon + nlon * size_deg]),
        np.array([start_lat, start_lat + nlat * size_deg, start_lat + nlat * size_deg, start_lat]),
        lonlat=True,
    )
    vis = np.zeros(hp.nside2npix(nside))
    vis[hp.query_polygon(nside, vec_vertices)] = 1

    glass_mock = _load_glass_mock()
    glass_mock.glass_mock(Oc, Ob, h, bias, z, n_arcmin2, dndz.copy(), vis, nside, glasscat, gls=gls)
    glass_mock.glass_mock(Oc, Ob, h, bias, z, n_arcmin2, dndz.copy(), vis, nside, glasscat_rand, gls=gls, random=True)
    logger.info("Generated mock catalogs: %s ; %s", glasscat, glasscat_rand)


def build_mean_p_map(df: pd.DataFrame) -> np.ndarray:
    """Build mean_p map from a pre-selected predictions catalog.

    Numerator : sum(detection) per stats-pixel.
    Denominator: total simulated galaxies per stats-pixel (geometric).
    """
    # Map to stats grid.
    nside_sim = int(SIM_CFG["sys_nside_sim"])
    mock_sys_map_path = utils.get_output_path("mock_sys_map")
    psf_hp_map = hp.read_map(mock_sys_map_path, field=0)
    npix = psf_hp_map.size
    mean_p = np.full(npix, hp.UNSEEN)

    _, seen_idx_stats, seen_idx_sim = selection.load_system_maps(return_sim_idx=True)

    pix_idx_stats = selection.map_pix_sim_to_stats(
        df["pix_idx_input_p"].to_numpy(),
        nside_sim=nside_sim,
        nside_stats=SYS_NSIDE,
    )

    # Numerator: detection-weighted count per stats-pixel.
    num = np.bincount(pix_idx_stats, weights=df["detection"].to_numpy(), minlength=npix)

    # Denominator: total simulated galaxies per stats-pixel (geometric).
    n_pop_sample = int(SIM_CFG["n_pop_sample"])
    den = selection.get_input_counts_per_stats_pixel(
        seen_idx_sim, nside_sim=nside_sim,
        nside_stats=SYS_NSIDE, n_pop_sample=n_pop_sample,
    ).astype(float)

    frac_pix = np.divide(num, den, out=np.zeros(npix, dtype=float), where=den > 0)
    mean_p[seen_idx_stats] = frac_pix[seen_idx_stats]

    if bool(MOCK_CFG.get("normalize_detection_by_max", True)):
        vmax = np.max(mean_p[seen_idx_stats]) if len(seen_idx_stats) > 0 else 0.0
        if vmax > 0:
            mean_p[seen_idx_stats] /= vmax

    logger.info("Built mean_p map (%d selected galaxies)", len(df))
    return mean_p


def load_mock_catalogs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load glass data/random catalogs from configured cache path and tag."""
    tag = _density_cache_tag()
    base = config.PATHS["sys_preds_dir"]
    glasscat = os.path.join(base, f"glass_testgalcat_density_{tag}.fits")
    glasscat_rand = os.path.join(base, f"glass_testgalcat_rand_density_{tag}.fits")
    reuse = bool(DEN_CFG.get("cache", {}).get("reuse_mock_catalogs", True))

    os.makedirs(base, exist_ok=True)
    if (not reuse) or (not os.path.exists(glasscat)) or (not os.path.exists(glasscat_rand)):
        _build_mock_catalogs(glasscat, glasscat_rand)

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
    """Keep only sources inside valid tiles."""
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
    min_sep=float(config.CLUSTERING_SETTINGS["min_sep_arcmin"]),
    max_sep=float(config.CLUSTERING_SETTINGS["max_sep_arcmin"]),
    nbins=int(config.CLUSTERING_SETTINGS["nbins"]),
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

    var_method = str(TREECORR_CFG.get("var_method", "bootstrap"))

    theta_ur, w_ur, cov_ur = treecorr_nncor(catd_selec, catr_ur, var_method=var_method)
    theta_or, w_or, cov_or = treecorr_nncor(catd_selec, catr_or, var_method=var_method)
    theta, w, cov = treecorr_nncor(catd_orig, catr_ur, var_method=var_method)

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


def compute_theory_curve(
    theta_min_arcmin: float,
    theta_max_arcmin: float,
    n_theta: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute theory w(theta) from the same n(z) used for the mock catalogs.
    """
    z_min = float(ANA_CFG.get("z_min", 0.0))
    z_max = float(ANA_CFG.get("z_max", 2.0))
    glass_z_nbins = int(MOCK_CFG.get("glass_z_nbins", 201))
    z = np.linspace(z_min, z_max, glass_z_nbins)
    dndz = _build_nz_for_glass(z)

    cosmo = clustering.build_pyccl_cosmology()
    bias_val = float(MOCK_CFG.get("bias", 1.0))

    theta_deg = np.logspace(
        np.log10(theta_min_arcmin / 60.0),
        np.log10(theta_max_arcmin / 60.0),
        n_theta,
    )
    w = clustering.compute_theory_wtheta_from_dndz(
        cosmo, z, dndz, theta_deg,
        ell_min=int(config.CLUSTERING_SETTINGS["ell_min"]),
        ell_max=int(config.CLUSTERING_SETTINGS["ell_max"]),
        bias=np.full_like(z, bias_val),
    )
    logger.info("Computed theory w(theta) over %d angles", n_theta)
    return theta_deg * 60.0, w


def load_predictions() -> pd.DataFrame:
    """Load and validate the predictions file."""
    preds_path = utils.get_output_path("output_preds")
    if not os.path.exists(preds_path):
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    df = pd.read_feather(preds_path)
    required = {"pix_idx_input_p", "redshift_input_p", "detection"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Predictions file missing required columns: {missing}")
    logger.info("Loaded %d predictions from %s", len(df), preds_path)
    return df


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, SIM_CFG.get("log_level", "INFO")),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 0. Load predictions once, apply selection once.
    df_raw = load_predictions()
    sel_mode = str(MOCK_CFG.get("selection_mode", "snr")).strip().lower()
    sel_mode = None if sel_mode in ("", "none") else sel_mode
    df = selection.apply_galaxy_selection(df_raw, mode=sel_mode)
    del df_raw
    logger.info("Selected %d galaxies (mode=%s)", len(df), sel_mode)

    # 1. Build detection-fraction map.
    mean_p = build_mean_p_map(df)

    # 2. Generate / load mock galaxy catalogs.
    #    If nz_source='measured', the n(z) is computed from df here.
    nz_source = str(MOCK_CFG.get("nz_source", "toy")).strip().lower()
    if nz_source == "measured":
        z_sel, dndz_sel = _compute_measured_nz(df)
        # Stash for _build_nz_for_glass to pick up.
        _MEASURED_NZ_CACHE["z"] = z_sel
        _MEASURED_NZ_CACHE["dndz"] = dndz_sel

    lon, lat, lon_rand, lat_rand = load_mock_catalogs()
    lon, lat, lon_rand, lat_rand = apply_tile_filter(lon, lat, lon_rand, lat_rand)
    logger.info("Tile-filtered mock catalog sizes: data=%d random=%d", lon.size, lon_rand.size)

    # 3. Measure w(theta) with selection applied.
    result = measure_wtheta(lon, lat, lon_rand, lat_rand, mean_p)

    # 4. Theory curve (same n(z) and bias as the mocks).
    theta_theory, w_theory = compute_theory_curve(
        theta_min_arcmin=result.theta_orig.min(),
        theta_max_arcmin=result.theta_orig.max(),
    )
    from dataclasses import replace
    result = replace(result, theta_theory=theta_theory, w_theory=w_theory)

    # 5. Save plot.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "wtheta_density_variation.png")

    plotting.plot_density_variation_wtheta(result, out_path)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
