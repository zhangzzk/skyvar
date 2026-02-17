import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from dataclasses import dataclass
from importlib import import_module

import camb
import healpy as hp
import numpy as np
import pandas as pd
import pyccl
import treecorr
from astropy.io import fits

try:
    from . import config
    from . import utils
    from . import plotting as plt_nz
except ImportError:
    import config
    import utils
    import plotting as plt_nz


DENSITY_SETTINGS = config.DENSITY_SETTINGS
EXTERNAL_CFG = DENSITY_SETTINGS["external"]
MOCK_CFG = DENSITY_SETTINGS["mock"]
TREECORR_CFG = DENSITY_SETTINGS["treecorr"]
THEORY_CFG = DENSITY_SETTINGS["theory"]
CATALOG_CFG = DENSITY_SETTINGS["catalog"]

SYS_NSIDE = config.SIM_SETTINGS["sys_nside_stats"]
N_POP_SAMPLE = config.SIM_SETTINGS["n_pop_sample"]
OUTPUT_PREDS = utils.get_output_path("output_preds")
MOCK_SYS_MAP = utils.get_output_path("mock_sys_map")


@dataclass(frozen=True)
class DensityWThetaResult:
    theta: np.ndarray
    w_orig: np.ndarray
    w_ur: np.ndarray
    w_or: np.ndarray
    cov_orig: np.ndarray
    cov_ur: np.ndarray
    cov_or: np.ndarray


def get_tiaogeng_path() -> str:
    """Resolve tiaogeng path from environment override or config default."""
    return os.environ.get("TIAOGENG_PATH", EXTERNAL_CFG["tiaogeng_path"])


def load_tiaogeng_modules(tiaogeng_path: str):
    """Import original external modules used by the prototype script."""
    src_path = os.path.join(tiaogeng_path, "codes", "src")
    if not os.path.isdir(src_path):
        raise FileNotFoundError(f"tiaogeng source path not found: {src_path}")
    if src_path not in sys.path:
        sys.path.append(src_path)

    glass_mock = import_module("glass_mock")
    generate_mocksys = import_module("generate_mocksys")
    glass_shells = import_module("glass.shells")
    return glass_mock, generate_mocksys, glass_shells


def cat_to_hpcat(lon, lat, nside, keep=None, frac=None, mask=None, patch_centers=None, rng=None):
    """
    Pixelize a catalog and build TreeCorr catalog from occupied HEALPix pixels.
    Matches the original script behavior.
    """
    if (frac is not None) and (keep is not None):
        raise ValueError("Cannot specify both frac and keep.")

    if frac is not None:
        if rng is None:
            rng = np.random.default_rng()
        rand = rng.random(size=frac.size)
        keep = (rand < frac).astype(int)
    elif keep is None:
        keep = np.ones(lon.size, dtype=int)

    hp_ind = hp.ang2pix(nside, lon, lat, lonlat=True)
    gmap = np.zeros(hp.nside2npix(nside), dtype=float)
    np.add.at(gmap, hp_ind, keep)

    if mask is None:
        mask = np.ones_like(gmap)
    gmap *= mask

    hp_ind_unmasked = np.arange(mask.size)[(mask != 0) * (gmap > 0)]
    hp_lon, hp_lat = hp.pix2ang(nside, hp_ind_unmasked, lonlat=True)
    delta = gmap[hp_ind_unmasked]

    if patch_centers is None:
        cat = treecorr.Catalog(
            ra=hp_lon,
            dec=hp_lat,
            ra_units="deg",
            dec_units="deg",
            w=delta,
            npatch=int(TREECORR_CFG["n_patches"]),
        )
    else:
        cat = treecorr.Catalog(
            ra=hp_lon,
            dec=hp_lat,
            ra_units="deg",
            dec_units="deg",
            w=delta,
            patch_centers=patch_centers,
        )
    return cat


def pix2galaxy(lon, lat, hp_vals, val_nside):
    """Assign each galaxy a value from a HEALPix map."""
    hp_ind = hp.ang2pix(val_nside, lon, lat, lonlat=True)
    return hp_vals[hp_ind]


def treecorr_nncor(
    cat_data: treecorr.Catalog,
    cat_rand: treecorr.Catalog,
    nbins: int,
    min_sep_arcmin: float,
    max_sep_arcmin: float,
    var_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Landy-Szalay NN correlation with TreeCorr."""
    nn_dd = treecorr.NNCorrelation(
        nbins=nbins,
        min_sep=min_sep_arcmin,
        max_sep=max_sep_arcmin,
        sep_units="arcmin",
        var_method=var_method,
    )
    nn_dr = treecorr.NNCorrelation(
        nbins=nbins,
        min_sep=min_sep_arcmin,
        max_sep=max_sep_arcmin,
        sep_units="arcmin",
        var_method=var_method,
    )
    nn_rr = treecorr.NNCorrelation(
        nbins=nbins,
        min_sep=min_sep_arcmin,
        max_sep=max_sep_arcmin,
        sep_units="arcmin",
        var_method=var_method,
    )

    nn_dd.process(cat_data)
    nn_dr.process(cat_data, cat_rand)
    nn_rr.process(cat_rand)

    xi, varxi = nn_dd.calculateXi(rr=nn_rr, dr=nn_dr)
    theta = np.exp(nn_dd.meanlogr)

    cov = getattr(nn_dd, "cov", None)
    if cov is None or np.shape(cov) != (len(theta), len(theta)):
        cov = np.diag(varxi)

    return theta, xi, cov


def get_selection_weights_map(output_preds: str, mock_sys_map: str, sys_nside: int, n_pop_sample: int):
    """Derive per-pixel selection weights from prediction catalog (original intent of selec_weights)."""
    cla_cat = pd.read_feather(output_preds)

    maps = hp.read_map(mock_sys_map, field=None)
    psf_hp_map = maps[0]
    seen_idx = np.where(~np.isnan(psf_hp_map))[0]

    npix = hp.nside2npix(sys_nside)
    det_counts = np.bincount(
        cla_cat["pix_idx_input_p"].to_numpy(dtype=int),
        weights=cla_cat["detection"].to_numpy(dtype=float),
        minlength=npix,
    )

    selec_weights = np.zeros(npix, dtype=float)
    selec_weights[seen_idx] = det_counts[seen_idx] / float(n_pop_sample)
    return selec_weights, seen_idx


def build_mock_catalogs(glass_mock, glass_shells, generate_mocksys, gls_path: str, tiaogeng_path: str):
    """Build original GLASS mock and uniform-random catalogs."""
    h = float(config.COSMO_PARAMS["h"])
    omega_c = float(config.COSMO_PARAMS["Omega_c"])
    omega_b = float(config.COSMO_PARAMS["Omega_b"])
    bias = float(MOCK_CFG["bias"])
    nside = int(MOCK_CFG["glass_nside"])
    n_arcmin2 = float(MOCK_CFG["n_arcmin2"])

    z = np.linspace(float(MOCK_CFG["z_min"]), float(MOCK_CFG["z_max"]), int(MOCK_CFG["z_samples"]))
    dz = (z[-1] - z[0]) / z.size
    dndz = np.exp(-((z - float(MOCK_CFG["dndz_mean"])) ** 2) / (2 * float(MOCK_CFG["dndz_sigma"]) ** 2))
    dndz /= (dndz.sum() * dz)

    zb = glass_shells.redshift_grid(z[0], z[-1], dz=float(MOCK_CFG["shell_dz"]))
    _ = glass_shells.tophat_windows(zb, dz=float(MOCK_CFG["window_dz"]))

    pars = camb.set_params(
        H0=100 * h,
        omch2=omega_c * h**2,
        ombh2=omega_b * h**2,
        WantTransfer=True,
        NonLinear=camb.model.NonLinear_both,
        halofit_version="takahashi",
    )

    gls = np.loadtxt(gls_path)

    fp = MOCK_CFG["footprint"]
    dx = float(fp["dx"])
    dy = float(fp["dy"])
    nlon = int(fp["nlon"])
    nlat = int(fp["nlat"])
    start_lon = float(fp["start_lon"])
    start_lat = float(fp["start_lat"])

    vec_vertices = hp.ang2vec(
        np.array([start_lon, start_lon, start_lon + nlon, start_lon + nlon]),
        np.array([start_lat, start_lat + nlat, start_lat + nlat, start_lat]),
        lonlat=True,
    )
    vis = np.zeros(hp.nside2npix(nside))
    vis[hp.query_polygon(nside, vec_vertices)] = 1.0

    test_tiles = generate_mocksys.tiles(start_lon, start_lat, dx, dy, nlon, nlat)

    glasscat = os.path.join(tiaogeng_path, f"data/glass_testgalcat_density_{n_arcmin2}.fits")
    glasscat_unirand = os.path.join(tiaogeng_path, f"data/glass_testgalcat_rand_density_{n_arcmin2}.fits")

    glass_mock.glass_mock(omega_c, omega_b, h, bias, z, n_arcmin2, dndz, vis, nside, glasscat, gls=gls)
    glass_mock.glass_mock(
        omega_c,
        omega_b,
        h,
        bias,
        z,
        n_arcmin2,
        dndz,
        vis,
        nside,
        glasscat_unirand,
        gls=gls,
        random=True,
    )

    return {
        "glasscat": glasscat,
        "glasscat_unirand": glasscat_unirand,
        "tiles": test_tiles,
        "z": z,
        "dndz": dndz,
        "pars": pars,
    }


def load_and_filter_tile_catalogs(glasscat: str, glasscat_unirand: str, test_tiles):
    """Load generated FITS catalogs and keep objects inside tiles."""
    with fits.open(glasscat) as hdul:
        lons = hdul[1].data["RA"]
        lats = hdul[1].data["Dec"]

    with fits.open(glasscat_unirand) as hdul:
        lons_rand = hdul[1].data["RA"]
        lats_rand = hdul[1].data["Dec"]

    source_tileind = test_tiles.get_tileind(lons, lats)
    source_tileind_rand = test_tiles.get_tileind(lons_rand, lats_rand)

    keep = source_tileind != -1
    keep_rand = source_tileind_rand != -1

    return lons[keep], lats[keep], lons_rand[keep_rand], lats_rand[keep_rand]


def measure_wtheta(lons, lats, lons_rand, lats_rand, selec_weights, sys_nside, seed: int):
    """Compute original trio of w(theta): baseline, uniform-random, organized-random."""
    frac_per_galaxy = pix2galaxy(lons, lats, selec_weights, sys_nside)
    frac_per_galaxy_rand = pix2galaxy(lons_rand, lats_rand, selec_weights, sys_nside)

    gal_nside = int(TREECORR_CFG["gal_nside"])

    rng = np.random.default_rng(seed)

    catd_selec = cat_to_hpcat(lons, lats, gal_nside, frac=frac_per_galaxy, rng=rng)
    catd_orig = cat_to_hpcat(lons, lats, gal_nside, patch_centers=catd_selec.patch_centers)
    catr_or = cat_to_hpcat(lons_rand, lats_rand, gal_nside, frac=frac_per_galaxy_rand, rng=rng,
                           patch_centers=catd_selec.patch_centers)
    catr_ur = cat_to_hpcat(lons_rand, lats_rand, gal_nside, patch_centers=catd_selec.patch_centers)

    treecorr.config.thread_count = int(TREECORR_CFG["n_threads"])

    theta_ur, w_ur, cov_ur = treecorr_nncor(
        catd_selec,
        catr_ur,
        nbins=int(TREECORR_CFG["nbins"]),
        min_sep_arcmin=float(TREECORR_CFG["min_sep_arcmin"]),
        max_sep_arcmin=float(TREECORR_CFG["max_sep_arcmin"]),
        var_method=str(TREECORR_CFG["var_method"]),
    )
    theta_or, w_or, cov_or = treecorr_nncor(
        catd_selec,
        catr_or,
        nbins=int(TREECORR_CFG["nbins"]),
        min_sep_arcmin=float(TREECORR_CFG["min_sep_arcmin"]),
        max_sep_arcmin=float(TREECORR_CFG["max_sep_arcmin"]),
        var_method=str(TREECORR_CFG["var_method"]),
    )
    theta, w_orig, cov_orig = treecorr_nncor(
        catd_orig,
        catr_ur,
        nbins=int(TREECORR_CFG["nbins"]),
        min_sep_arcmin=float(TREECORR_CFG["min_sep_arcmin"]),
        max_sep_arcmin=float(TREECORR_CFG["max_sep_arcmin"]),
        var_method=str(TREECORR_CFG["var_method"]),
    )

    if not (np.allclose(theta, theta_ur) and np.allclose(theta, theta_or)):
        raise RuntimeError("Inconsistent theta bins across TreeCorr measurements.")

    return DensityWThetaResult(
        theta=theta,
        w_orig=w_orig,
        w_ur=w_ur,
        w_or=w_or,
        cov_orig=cov_orig,
        cov_ur=cov_ur,
        cov_or=cov_or,
    )


def theory_wtheta(theta_arcmin, z, dndz, pars):
    """Reproduce original pyccl theory curve."""
    cosmo_pyccl = pyccl.Cosmology(
        Omega_c=float(config.COSMO_PARAMS["Omega_c"]),
        Omega_b=float(config.COSMO_PARAMS["Omega_b"]),
        h=float(config.COSMO_PARAMS["h"]),
        n_s=float(config.COSMO_PARAMS["n_s"]),
        A_s=pars.InitPower.As,
    )
    tracer = pyccl.NumberCountsTracer(
        cosmo_pyccl,
        has_rsd=False,
        dndz=(z, dndz),
        bias=(z, np.ones_like(z)),
    )

    ell = np.linspace(0, float(THEORY_CFG["ell_max"]), int(THEORY_CFG["ell_samples"]))
    cell = pyccl.angular_cl(cosmo_pyccl, tracer, tracer, ell)
    return pyccl.correlation(cosmo_pyccl, ell=ell, C_ell=cell, theta=np.asarray(theta_arcmin) / 60.0)


def to_plot_payload(result: DensityWThetaResult) -> dict[str, np.ndarray]:
    """Convert result dataclass to plotting payload used across the package."""
    return {
        "theta": result.theta,
        "w_orig": result.w_orig,
        "w_ur": result.w_ur,
        "w_or": result.w_or,
        "cov_orig": result.cov_orig,
        "cov_ur": result.cov_ur,
        "cov_or": result.cov_or,
    }


def main() -> None:
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    seed = int(CATALOG_CFG["seed"])
    tiaogeng_path = get_tiaogeng_path()
    gls_path = os.path.join(tiaogeng_path, "data", EXTERNAL_CFG["gls_filename"])

    if not os.path.exists(OUTPUT_PREDS):
        print(f"Error: Predictions file {OUTPUT_PREDS} not found. Run selection.py first.")
        return
    if not os.path.exists(MOCK_SYS_MAP):
        print(f"Error: Systematics map {MOCK_SYS_MAP} not found. Run systematics.py first.")
        return
    if not os.path.exists(gls_path):
        print(f"Error: GLASS spectrum file {gls_path} not found.")
        return

    # 1) External module loading (original dependency)
    glass_mock, generate_mocksys, glass_shells = load_tiaogeng_modules(tiaogeng_path)

    # 2) Build/Load GLASS mocks (original workflow)
    mock_meta = build_mock_catalogs(
        glass_mock,
        glass_shells,
        generate_mocksys,
        gls_path=gls_path,
        tiaogeng_path=tiaogeng_path,
    )

    # 3) Keep only galaxies in tiles (same as original)
    lons, lats, lons_rand, lats_rand = load_and_filter_tile_catalogs(
        mock_meta["glasscat"],
        mock_meta["glasscat_unirand"],
        mock_meta["tiles"],
    )

    # 4) Load selection map from package outputs (fills original 'selec_weights' intent)
    selec_weights, _ = get_selection_weights_map(
        output_preds=OUTPUT_PREDS,
        mock_sys_map=MOCK_SYS_MAP,
        sys_nside=SYS_NSIDE,
        n_pop_sample=N_POP_SAMPLE,
    )

    # 5) Measure correlation with different random strategies
    result = measure_wtheta(lons, lats, lons_rand, lats_rand, selec_weights, SYS_NSIDE, seed=seed)

    # 6) Theory and plotting (original comparison)
    w_theory = theory_wtheta(
        theta_arcmin=result.theta,
        z=mock_meta["z"],
        dndz=mock_meta["dndz"],
        pars=mock_meta["pars"],
    )

    plt_nz.plot_wtheta_comparison(
        to_plot_payload(result),
        w_theory=w_theory,
        save_path=os.path.join(output_dir, "wtheta_density_variation.png"),
    )


if __name__ == "__main__":
    main()
