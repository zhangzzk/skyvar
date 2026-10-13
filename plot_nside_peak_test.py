"""Is the ell~700 peak in Delta C_ell / C_ell set by the selection-map resolution?

A) Same data: degrade the baseline nside-256 selection maps to 128 and 64.
B) Independent pipeline pair: sim512_stats256 vs sim512_stats64 (pop100000, tile 2.0).
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from src import config
from src.clustering import ClusteringEnhancement, build_pyccl_cosmology

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
PRED = config.PATHS["sys_preds_dir"]
BASE = os.path.join(PRED, "sys_nz_sim1024_stats256_pop50000_tile1.0")
PAIR = {256: os.path.join(PRED, "sys_nz_sim512_stats256_pop100000_tile2.0"),
        64: os.path.join(PRED, "sys_nz_sim512_stats64_pop100000_tile2.0")}
cs = config.CLUSTERING_SETTINGS
edges = config.ANALYSIS_SETTINGS["tomo_bin_edges"]
L_PLOT = 3000
cosmo = build_pyccl_cosmology()


def load(run_dir, i):
    with fits.open(os.path.join(run_dir, f"bin_{i}.fits")) as h:
        return {"z": h["Z"].data, "dndzs": h["DNDZS"].data, "dndz_det": h["DNDZ_DET"].data,
                "frac_pix": h["FRAC_PIX"].data, "SEEN_idx": h["SEEN_IDX"].data}


def degrade(stats, nside_in, nside_out):
    """Detection-weighted mean of child pixels (RING in/out)."""
    k = (nside_in // nside_out) ** 2
    parent_nest = hp.ring2nest(nside_in, stats["SEEN_idx"]) // k
    parents, inv = np.unique(parent_nest, return_inverse=True)
    w = np.asarray(stats["frac_pix"], dtype=float)
    wsum = np.bincount(inv, weights=w)
    dndzs = np.stack([np.bincount(inv, weights=w * stats["dndzs"][:, j]) for j in range(stats["dndzs"].shape[1])],
                     axis=1) / wsum[:, None]
    seen = hp.nest2ring(nside_out, parents)
    order = np.argsort(seen)
    return {**stats, "dndzs": dndzs[order], "SEEN_idx": seen[order], "frac_pix": wsum[order] / k}


def ratio(stats, nside):
    lmax_sel = 3 * nside - 1
    ce = ClusteringEnhancement(cosmo, ell_max=L_PLOT + lmax_sel, ell_min=cs["ell_min"])
    res = ce.compute_enhanced_cl_from_maps(
        n_maps=stats["dndzs"].T, nbar=stats["dndz_det"], z=stats["z"], seen_idx=stats["SEEN_idx"],
        nside=nside, auto_only=cs.get("auto_only", False), lmax_out=L_PLOT)
    return res.delta_cl / res.cl_model


def peak_ell(r):
    lo = 50
    return lo + int(np.argmax(r[lo:]))


runs = {}  # (label, nside) -> {bin: ratio}
for i in range(len(edges) - 1):
    s256 = load(BASE, i)
    print(f"bin {i}: baseline maps {s256['dndzs'].shape}, seen pix {len(s256['SEEN_idx'])}", flush=True)
    for nside in (256, 128, 64):
        s = s256 if nside == 256 else degrade(s256, 256, nside)
        runs.setdefault(("degraded", nside), {})[i] = ratio(s, nside)
    for nside, d in PAIR.items():
        runs.setdefault(("sim512", nside), {})[i] = ratio(load(d, i), nside)
    print("   peak ell: " + ", ".join(f"{lab} nside{ns}={peak_ell(r[i])}" for (lab, ns), r in runs.items()),
          flush=True)

ell = np.arange(L_PLOT + 1)
np.savez(os.path.join(out_dir, "nside_peak_test.npz"), ell=ell,
         **{f"{lab}_nside{ns}_bin{i}": r for (lab, ns), rr in runs.items() for i, r in rr.items()})

print("\nPeak ell (argmax of dCl/Cl over ell >= 50); map limit 3*nside-1 = 767 / 383 / 191")
print("bin  | degraded 256  128  64 | sim512 256  64")
for i in range(len(edges) - 1):
    d = [peak_ell(runs[("degraded", n)][i]) for n in (256, 128, 64)]
    p = [peak_ell(runs[("sim512", n)][i]) for n in (256, 64)]
    print(f"{i:4d} | {d[0]:12d} {d[1]:4d} {d[2]:3d} | {p[0]:10d} {p[1]:3d}")

colors = {256: "#2a78d6", 128: "#eb6834", 64: "#1baf7a"}
ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted})
for lab, nsides, title in (("degraded", (256, 128, 64), "baseline maps averaged to lower nside"),
                           ("sim512", (256, 64), "sim512 runs at stats nside 256 vs 64")):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
    for i, ax in enumerate(axes.flat):
        for n in nsides:
            ax.plot(ell[2:], runs[(lab, n)][i][2:], color=colors[n], lw=2, label=f"nside {n}")
            ax.axvline(3 * n - 1, color=colors[n], lw=1, ls=(0, (3, 3)))
        ax.set_xscale("log")
        ax.set_xlim(2, L_PLOT)
        ax.set_title(f"$z_{{\\rm phot}}$ {edges[i]:.2f} – {edges[i + 1]:.2f}", color=ink, fontsize=10, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(r"$\ell$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\Delta C_\ell \,/\, C_\ell$")
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=ink)
    fig.suptitle(f"Peak position vs selection-map resolution ({title}); dashed: map limit $3\\,n_{{\\rm side}}-1$",
                 color=ink, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"nside_peak_test_{lab}.{ext}"), dpi=180)
print("saved", out_dir)
