"""Delta C_ell / C_ell vs ell for the 6 tomographic bins: pop50000 (solid) vs pop100000 (dashed).

Each run on its own full footprint (sim1024, stats nside 256, tile 1.0).
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.lines import Line2D

from src import config
from src.clustering import ClusteringEnhancement, build_pyccl_cosmology

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

PRED = config.PATHS["sys_preds_dir"]
RUNS = {50000: os.path.join(PRED, "sys_nz_sim1024_stats256_pop50000_tile1.0"),
        100000: os.path.join(PRED, "sys_nz_sim1024_stats256_pop100000_tile1.0")}
cs = config.CLUSTERING_SETTINGS
edges = config.ANALYSIS_SETTINGS["tomo_bin_edges"]
NSIDE = 256
L_MIN, L_PLOT = 10, 3000
# Matter C_ell to L_PLOT + lmax_sel so every plotted ell gets its full coupling window.
ce = ClusteringEnhancement(build_pyccl_cosmology(), ell_max=L_PLOT + 3 * NSIDE - 1, ell_min=cs["ell_min"])

ratios = {}
for pop, run_dir in RUNS.items():
    for i in range(len(edges) - 1):
        with fits.open(os.path.join(run_dir, f"bin_{i}.fits")) as h:
            z, dndzs, seen = h["Z"].data, h["DNDZS"].data, h["SEEN_IDX"].data
            nbar = h["DNDZ_DET_FLAT"].data if cs.get("flat_global", False) else h["DNDZ_DET"].data
        res = ce.compute_enhanced_cl_from_maps(
            n_maps=dndzs.T, nbar=nbar, z=z, seen_idx=seen, nside=NSIDE,
            auto_only=cs.get("auto_only", False), lmax_out=L_PLOT,
        )
        ratios[(pop, i)] = res.delta_cl / res.cl_model
        print(f"pop {pop} bin {i} ({len(seen)} pix): dCl/Cl at ell=10,30,100,700,3000 =",
              ratios[(pop, i)][[10, 30, 100, 700, 3000]].round(4), flush=True)

ell = np.arange(L_PLOT + 1)
np.savez(os.path.join(out_dir, "dcl_over_cl_tomo_pop.npz"), ell=ell, edges=np.asarray(edges),
         **{f"pop{p}_bin{i}": r for (p, i), r in ratios.items()})

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted})

fig, ax = plt.subplots(figsize=(7.0, 4.4))
sl = slice(L_MIN, L_PLOT + 1)
for i in range(len(edges) - 1):
    ax.plot(ell[sl], ratios[(50000, i)][sl], color=colors[i], lw=2, label=f"{edges[i]:.2f} – {edges[i + 1]:.2f}")
    ax.plot(ell[sl], ratios[(100000, i)][sl], color=colors[i], lw=1.6, ls=(0, (4, 2)))
ax.axhline(0, color=muted, lw=1, zorder=0)
ax.set_xscale("log")
ax.set_xlim(L_MIN, L_PLOT)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\Delta C_\ell \,/\, C_\ell$")
ax.set_title("Clustering enhancement from spatially varying selection", color=ink, fontsize=12, loc="left")
ax.grid(True, which="major", color=grid, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
leg = ax.legend(title=r"tomographic bin ($z_{\rm phot}$)", frameon=False, fontsize=9.5, title_fontsize=9.5,
                loc="upper left", labelcolor=ink)
leg.get_title().set_color(ink)
ax.add_artist(leg)
style_handles = [Line2D([], [], color=muted, lw=2, label="pop 50k"),
                 Line2D([], [], color=muted, lw=1.6, ls=(0, (4, 2)), label="pop 100k")]
ax.legend(handles=style_handles, frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.3, 1.0),
          labelcolor=ink)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out_dir, f"dcl_over_cl_tomo_pop.{ext}"), dpi=200)
print("saved", out_dir)
