"""Delta C_ell / C_ell vs ell for the 6 tomographic bins, one panel.

Run from the repo root on a compute node:  python -B plot_dcl_tomo.py [out_dir, default output/]
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src import selection as sel
from src.clustering import ClusteringEnhancement, build_pyccl_cosmology

out_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
os.makedirs(out_dir, exist_ok=True)

cs = config.CLUSTERING_SETTINGS
nside = config.SIM_SETTINGS["sys_nside_stats"]
edges = config.ANALYSIS_SETTINGS["tomo_bin_edges"]
L_MIN, L_PLOT = 10, 3000
# Matter C_ell to L_PLOT + lmax_sel so every plotted ell gets its full coupling window.
ELL_MAX_MATTER = L_PLOT + 3 * nside - 1

ce = ClusteringEnhancement(build_pyccl_cosmology(), ell_max=ELL_MAX_MATTER, ell_min=cs["ell_min"])
ratios = {}
for i in range(len(edges) - 1):
    stats = sel.load_fits_output(bin_idx=i)
    nbar = stats["dndz_det_flat"] if cs.get("flat_global", False) else stats["dndz_det"]
    res = ce.compute_enhanced_cl_from_maps(
        n_maps=stats["dndzs"].T, nbar=nbar, z=stats["z"], seen_idx=stats["SEEN_idx"],
        nside=nside, auto_only=cs.get("auto_only", False), lmax_out=L_PLOT,
    )
    with np.errstate(divide="ignore", invalid="ignore"):  # cl_model is 0 below ell_min
        ratios[i] = res.delta_cl / res.cl_model
    print(f"bin {i}: dCl/Cl at ell=10,100,1000,3000 =", ratios[i][[10, 100, 1000, 3000]].round(4), flush=True)

ell = np.arange(L_PLOT + 1)
np.savez(os.path.join(out_dir, "dcl_over_cl_tomo.npz"), ell=ell, edges=np.asarray(edges),
         **{f"bin{i}": r for i, r in ratios.items()})

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ink, muted = "#2b2b29", "#6b6a64"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted})

fig, ax = plt.subplots(figsize=(7.0, 4.4))
sl = slice(L_MIN, L_PLOT + 1)
for i, r in ratios.items():
    ax.plot(ell[sl], r[sl], color=colors[i], lw=2, label=f"bin {i + 1}")
ax.axhline(0, color=muted, lw=1, zorder=0)
ax.set_xscale("log")
ax.set_xlim(L_MIN, L_PLOT)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\Delta C_\ell \,/\, C_\ell$")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=ink)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out_dir, f"dcl_over_cl_tomo.{ext}"), dpi=200)
print("saved", out_dir)
