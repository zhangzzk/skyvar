"""Delta C_ell / C_ell vs ell for the 6 tomographic bins, one panel.  Run on a compute node:  python -B plot_dcl_tomo.py output/"""
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

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

cs = config.CLUSTERING_SETTINGS
nside = config.SIM_SETTINGS["sys_nside_stats"]
edges = config.ANALYSIS_SETTINGS["tomo_bin_edges"]
L_PLOT = 3000
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
    ratios[i] = res.delta_cl / res.cl_model
    print(f"bin {i}: dCl/Cl at ell=10,100,1000,3000 =", ratios[i][[10, 100, 1000, 3000]].round(4), flush=True)

ell = np.arange(L_PLOT + 1)
np.savez(os.path.join(out_dir, "dcl_over_cl_tomo.npz"), ell=ell, edges=np.asarray(edges),
         **{f"bin{i}": r for i, r in ratios.items()})

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted})

fig, ax = plt.subplots(figsize=(7.0, 4.4))
sl = slice(cs["ell_min"], L_PLOT + 1)
for i, r in ratios.items():
    ax.plot(ell[sl], r[sl], color=colors[i], lw=2, label=f"{edges[i]:.2f} – {edges[i + 1]:.2f}")
ax.axhline(0, color=muted, lw=1, zorder=0)
ax.set_xscale("log")
ax.set_xlim(cs["ell_min"], L_PLOT)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\Delta C_\ell \,/\, C_\ell$")
ax.set_title("Clustering enhancement from spatially varying selection", color=ink, fontsize=12, loc="left")
ax.grid(True, which="major", color=grid, lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
leg = ax.legend(title=r"tomographic bin ($z_{\rm phot}$)", frameon=False, fontsize=9.5, title_fontsize=9.5,
                loc="upper left", bbox_to_anchor=(0.2, 1.0), labelcolor=ink)
leg.get_title().set_color(ink)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out_dir, f"dcl_over_cl_tomo.{ext}"), dpi=200)
print("saved", out_dir)
