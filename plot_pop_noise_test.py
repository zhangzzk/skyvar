"""Is the fine-scale selection power per-pixel sampling noise?  pop50000 vs pop100000.

Both runs: sim1024, stats nside 256, tile 1.0.  Restricted to their common pixels.
If C_sel = S + N / pop, then the ratio C_sel(50k) / C_sel(100k) -> 1 where signal
dominates and -> 2 where noise dominates, and since Delta C is linear in C_sel,
Delta C(pop -> inf) ~= 2 Delta C(100k) - Delta C(50k).
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from src import config
from src.clustering import ClusteringEnhancement, build_pyccl_cosmology

out_dir = sys.argv[1]
PRED = config.PATHS["sys_preds_dir"]
RUNS = {50000: os.path.join(PRED, "sys_nz_sim1024_stats256_pop50000_tile1.0"),
        100000: os.path.join(PRED, "sys_nz_sim1024_stats256_pop100000_tile1.0")}
cs = config.CLUSTERING_SETTINGS
edges = config.ANALYSIS_SETTINGS["tomo_bin_edges"]
NSIDE = 256
L_PLOT = 3000
ce = ClusteringEnhancement(build_pyccl_cosmology(), ell_max=L_PLOT + 3 * NSIDE - 1, ell_min=cs["ell_min"])


def load(run_dir, i):
    with fits.open(os.path.join(run_dir, f"bin_{i}.fits")) as h:
        return {"z": h["Z"].data, "dndzs": h["DNDZS"].data, "dndz_det": h["DNDZ_DET"].data,
                "SEEN_idx": h["SEEN_IDX"].data}


def restrict(s, common):
    keep = np.isin(s["SEEN_idx"], common)
    return {**s, "dndzs": s["dndzs"][keep], "SEEN_idx": s["SEEN_idx"][keep]}


ell = np.arange(L_PLOT + 1)
sel_ratio, dcl = {}, {}
for i in range(len(edges) - 1):
    st = {p: load(d, i) for p, d in RUNS.items()}
    common = np.intersect1d(st[50000]["SEEN_idx"], st[100000]["SEEN_idx"])
    st = {p: restrict(s, common) for p, s in st.items()}
    frac_s = {}
    for p, s in st.items():
        _, _, cl_sel = ce.selection_cls_from_map(s["dndzs"].T, s["dndz_det"], nside=NSIDE,
                                                 seen_idx=s["SEEN_idx"], auto_only=True)
        dz = np.gradient(s["z"])
        k = np.arange(len(dz))
        # Pair-summed selection power, normalized by the run's own mean n(z) so normalization differences drop out.
        frac_s[p] = (cl_sel[k, k] * (dz**2)[:, None]).sum(axis=0) / np.sum((s["dndz_det"] * dz) ** 2)
        res = ce.compute_enhanced_cl_from_maps(s["dndzs"].T, s["dndz_det"], s["z"], nside=NSIDE,
                                               seen_idx=s["SEEN_idx"], auto_only=cs.get("auto_only", False),
                                               lmax_out=L_PLOT)
        dcl[(p, i)] = res.delta_cl / res.cl_model
    dcl[("inf", i)] = 2 * dcl[(100000, i)] - dcl[(50000, i)]
    sel_ratio[i] = frac_s[50000] / frac_s[100000]
    pick = [10, 30, 100, 300, 700]
    print(f"bin {i}: common pix {len(common)} | C_sel(50k)/C_sel(100k) at ell {pick}: "
          f"{np.round(sel_ratio[i][pick], 2)}", flush=True)
    pick2 = [10, 100, 700, 3000]
    for p in (50000, 100000, "inf"):
        print(f"        dCl/Cl pop={str(p):>6} at ell {pick2}: {np.round(dcl[(p, i)][pick2], 4)}", flush=True)

np.savez(os.path.join(out_dir, "pop_noise_test.npz"), ell=ell, edges=np.asarray(edges),
         **{f"selratio_bin{i}": r for i, r in sel_ratio.items()},
         **{f"dcl_pop{p}_bin{i}": r for (p, i), r in dcl.items()})

colors6 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                     "xtick.color": muted, "ytick.color": muted})


def style(ax):
    ax.set_xscale("log")
    ax.grid(True, color=grid, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig, ax = plt.subplots(figsize=(7.0, 4.4))
lmax_sel = 3 * NSIDE - 1
for i, r in sel_ratio.items():
    ax.plot(np.arange(2, lmax_sel + 1), r[2:], color=colors6[i], lw=1.6, label=f"{edges[i]:.2f} – {edges[i + 1]:.2f}")
for y in (1, 2):
    ax.axhline(y, color=muted, lw=1, ls=(0, (3, 3)), zorder=0)
style(ax)
ax.set_xlim(2, lmax_sel)
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$C^{\rm sel}_\ell$(pop 50k) / $C^{\rm sel}_\ell$(pop 100k)")
ax.set_title("Selection-map power, half vs full galaxy sample (1 = real, 2 = pure noise)", color=ink,
             fontsize=11, loc="left")
leg = ax.legend(title=r"tomographic bin ($z_{\rm phot}$)", frameon=False, fontsize=9, title_fontsize=9,
                loc="upper left", labelcolor=ink)
leg.get_title().set_color(ink)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out_dir, f"pop_noise_test_selratio.{ext}"), dpi=180)

cols = {50000: "#2a78d6", 100000: "#eb6834", "inf": "#2b2b29"}
labels = {50000: "pop 50k", 100000: "pop 100k", "inf": r"extrapolated, pop $\to\infty$"}
fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
for i, ax in enumerate(axes.flat):
    for p in (50000, 100000, "inf"):
        ax.plot(ell[2:], dcl[(p, i)][2:], color=cols[p], lw=2 if p != "inf" else 1.6,
                ls="-" if p != "inf" else (0, (4, 2)), label=labels[p])
    ax.axhline(0, color=muted, lw=1, zorder=0)
    style(ax)
    ax.set_xlim(2, L_PLOT)
    ax.set_title(f"$z_{{\\rm phot}}$ {edges[i]:.2f} – {edges[i + 1]:.2f}", color=ink, fontsize=10, loc="left")
for ax in axes[1]:
    ax.set_xlabel(r"$\ell$")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\Delta C_\ell \,/\, C_\ell$")
axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=ink)
fig.suptitle("Clustering enhancement vs galaxy sample size per pixel (common footprint)", color=ink,
             fontsize=11, x=0.01, ha="left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out_dir, f"pop_noise_test_dcl.{ext}"), dpi=180)
print("saved", out_dir)
