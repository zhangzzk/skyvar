```text
                         ███████╗██╗  ██╗██╗   ██╗██╗   ██╗ █████╗ ██████╗
                         ██╔════╝██║ ██╔╝╚██╗ ██╔╝██║   ██║██╔══██╗██╔══██╗
                         ███████╗█████╔╝  ╚████╔╝ ██║   ██║███████║██████╔╝
                         ╚════██║██╔═██╗   ╚██╔╝  ╚██╗ ██╔╝██╔══██║██╔══██╗
                         ███████║██║  ██╗   ██║    ╚████╔╝ ██║  ██║██║  ██║
                         ╚══════╝╚═╝  ╚═╝   ╚═╝     ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═╝
```

---

Code for measuring and modelling the impact of spatially varying selection on
the angular two-point function `w(θ)` and on tomographic redshift distributions
`n(z)`. The pipeline simulates galaxies on a HEALPix grid under varying
observing conditions, applies an XGBoost detection classifier, and quantifies

  - the **clustering enhancement** induced by spatial variation of `n(z, θ)`
    (the "true" `w(θ)` exceeds the model that uses the global `n̄(z)`), and
  - the **density variation** that selection imprints on `w(θ)` directly,
    measured against GLASS mock catalogs with TreeCorr.

The companion paper (reference to be added on publication) gives the full derivations.

## Running

The four entry points each have a `main()` and can be run as scripts:

```
python -m src.selection           # simulate, classify, write predictions
python -m src.density_variation   # measure w(θ) with selection applied
python -m src.nz_variation        # clustering enhancement on measured n(z)
python -m src.toy_variation       # toy-model sanity check (no catalog needed)
```

`selection.py` is the heaviest step: it streams chunks of the input catalog
through HEALPix pixels and the XGBoost classifier, so it expects to run on a
machine with enough memory and CPUs to make the `n_jobs=-1` default
worthwhile. The `chunk_size` and `n_pop_sample` knobs in `config.py` trade
memory for speed.

The other scripts consume the predictions cache that `selection.py` produces.
Set `ANALYSIS_SETTINGS['load_preds'] = True` to skip the simulation and reuse
the cached predictions file.

Outputs land in `${SKYVAR_DATA_DIR}` (predictions, FITS results) and in
`./output/` (PNG plots). Both are gitignored.

### Harmonic-space (C_ell) enhancement

With `CLUSTERING_SETTINGS['compute_cl'] = True`, `nz_variation` also writes the
enhanced and unenhanced angular power spectra to `cl_{run_tag}.fits` next to the
predictions (`ClusteringEnhancement.compute_enhanced_cl_from_maps`, exact
full-sky mode coupling of the selection and matter spectra).

The coupling needs matter C_ell up to `L + 3*nside_stats - 1` to be exact at
multipole `L`. With the pipeline default `ell_max = 3000` and nside 256, Delta
C_ell is exact only up to ell ~ 2230 and too low above that. The standalone
scripts below raise `ell_max` for their own use; `config.py` is unchanged.

```
python -B plot_dcl_tomo.py          # Delta C_ell / C_ell for the 6 tomographic bins
python -B plot_nside_peak_test.py   # peak position vs selection-map resolution
python tests/test_enhanced_cl.py    # C_ell route vs theta-space model on toy maps
```

Caveats: the bump near ell ~ 700 sits at the selection-map pixel limit
(3*nside - 1) and moves with nside, and the plateau above it is not converged
with resolution. Fine-scale selection-map power is partly per-pixel sampling
noise from the finite `n_pop_sample`.


## Dependencies

Python ≥ 3.10. Install the Python packages with

```
pip install -r requirements.txt
```

The pipeline also calls into a few external research codes that are *not*
packaged here. You need to clone them separately and point at them via
environment variables:

| Variable         | Used for                                                   |
|------------------|------------------------------------------------------------|
| `BLENDEMU_DIR`   | parent dir of the `blendemu` package (provides `nz_utils`) |
| `TIAOGENG_DIR`   | provides `glass_mock` and `generate_mocksys`               |

`cosmic_toolbox.arraytools` is also imported directly — install it separately.

If you do not have access to these, the corresponding entry points
(`density_variation.main`, parts of `selection.main`) will fail on import.


## Configuration

All runtime knobs live in [`src/config.py`](src/config.py): HEALPix resolution,
sample size, footprint, photo-z model, MagLim/SNR cuts, tomographic edges,
cosmology, GLASS settings, TreeCorr binning. Read the section headers — every
parameter is commented.

Filesystem paths are read from environment variables so the same config works
on different machines:

```
export SKYVAR_BASE_DIR=/path/to/skyvar
export SKYVAR_DATA_DIR=/path/to/large/scratch
export SKYVAR_GAL_CAT=/path/to/input/catalog.fits
export SKYVAR_MODEL_JSON=/path/to/classifier.json
export SKYVAR_BOUNDARY_NPY=/path/to/boundary.npy
export BLENDEMU_DIR=/path/to/blendemu
export TIAOGENG_DIR=/path/to/tiaogeng
```

`SKYVAR_BASE_DIR` defaults to the repo root; the rest must be set explicitly
before running anything that touches the catalog or classifier.




