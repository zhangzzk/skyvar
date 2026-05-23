# Bias Variation

This package extends the default SkyVar pipeline with spatially varying mean
galaxy-bias products.  Catalog/source-pool generation and SHMR/halo-bias
assignment are local to `bias_variation`; the selection workflows intentionally
reuse `src.selection` for observing-condition maps, classifier schema,
detection threshold, and optional SNR/MagLim cuts.

## Pipeline

1. `python -m bias_variation.catalog` builds GalSBI-SPS source pools
   independently of SkyVar systematics maps.  All generation and quality-cut
   parameters come from `config.py`.
   Catalog generation uses GalSBI's stellar-mass-function count model and
   auto-computes high enough `n_gal_max_*` caps so the requested footprint is
   density-faithful.  If the requested footprint would generate an enormous
   catalog, the preflight fails early unless
   `SPS_GENERATION_SETTINGS["allow_large_density_run"]` is enabled.  Set
   `USE_CUSTOM_SPS_POSTERIOR` and `SPS_POSTERIOR_PATH` in `config.py` to use a
   custom GalSBI-compatible HDF5 posterior table; disable it to use GalSBI's
   packaged default.  The generated JSON summary includes the post-quality-cut
   density.
2. `python -m bias_variation.halo_bias assign-source` loads that source pool,
   assigns stellar mass, halo mass, and Tinker+10 bias, and writes the compact
   Feather source table consumed by selection.  Its JSON summary carries the
   catalog-derived post-quality-cut source density.
3. `python -m bias_variation.selection full --source-table ...` follows the
   default SkyVar per-simulation-pixel draw semantics, classifies the drawn
   sources, applies selection, and writes selected/unselected `b(z, pixel)` maps.
   `n_pop` is computed from the source-table summary density and
   `nside_sim` pixel area, with a config fallback only when that summary is
   missing.
   Set `FULL_WORKFLOW_SETTINGS["observing_conditions_mode"] = "constant"` to
   keep PSF/noise fixed at the baseline values from `src.config.OBS_CONDITIONS`
   and set Galactic extinction to zero, which isolates sample-variance-induced
   variation.  The same mode can be selected with `--observing-conditions
   constant`.
4. `python -m bias_variation.selection tomo --source-table ...` adds the same
   tomographic photo-z bins used by the default `n(z)` workflow.
5. `plotting.py` provides diagnostics for the maps written by the `full` and
   `tomo` workflows.

## Current Caveats

- The current GalSBI-SPS `Tortorelli+25` `mode="intrinsic"` source pool is not
  yet a validated F24-density replacement.  A one-pixel smoke test gave
  `177.31 arcmin^-2` intrinsic SPS count density but only `31.65 arcmin^-2`
  after the SkyVar quality cuts; the F24 catalog gives `183.48 arcmin^-2`
  after the same cuts.
- The mismatch is driven by photometry, not caps or footprint arithmetic.  The
  SPS smoke r-band `int_mag` median is about `29.83`, while F24 is about
  `26.65`; most SPS intrinsic rows fail `r < 28`.
- GalSBI's intrinsic SPS API is valid, but the plugin samples the stellar-mass
  function before computing apparent magnitudes and does not apply the F24-like
  apparent-magnitude preselection used by the older catalog.
- Production `n_pop` policy is therefore unresolved: either build/identify a
  magnitude-limited SPS source-pool path that matches F24 after cuts, or use the
  F24 post-cut density for `n_pop` while using SPS only as the stellar-mass/bias
  property reservoir.
- Until that decision is encoded in config and job names, do not interpret
  SPS-density-derived `n_pop` production maps as science products.

Numerical workflow knobs live in `config.py`.  Common sections are
`SOURCE_SETTINGS`, `QUALITY_CUT_SETTINGS`, `FULL_WORKFLOW_SETTINGS`,
`SPS_GENERATION_SETTINGS`, and `BIAS_ASSIGNMENT_SETTINGS`.  The Slurm scripts
intentionally do not override these values.

## Structure

- `config.py`: bias-workflow defaults, cosmology defaults, and GalSBI-SPS
  generation settings.
- `utils.py`: small technical helpers for JSON, pixels, NPZ, and statistics.
- `catalog.py`: minimal catalog discovery/loading/standardization, configured
  quality cuts, density helpers, and no-argument GalSBI-SPS generation.
- `shmr.py`, `halo_bias.py`: the physical stellar-to-halo-mass and halo-bias
  relation, source-table bias assignment, and halo-bias diagnostics.
- `selection.py`: a thin wrapper around the default SkyVar selection flow.
  It expects preassigned `bias_tinker10`, uses `src.selection` for source
  drawing, classification inputs, galaxy selection, and tomographic binning,
  and changes only the accumulator from `n(z)` counts to mean `b(z)`.
  Exposed commands are `full` and `tomo`.
- `plotting.py`: diagnostics and map/profile plotting.

Generated products should stay under `bias_variation/output/` or the project
scratch directory configured by `PROJECT_BIAS_DIR` in `config.py`.
Cached classified chunks are tied to their observing-condition mode; do not
reuse map-varying chunks for constant-baseline runs.
