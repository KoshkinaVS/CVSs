# CVSs

Toolkit for identifying, tracking, and analyzing coherent vortical structures (CVS) —
mesoscale atmospheric vortices such as polar mesocyclones and tropical cyclones — in
gridded velocity field data. This repository ties together the full pipeline (Eulerian
identification, clustering, tracking, alternative-method cross-checks, and statistics)
around a shared set of reanalysis and reference datasets. It's a general instrument set,
not scoped to a single study — pieces of it get reused across different analyses.

## What's inside

### `CS_processing/` — the main pipeline

- **`CVS_identification/`** — Eulerian vortex identification from velocity fields (Q-,
  Δ-, λ₂-criteria, swirling strength, Rortex), followed by DBSCAN clustering of the
  identified points into discrete vortex objects, in 2D and 3D. Builds on the same
  methods as [`vortex_identification`](https://github.com/KoshkinaVS/vortex_identification).
- **`CVS_tracking/`** — links identified vortices across time steps into trajectories,
  using global- and local-extrema—based association. Builds on
  [`CVS_tracking`](https://github.com/KoshkinaVS/CVS_tracking).
- **`CVS_clustering/`** — clustering of tracked vortices/trajectories, including
  clustering built on EddyClicker-derived tracks.
- **`CVS_alt_tracking/`** — an alternative tracking route via
  [TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes), used to
  cross-check results from the primary DBSCAN-based tracking.
- **`CVS_stat_tracks/`** — statistics on the resulting tracks, including comparison
  against the external [SyCLoPS](https://github.com/yepkids/SyCLoPS) cyclone catalog
  (System for Classification of Low-Pressure Systems) for validation.

### `ERA5/`

Download and preprocessing of ERA5 reanalysis fields used as input to identification and
tracking.

### `EddyClicker_tracks_processing/`

Processing of internally produced EddyClicker-labeled tracks. Used both as a comparison
baseline for algorithmically tracked trajectories and as a source for clustering in
`CVS_clustering/`.

## Data sources

- **ERA5** reanalysis (primary input for the current pipeline)
- **WRF** and **NAAD** velocity fields (supported by the underlying identification code)
- **SyCLoPS** — external cyclone-tracking catalog, used for comparison/validation
- **EddyClicker** — internally produced reference tracks, used for comparison and as
  clustering input

## Repository structure

Only code and configuration are version-controlled here — data, intermediate results, and
figures are intentionally left out (see `.gitignore`) since reanalysis/track data is too
large and machine-specific to belong in git. Expect each processing folder to contain
`.py` scripts and `.json` configs; the data/results they read and write live alongside
them locally but aren't tracked.

```
CVSs/
├── CS_processing/
│   ├── CVS_identification/
│   ├── CVS_tracking/
│   ├── CVS_clustering/
│   ├── CVS_alt_tracking/
│   └── CVS_stat_tracks/
├── ERA5/
├── EddyClicker_tracks_processing/
├── pyproject.toml          # ruff lint/format config
└── .pre-commit-config.yaml
```

## Installation & Dependencies

```bash
git clone git@github.com:KoshkinaVS/CVSs.git
cd CVSs
pip install -r requirements.txt   # numpy, scipy, xarray, netCDF4, scikit-learn, pandas, matplotlib
pre-commit install                # optional, runs ruff on commit
```

`CVS_alt_tracking/` additionally depends on a working
[TempestExtremes](https://github.com/ClimateGlobalChange/tempestextremes) installation.

## License & Contact

MIT License. Questions — koshkina.vs@phystech.edu.
