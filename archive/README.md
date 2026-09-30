# Archive of the H200 server workspace (taken 2026-09-30, before the server was decommissioned)

- `results/` (in git): every run's metrics and metadata mirrored from `outputs/` on the server —
  `results*.json`, `per_view*.json`, `run_meta.txt`, `stages.txt`, `cfg_args`, learned phase fields
  (`phase_fields.pt`, `phase_shared.json`, `phase_summary.json`), gauge fits, and all calibration-audit JSONs
  (`outputs/calib/<scene>/c1*.json`, `c4_holdout.json`, `c5_*.json`, `pf_focals.json`, `swap/*.json`).
  Paths are the server paths relative to the project root.
- `calib_sparse/` (local only, ~1.1 GB; copy on Drive): the re-calibrated COLMAP models
  `<scene>_{pf,pfpp,pfa}/sparse/0` (per-image focal; + per-image principal point for `pfpp`; `pfa` = Sim3-aligned
  to the released frame). Images are NOT included — use the official dataset releases (the calibrations use the
  released `images*` folders unchanged).
- Drive folder `3dgs-refiner-it/` (inside the shared folder): `checkpoints/` = the best model per scene on the
  re-calibrated cameras (report §8.5: MCMC `*_mcmc_pf`, `treehill_mcmc_pfa`, truck/train `pfpp`, treehill + phase field),
  each with `point_cloud/iteration_30000/point_cloud.ply`, `cfg_args`, `cameras.json`, results; plus `calib_sparse/`.

Not archived (re-obtainable): public datasets, IBGS released checkpoints, GADA published renders, intermediate runs.
