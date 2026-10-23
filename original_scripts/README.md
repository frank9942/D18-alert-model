# Archived original analysis scripts (stages 07–10)

These are byte-identical copies of the Python scripts that fitted D18 and produced its supplementary analyses in the private project, as they stood when the code package was built on 15 October 2026. `audit/public_release_build.json` lists the SHA-256 of each file.

How these copies relate to the logged runs:

- **Stages 09 and 10:** the `train.py` hash matches the `code_sha256` recorded before fitting (`audit/original_stage09/before_fit_*.json`, `audit/original_stage10/before_fit_*.json`).
- **Stage 07:** the hashes do not match. Stage 07 scripts were edited after the fits; the stage logs record an ID-type repair of `common.py` (`audit/original_stage07/id_type_repair.json`) and a metadata clean-up of the model definitions (`audit/original_stage07/model_metadata_cleanup.json`). The versions that ran the stage 07 fits are not in this release.
- **D18 itself is unaffected:** its model file hash is unchanged, and the portable code refits it with identical coefficients (`audit/portable_reproduction_verification.json`).

They cannot run from this repository:

- they resolve paths relative to the private project directory, whose folder names are in Chinese;
- they read private patient-level inputs;
- they verify frozen plan files (`plan.json`) that are not published.

Comments and some string literals are in Chinese. This includes the hospital laboratory-system specimen names used for mapping; English glosses are in `results/stage07/t0_specimen_mapping.csv`. The portable re-implementations used by this repository are in `code/kernels.py`, and `audit/portable_kernel_origins.json` maps each one to its original function.

| Folder | Original stage |
|---|---|
| `stage07_specimen_candidate_development/` | Stage 07: targeted model changes and repeat-alert simulation. The `T0_specimen` candidate fitted here was frozen as D18. `workflow.py` simulated alert suppression for the earlier 17-predictor reference model, not for D18. |
| `stage08_D18_freeze_and_unknown_outcomes/` | Stage 08: freezing of D18; unknown-outcome scenarios and bounds |
| `stage09_D18_supplementary_analyses/` | Stage 09: outcome windows, laboratory ablation, pathogen composition, history span, early-development exclusion, workflow and calibration intervals |
| `stage10_D18_history_and_ICU_LOS_ablation/` | Stage 10: group ablation of the microbiology-history and ICU/length-of-stay predictors |

Not included:

- `plan.json` files, which are frozen plans containing private paths;
- `report.py` and `finalize.py`, which generated Chinese HTML reports and delivery manifests;
- `fix_zero_workload_ratios.py`, a one-off fix for the reference model's workflow tables;
- scripts from earlier stages that concern only the earlier reference model.
