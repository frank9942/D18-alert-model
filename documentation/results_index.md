# Results index

All files are aggregate results. Keep the `variant`, `family` and `cohort` columns when combining files, so that results from different models or cohorts are not mixed. Cohort names: `development_nested` (nested out-of-fold scores, development period), `hospital_2025`, `hospital_2026` and `MIMIC_primary_adapted`.

## results/current_D18/ (main results)

- `clinical_points.csv`, `clinical_intervals.csv`: D18, the ICU-priority rule and the ICU+LOS rule at 1%, 2% and 5% additional budgets, with paired differences.
- `performance_points.csv`, `performance_intervals.csv`: AUROC, AP and other measures in all events with known outcomes and in known-outcome events without an H90 alert.
- `calibration_points.csv`, `calibration_intervals.csv`, `calibration_bins.csv`: D18 calibration point estimates, intervals and bins.

These files are the D18 rows of the stage 09 results, filtered or copied without changing any value.

## results/stage07/ (D18 development stage)

D18 was fitted in stage 07 as the specimen candidate `T0_specimen`. Only the D18 files of this stage are included:

- `coefficients_T0_specimen.csv`: encoded coefficients of D18.
- `t0_specimen_mapping.csv`: dictionary from hospital and MIMIC-IV specimen names to T0-specimen categories. Hospital names are kept in the original Chinese of the laboratory system, with English glosses.
- `t0_specimen_distribution.csv`: T0-specimen categories by cohort.

## results/stage08/ (unknown outcomes)

- `scenario_*`: three unknown-outcome scenarios for the D18 ranking.
- `cumulative_sharp_bounds.csv`, `first_sharp_bounds.csv`: bounds on cumulative and first-positive coverage over all feasible assignments of unknown outcomes. They are identification bounds, not confidence intervals.
- `specimen_unknown_profile*`: unknown outcomes by specimen category.
- `baselines_*`, `points_*`, `intervals_*`: per-cohort inputs to the above.

## results/stage09/ (D18 supplementary analyses)

- `endpoints_*`: complete 24-hour window, all concurrent T0 requests, and the single first T0 request; scores, history and candidate pool fixed.
- `ablation_*`: D18 versus `D18_no_labs` and `D18_availability_only` (models refitted).
- `after90_*`: `D18_after90`, refitted after excluding the first 90 days of development events.
- `history_*`: strata by observable history span. Each stratum is ranked and budgeted separately, so strata must not be added together to stand in for the full cohort.
- `pathogen_*`: exclusive and overlapping CRE/CRPA/CRAB composition, H90 coverage and additional D18 coverage. MIMIC-IV has little or no CRAB evidence, so CRAB is not fully validated there. The `D18_only_vs_D17` rows compare D18 with the earlier 17-predictor reference model.
- `workflow_*`: development-frozen thresholds with 0-, 1-, 3- and 7-day suppression. Realised budgets in other cohorts need not be 1%, 2% or 5%.
- `calibration_*`: D18 calibration measures with patient bootstrap intervals. No recalibration was applied.
- `coefficients_*`: encoded coefficients of the refitted variants.

## results/stage10/ (group ablation)

- `group_ablation_*`: D18 versus `D18_no_micro_history` and `D18_no_ICU_LOS`, paired.
- The first-layer H90 rule is unchanged when the two history predictors are removed; only the model inputs change. Removing ICU status and length of stay does not remove the fields needed by the clinical comparators.
- An ablation interval that includes 0 does not establish equivalence and does not decide whether a predictor should be removed.

## Other folders

- `figures/stage08`–`stage10`: PNG and PDF figures matching the results of the same stage.
- `aggregate_results/`: aggregate tables behind the Supplementary Material. `table_crosswalk.csv` maps the original table numbers to Supplementary Tables S1–S22, and `Core_views/` holds the panels shown in the Supplementary Material. See `aggregate_results/README.txt`.
- `documentation/cohort/`: event-level cohort descriptions ([README](cohort/README.md)).

## Audit records

- `configuration/model_registry.json`: path and SHA-256 of every model.
- `audit/portable_kernel_origins.json`: the original function and definition hash behind each portable computation in `code/kernels.py`.
- `audit/portable_reproduction_verification.json`: refit, score replay, main evaluation, workflow and 2026 interval checks run when the code package was built (15 October 2026).
- `audit/package_acceptance.json`, `audit/relocation_and_cli_verification.json`, `audit/build_stage1.json`, `audit/fixture_generation.json`, `audit/numeric_serialization_note.json`: records of that original package build. Counts in them refer to the original package, which also contained material left out of this release.
- `audit/original_stage07`–`original_stage10/`: logs of the original stage runs (fit, cross-validation, scoring and verification). Only the D18 records of stage 07 are included.
- `audit/public_release_build.json`: how this public release was derived from the original package: files removed, files changed and the checks run.
- `PACKAGE_MANIFEST.json`: SHA-256 of every code, model and data file, checked by `code/selftest.py`.

In the audit records, local project paths were replaced with `<PROJECT_ROOT>` (or `<MANUSCRIPT_MATERIALS>`) and Chinese folder and file names were translated; the hashes refer to the original files.
