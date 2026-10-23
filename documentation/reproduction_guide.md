# Reproduction guide

What can be reproduced depends on the data available.

## A. No real data: self-test and example scoring

Run the commands in the [README](../README.md). The repository can be moved to any directory; the entry points do not read absolute paths. The models are byte-identical copies of the frozen files, and their SHA-256 values are checked against `configuration/model_registry.json` whenever a model is loaded.

## B. Prepared, frozen feature tables: training, scoring and main evaluation

Two private inputs are needed and are not included: the adult development feature matrix (`train_adult_matrix.csv`) and the frozen T0-specimen mapping for development events (`t0_specimens_development_nested.parquet`). Replace the paths with the locations of your copies:

```bash
python code/attach_specimens.py --input train_adult_matrix.csv --mapping t0_specimens_development_nested.parquet --id-column event_id --out development_18_features_with_folds.csv
python code/train.py --input development_18_features_with_folds.csv --out runs/refit_D18 --variant D18
```

- `train.py` writes the final model, the five outer-fold models, the C-selection record, nested out-of-fold scores, coefficients and input hashes to a new directory. Use `--variant` for the five sensitivity variants.
- The final C = 0.1 of D18 is not hard-coded. The script selects C by within-development log loss.
- Development-period evaluation uses the nested out-of-fold scores (`private/oof_D18.csv` in the training output), joined to the event metadata by `event_id`/`episode_id`. The 2025, 2026 and MIMIC-IV cohorts use the final model.
- Cohort names are fixed: `development_nested`, `hospital_2025`, `hospital_2026` and `MIMIC_primary_adapted`, with 25 350, 20 309, 15 789 and 310 977 events. If your data version gives different counts, the bundled results are not evidence that a rerun passed.
- The main evaluation uses 1000 paired patient-cluster bootstrap replicates with the scores held fixed; budgets and rankings are recomputed in each replicate. The intervals do not include uncertainty from model development or design choices, and are not corrected for multiplicity.
- `evaluate.py` gives AUROC, AP, Brier score and log loss with intervals, coverage and alert cost with intervals, and calibration point estimates and bins.

When the package was built, the portable code was checked against the original outputs (`audit/portable_reproduction_verification.json`):

- refitting reproduced the coefficients of the final and five outer-fold D18 models exactly;
- 372 425 scores were replayed with a maximum error of 5.6 × 10⁻¹⁷;
- the main evaluation and 96 workflow settings were reproduced, and the 2026 intervals were replayed.

The calibration intervals and the sensitivity bootstraps were not rerun; their original outputs are included and identified by hash.

## C. Full sensitivity analyses: original frozen data tree required

`original_scripts/` contains the original scripts for stages 07–10, copied byte for byte. They were written for the private project directory: they resolve paths relative to it, read private patient-level inputs and verify frozen plan files that are not published. They cannot run from this repository and are provided to show how each stage result was produced. See [original_scripts/README.md](../original_scripts/README.md).

| Stage | What it did |
|---|---|
| 07 | `prepare.py` (T0-specimen mapping), `train.py` (nested fitting), `analyze.py` (comparison of model changes). The specimen candidate fitted here (`T0_specimen`) was frozen as D18. `workflow.py` in this stage simulated alert suppression for the earlier 17-predictor reference model, not for D18. |
| 08 | Freezing of D18; unknown-outcome scenarios; cumulative and first-positive bounds |
| 09 | D18 outcome windows, laboratory ablation, pathogen composition, history observation span and early-development exclusion, alert workflow and calibration intervals |
| 10 | Group ablation removing the microbiology-history predictors or ICU status and length of stay |

Re-extracting data from the hospital laboratory and information systems needs the original extraction project, data dictionary and an approved connection, none of which are included. MIMIC-IV must be used inside an environment that holds PhysioNet credentialed access.

## Model files

`models/` holds 36 joblib files:

- the final D18 model and its five outer-fold models;
- five sensitivity variants, each with a final model and five outer-fold models.

Each final model has a `*_original_definition.json` file. Local project paths in these definitions were replaced with `<PROJECT_ROOT>` and their Chinese folder names translated; the hashes refer to the original files.
