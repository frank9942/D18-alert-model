# D18: a prediction model to supplement resistance-history alerts at culture ordering

Code, frozen models and aggregate results for the manuscript *A clinical prediction model to supplement resistance-history alerts for carbapenem-resistant Gram-negative bacteria at culture ordering* (submitted; citation details will be added on publication).

D18 is an L2-penalised logistic regression with 18 predictors. It was developed on adult inpatient culture events (December 2023 to December 2024) at a two-campus Chinese hospital and frozen before evaluation on 2025 data, January–September 2026 data and an adapted MIMIC-IV cohort. A simulated 90-day resistance-history alert (H90) is kept as the first layer. D18 ranks the events left unflagged by H90, and the top 1%, 2% or 5% receive additional alerts. The outcome is laboratory detection of carbapenem-resistant Enterobacterales, *Pseudomonas aeruginosa* or *Acinetobacter baumannii* (CRE, CRPA, CRAB).

This is research code for offline evaluation. It produces ranking scores; it is not a deployed clinical tool and sends no alerts.

## Contents

| Path | Contents |
|---|---|
| `code/` | Portable Python code: `predict.py`, `evaluate.py`, `simulate_alerts.py`, `train.py`, `attach_specimens.py`, `selftest.py`, with shared `runtime.py` and `kernels.py` |
| `configuration/` | Current D18 definition, input schema, model registry with SHA-256 values, preprocessing parameters, encoded coefficients, workflow thresholds |
| `models/` | `main/`: final D18 model and its five outer-fold models. `sensitivity/`: five sensitivity variants, each with a final model and five outer-fold models |
| `examples/` | 60 synthetic records and their expected scores, used by the self-test |
| `results/` | Aggregate results by analysis stage; `results/current_D18/` is the entry point for the main results |
| `aggregate_results/` | Aggregate tables behind the Supplementary Material, with crosswalks to Supplementary Tables S1–S22 |
| `figures/` | Stage figures (PNG and PDF) |
| `documentation/` | Input specification, reproduction guide, results index, model card and cohort tables |
| `original_scripts/` | Archived original analysis scripts for stages 07–10 (not runnable outside the private project) |
| `audit/` | Run logs, verification records and the record of how this public release was built |
| `PACKAGE_MANIFEST.json` | SHA-256 of every code, model and data file, checked by `code/selftest.py` |

Not included: patient-level data, individual scores, bootstrap draws or database credentials. Access to the hospital data requires an application to the authors (see the Data availability statement of the article). MIMIC-IV is available from PhysioNet after credentialing and a data-use agreement.

## Quick start (no real data needed)

Use Python 3.11. From the repository root:

Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe code\selftest.py
```

macOS or Linux

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python code/selftest.py
```

`selftest.py` prints `"status": "PASS"` when all of the following hold:

- every model file matches the SHA-256 in `configuration/model_registry.json`;
- predictions for the synthetic examples match `examples/expected_scores.csv` (maximum absolute error below 1e-12);
- the allocation and performance arithmetic checks pass;
- invalid inputs (age under 18, H90 coded as 0, negative length of stay) are rejected;
- English sex labels give exactly the same predictions as the source labels, for every model;
- every file listed in `PACKAGE_MANIFEST.json` matches its SHA-256.

Then try the three entry points on the synthetic records (shown with the macOS/Linux interpreter path; on Windows use `.\.venv\Scripts\python.exe`):

```bash
.venv/bin/python code/predict.py --input examples/synthetic_18_features.csv --out runs/demo_scores.csv
.venv/bin/python code/evaluate.py --input runs/demo_scores.csv --cohort hospital_2026 --out runs/demo_evaluation --repeats 0
.venv/bin/python code/simulate_alerts.py --input runs/demo_scores.csv --cohort hospital_2026 --out runs/demo_alerts.csv --budget 2 --days 3
```

The 60 example records are synthetic and say nothing about model performance. Every output path must be a new file or an empty directory, so frozen results cannot be overwritten; use a new name under `runs/` for each run.

Keep files byte-identical. `.gitattributes` turns off line-ending conversion so that the hash checks also pass after a checkout on Windows. If you change a file listed in `PACKAGE_MANIFEST.json`, the self-test reports a mismatch for that file.

## Running on prepared data

The entry points start from a prepared 18-predictor table (extraction, time filtering and coding already done); they do not query hospital databases. See [documentation/input_data_specification.md](documentation/input_data_specification.md) for the columns and timing rules.

```bash
python code/predict.py --input prepared_events.csv --out runs/D18_scores.csv --model D18
python code/evaluate.py --input frozen_scored_events.csv --cohort hospital_2026 --out runs/evaluation_2026 --repeats 1000
python code/train.py --input development_18_features_with_folds.csv --out runs/refit_D18 --variant D18
```

`--cohort` takes `development_nested`, `hospital_2025`, `hospital_2026` or `MIMIC_primary_adapted`. Training is for development-period data only. Development-period evaluation must use the nested out-of-fold scores from the five outer models; scores of the final model on its own training data are not an internal validation. `evaluate.py` reports calibration as point estimates; the D18 calibration intervals are in `results/stage09/` and `results/current_D18/`.

| `--model` / `--variant` | Description |
|---|---|
| `D18` | Reported model: 18 predictors, final C = 0.1 |
| `D18_no_labs` | Five laboratory predictors removed (13 predictors) |
| `D18_availability_only` | Laboratory predictors reduced to available or missing |
| `D18_after90` | Refitted after excluding the first 90 days of development events |
| `D18_no_micro_history` | Two microbiology-history predictors removed (H90 first layer unchanged) |
| `D18_no_ICU_LOS` | ICU status and length of stay removed |

[documentation/reproduction_guide.md](documentation/reproduction_guide.md) describes what can be reproduced with and without the private data.

## Coding notes

- Sex can be given as `male`, `female` or `unknown` (case-insensitive), or as `MISSING` (or blank) when no value is recorded; `unknown` and `MISSING` are different categories. `code/runtime.py` converts the English values to the hospital source labels `男`, `女` and `未知`, which are the category values inside the frozen models. The models were not re-saved with English labels because any change would alter their SHA-256 values, which the article reports. Input files that use the source labels directly are also accepted.
- `MISSING` in history, medication and device predictors means that no qualifying record was found. It is not a confirmed absence, so do not recode it to 0.
- An unknown outcome is not a negative outcome. Unknown events stay in the candidate pool and use alert budget.
- The model files are scikit-learn pipelines saved with joblib. Loading a joblib file can execute code, so only load files whose SHA-256 matches `configuration/model_registry.json`; `code/runtime.py` checks this before loading.

## Documentation

- [Input data specification](documentation/input_data_specification.md)
- [Reproduction guide](documentation/reproduction_guide.md)
- [Results index](documentation/results_index.md)
- [Model card](documentation/model_card.md)
- [Cohort tables](documentation/cohort/README.md)
- [Archived original scripts](original_scripts/README.md)

## Licence

Code in `code/` and `original_scripts/` is released under the MIT License ([LICENSE](LICENSE)). Models, configuration, examples, results, aggregate results, figures, audit records and documentation are released under CC BY 4.0 ([LICENSE-DATA.md](LICENSE-DATA.md)).

## Citation

Citation details will be added when the article is published.
