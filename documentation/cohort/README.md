# Cohort tables (event level)

| File | Contents |
|---|---|
| `adult_frozen_cohort_size.csv` | Events, patients, known/positive/negative/unknown outcomes and H90 counts by cohort |
| `adult_baseline_characteristics_event_level.csv` | Age, length of stay, sex, ICU and H90 summaries by cohort |
| `adult_categorical_predictor_distribution.csv` | Distribution of every categorical predictor by cohort |
| `adult_17_predictor_missingness.csv` | Share of events with missing values or no qualifying evidence, by predictor and cohort |

Notes:

- Counts are per culture event. Patients with repeated cultures contribute several rows; `distinct_patients` gives the deduplicated number of patients.
- Candidate events with unknown outcomes are included. `MISSING` for a history or exposure predictor means that no qualifying positive evidence was found, not that exposure was confirmed absent.
- MIMIC-IV ages are de-identified ages, not exact ages.
- These tables were produced before the specimen-type distribution and an independent check of the request master table were added, so they are not the published baseline table; see the Supplementary Material for that.
- The tables cover the 17 predictors that D18 shares with the earlier reference model. The distribution of the 18th predictor (T0 specimen) is in `results/stage07/t0_specimen_distribution.csv`.
- Cohort labels: `train` = development (December 2023 to December 2024), `validation` = 2025, `evaluation2026` = January to September 2026.
- Sex labels: `男` = male, `女` = female, `未知` = unknown.
