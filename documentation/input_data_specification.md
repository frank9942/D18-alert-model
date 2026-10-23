# Input data specification

## 1. Columns

All 18 columns in `configuration/input_schema.csv` are required. The sensitivity variants take the same 18 columns and drop their excluded columns internally.

- Files are UTF-8 CSV. Category values are case-sensitive, except the English `sex` values below.
- `age` and `length_of_stay` are raw numbers; do not standardise them beforehand.
- The five laboratory predictors are categories (`low`, `normal`, `high`, `MISSING`), not concentrations.
- `sex` accepts `male`, `female` or `unknown` (case-insensitive) and `MISSING`. `unknown` means that sex was recorded as unknown; `MISSING` means that no value was available. The English values are converted to the source labels `男`, `女` and `未知` used inside the frozen models, so input files with the source labels give identical scores.
- Median imputation, standardisation, category encoding and coefficients are stored inside each joblib model. Readable versions are in `configuration/preprocessing_parameters.json` and `configuration/encoded_coefficients.csv`.

## 2. Eligibility and timing

- An event is adult when the patient is aged 18 or over at T0, the first culture request of the culture event. `age` must not be missing.
- `length_of_stay` is the number of days in hospital up to T0, not the total length of the eventual stay.
- History, medication and device predictors use information available before T0. The current outcome and later reports are never inputs. Reports from the 90 days before the 18th birthday may be used for an adult event if they were available in time.
- `prior_target_reported_90d`: a formal report of a target organism in the previous 90 days; resistance is not required.
- `prior_crgn_reported_90d`: formal CRE, CRPA or CRAB resistance evidence in the previous 90 days. It also defines the first-layer H90 alert.
- History uses the report verification time as a proxy for availability and excludes requests that belong to the current event.
- `MISSING` means that no qualifying record or result was found. It does not mean confirmed absence of history, medication or a procedure. Valid values for history and exposure predictors are `1` and `MISSING`; do not recode them to `1`/`0`.
- Laboratory categories must follow the original extraction rules: the latest result available within 48 hours before T0, classified against the reference range printed on its own report. This code does not derive categories from raw values and must not be given results obtained after the culture.

## 3. T0 specimen

| Code | Meaning |
|---|---|
| `RESP` | Respiratory |
| `BLOOD` | Blood |
| `URINE` | Urine |
| `OTHER` | Other specimen types |
| `MULTIPLE` | Two or more broad groups requested at T0 |
| `MISSING` | Could not be mapped |

- Only the requests made at the earliest time point T0 are used; requests added later within the 24-hour event are not.
- If any concurrent request cannot be mapped, the category is `MISSING`.
- Hospital T0 is the ordering time. MIMIC-IV uses the specimen chart time as a proxy, so the external mapping is not fully equivalent.
- The source-name dictionary is `results/stage07/t0_specimen_mapping.csv`. It lists hospital laboratory-system names (in Chinese, with English glosses) and MIMIC-IV names. The category distribution by cohort is in `results/stage07/t0_specimen_distribution.csv`.
- `code/attach_specimens.py` only joins an existing, frozen event-level mapping. It never reassigns a category on the basis of the outcome.

## 4. Score output

- `score` is the model probability without recalibration to the target setting. Use it as a ranking score, not as a validated individual probability of infection.
- `H90_flag` is the separate first-layer rule. Events with an H90 alert are still scored, but additional alerts are selected only from events without one.
- By default the final model scores the input. `--outer-fold 0..4` must only be used for the patients held out by that fold.

## 5. Columns required for evaluation

| Column | Rule |
|---|---|
| `episode_id` | Unique, non-empty event identifier (string) |
| `subject_id` | Non-empty, stable patient identifier |
| `t0` | Date-time with an explicit time zone, preferably ISO 8601 |
| `score` | Finite and strictly between 0 and 1 |
| `y` | `1` positive, `0` known non-positive, blank or `MISSING` unknown |
| `prior_crgn_reported_90d`, `length_of_stay`, `icu_status` | Used for H90 and the fixed clinical comparators |

- Unknown outcomes must stay in the candidate pool and use alert budget. Do not recode them to 0 or remove them first. Alerts on known non-positive events and on unknown events are reported separately.
- Times are handled in UTC, and text without a time zone is parsed as UTC. Hospital source data are usually in UTC+08:00; state the time zone explicitly, otherwise cooldown periods and cohort boundaries shift.

## 6. Columns required for training

- `event_id`, `patient_id`, `t0`, `split` (= `train`), `fold` and `y`.
- `fold` is the original patient-level fold (0–4); a patient never appears in two folds.
- Only the development period, 1 December 2023 to 31 December 2024 (Beijing time), may be used for training. Events with unknown `y` are not used for fitting but still receive outer-fold scores for the allocation evaluation.
- To reproduce the coefficients exactly, keep the original numeric text of the input CSV. Writing the floats out again (for example through Excel) changes a few trailing digits, which can move the liblinear stopping point. With the original text, all six fitted models reproduced the coefficients exactly.
