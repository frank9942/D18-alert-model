# Model card: D18

**Purpose.** At culture ordering for adult inpatients, the 90-day resistance-history alert (H90) is kept. D18 ranks the remaining culture events to show how many additional patients with CRE, CRPA or CRAB a model-based alert would cover, and at what additional review burden.

**Population and outcome.** A frozen cohort of culture events in patients aged 18 or over at T0. The outcome is laboratory detection of a target resistant organism supported by report evidence. It is not clinical infection and is not an indication to change antimicrobial treatment.

**Model.** L2-penalised logistic regression on 18 raw predictors, with numerical imputation and standardisation and fixed-domain one-hot encoding; final C = 0.1. The 18 raw predictors expand to 51 encoded coefficients plus an intercept, all exported with the numerical scales, imputation values and category domains (`configuration/encoded_coefficients.csv`, `configuration/preprocessing_parameters.json`). SHA-256 of the final model: `8f27b6f6bf656fa760f532624da3e12f8a2dde8f2c09ec5d08474d5a80db9d09`.

**Development.** 1 December 2023 to 31 December 2024; patient-level nested cross-validation with five outer and four inner folds; final C selected with five folds. Events with unknown outcomes were not used in supervised fitting.

**Evaluation.**
- Hospital cohorts: 2025 (full year) and 1 January to 15 September 2026.
- External cohort: adapted MIMIC-IV.
- Patient separation between cohorts follows the frozen adult protocol.
- The later cohorts had been examined during earlier research iterations. D18 was chosen as the reporting model after exploratory improvement, so the evaluation is not claimed to be fully untouched validation.

**Alert allocation.**
- Events with an H90 alert are alerted first.
- The additional budget is K = ceiling(p × number of events without an H90 alert), with p = 1%, 2% or 5%. The denominator is events, not patients.
- Events with unknown outcomes use budget.
- Additional alerts go to the highest scores. Ties are broken by a fixed hash of the event ID in hospital cohorts and by ID order in MIMIC-IV.
- The ICU-priority comparator reports the expected value under uniform allocation within the ICU stratum, so fractional counts are expected.

**Additional positive patients.** Distinct patients with a positive event selected by an additional alert, excluding patients covered by a positive H90 alert at any time in the evaluation period. This cumulative count does not imply earlier detection. Capture of the first known positive event is reported separately; earlier events with unknown outcomes could change which positive event truly came first, which the unknown-outcome analyses address.

**Cost.** Additional alerts needed per additional positive patient. Alerts on known non-positive events and on unknown events are kept separate; unknown alerts are not called false alerts.

**Fixed threshold versus fixed budget.** These answer different questions.
- The D18 threshold that gave a 2% budget in development is 0.25359903645222276. In other cohorts the realised alert volume at this threshold can be well above or below 2%.
- Repeat-alert suppression applies only to the model's additional alerts; H90 alerts are kept.
- Suppression runs in chronological order for each patient. Only alerts actually issued restart the cooldown, and unused budget is not carried forward.

**Limitations.**
- The probabilities have not been recalibrated for the target setting.
- No clinically acceptable alert cost was prespecified or tested in a prospective workflow.
- Unknown outcomes in MIMIC-IV and limited species-level evidence (especially for CRAB) remain.
- The release supports research reproduction, the article and offline demonstration. Matching hashes show that files are identical; they do not remove bias or show that clinical use is safe.
