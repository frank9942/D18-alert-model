Supplementary Data 1, version 3: exploratory analyses of frozen D18

Accompanies manuscript v22 and Supplementary Information v10 (Methods S17-S24 / Tables S32-S50).
The formal D18 model and original primary cohort estimates remain unchanged.
This archive contains aggregate tables, not individual patient data or bootstrap draws.

FIRST ROUND (original aggregate files preserved byte for byte)
A/: Development-derived specimen ordering and specimen plus ICU/LOS rules,
    all budgets, globally allocated specimen summaries and all-known-outcome
    specimen-stratified discrimination (not H90-unflagged discrimination).
B/: INITIAL returning-candidate recovery and original-cohort history checks.
    Missing expanded outcomes in these historical tables were later addressed
    in Round2/B2_returning. They must not be read as the final analysis status.
C/: Restricted respiratory-flora hypothetical non-positive labels and performance.
    This is not confirmation of negative status or full mapping harmonisation.
D/: Pathogens by actual positive specimen; cross-tabulation with T0 predictor.
    Cells overlap and must not be added to obtain distinct patient totals.
E/: Separate H90-unflagged training sensitivity, with 17 raw inputs after
    removal of constant resistance history, 49 coefficients and intercept, C=10.
    This is not the formal D18 model or a replacement for it. Original encoded
    source-language labels are retained for faithful model reconstruction.

SECOND ROUND (Round2/; complete combined CSVs, without per-cohort duplicates)
A2_rules/: Fixed R1 and R2, earlier specimen combination, ICU+LOS and D18.
    S=1 for RESP or MULTIPLE, equally prioritised; MULTIPLE need not include RESP.
    R1: ICU descending, S descending, pre-T0 LOS descending, fixed hash ascending.
    R2: S descending, ICU descending, pre-T0 LOS descending, fixed hash ascending.
    Nonfinite LOS is last. Both rules selected identical event sets at every
    reported cohort/budget, so they are not independent replications.
    S44-S45 summarise results. Full intervals and allocations are retained here.
B2_returning/: Completed returning-only and globally reranked expanded evaluation.
    6096/6097 adult candidates evaluated; one 2026 event excluded for a 32-second
    T0 discrepancy. The original population is retained as a reference. Expanded
    counts are not sums of separately allocated original and returning results.
    Development overlap is explicitly recorded: this is a sensitivity analysis,
    not independent new-patient validation. Flow, overlap, source completeness,
    case mix, both prediction populations, strategy metrics and histories are
    supplied. S46-S48 contain selected summaries and all-budget coverage contrasts.
C2_bounds/: O=original outcome mapping; F=restricted respiratory-flora hypothesis.
    O has 78,219 unknowns; F locks 9721 assumed non-positive and leaves 68,498.
    Sharp bounds hold one common completed label vector across strategies,
    recompute positive-patient H90 baselines and earliest positive events, and
    keep scores, H90 flags and allocation fixed. Bounds are not confidence
    intervals. Uniform scenarios need not attain their limits. S49-S50 summarise
    bounds at all budgets and the 2% uniform scenarios; all-budget outputs are here.

ENDPOINTS AND FORMATS
budget is percent (1, 2 or 5) of H90-unflagged candidate events, including unknowns;
extra quota is ceil(N*budget/100). model=D18; icu_los=ICU+LOS;
icu_expected=exact expected ICU-priority coverage. Decimal expected counts are valid.
new_positive_patients = additional cumulative positive patients beyond H90.
baseline_positive_patients = patients with an H90-alerted positive event in scope.
first_add = added capture of the earliest known-positive event, not cumulative gain.
first_h90 = earliest known-positive events flagged by H90.
positive/nonpositive/unknown event counts are not distinct patient counts.
ppv_known = positive alerted events / all known-outcome alerted events.
reviews_per_new_positive_patient includes all extra alerts, repeated and unknown.
Zero additional patients can imply infinite workload ratios; NA is not zero.
Mappings in original source-language dictionary files are retained unchanged;
English scope/source-gap companion files and the English supplement explain them.
CSV files are UTF-8 (some have BOM), with original numeric precision preserved.

INTERVAL VERSIONS
First round: 1000 paired patient-cluster draws. A seeds 2026101601-2026101604;
C seed 2026101621; E seeds 2026101640-2026101643 (development, 2025, 2026, MIMIC).
Second round: 5000 paired draws. A2 seeds 2026101601-2026101604.
B2 seeds for 2025/2026: original 2026101761/2026101762,
returning 2026101861/2026101862, expanded 2026101661/2026101662.
C2 seed 2026101621 with common draws across mappings.
All intervals are pointwise percentile intervals, not multiplicity adjusted,
conditional on frozen scores, and exclude fitting/algorithm-selection uncertainty.
Seeds and replicate counts differ between runs; slightly different baseline limits
are not new model fits. Original primary intervals remain in the manuscript.
All second-round reported interval rows had 5000 valid replicates.

VERIFICATION AND LIMITS
Input package manifests, aggregate arithmetic, label transitions, flow and bounds
consistency were checked, as was transfer into the manuscript/tables/figures.
The second-round audit checked 87 manifest files, 12,888 numerical relationships,
5232 interval rows and 24 bound rows. This is aggregate verification, not a rerun
of the private source reconstruction, scoring or patient-bootstrap computation.
The execution package includes source-dependent code; this aggregate archive does
not contain a standalone analysis pipeline, the formal model binary or private data.
Clinical infection/colonisation adjudication was not undertaken.
aggregate_file_manifest.json records current archive payload hashes and byte sizes.

VERSION 3 METADATA UPDATE
Only this README and its checksum entry have changed since version 2. All aggregate results, model specifications and other archive payloads are byte-identical.
