Aggregate results accompanying the D18 Supplementary Material

This aggregate archive accompanies the 22-table Supplementary Information.
No model fitting, scoring, bootstrap resampling or source-data reconstruction was rerun.
Original study estimates and interval versions are preserved.

DIRECTORY GUIDE
Original_v3/: every payload file from version 3, byte for byte, with its original paths,
README and manifest preserved inside this directory. References to S32–S50 in that
historical README use ORIGINAL numbering; consult table_crosswalk.csv.
Original_tables/S01/ through S51/: complete CSV exports of every physical table
under each originally numbered table, plus original paragraph notes. These CSVs
retain the precision reported in Word; they are not unrounded reanalysis outputs.
original_table_index.csv and original_supplement_structure.json preserve table
order and paragraph context. CSV physical-part numbers follow source order.
Core_views/: exact panels displayed in the reorganised supplement. S00 holds
the reporting checklist. core_view_provenance.json records original row and
column indices for numerical source panels; row indices are zero-based including
the header (the first data row is index 1). Joined panels list all source segments.
table_crosswalk.csv / methods_crosswalk.csv: original-to-current numbering.
ablation_all_budget_contrasts.csv: all original paired ablation comparisons.
original_methods_and_legends.txt: historical full methods and legends, with
ORIGINAL citations retained for provenance, not current manuscript references.

INTERPRETATION
Formal D18: 18 inputs, 51 encoded coefficients and intercept, final C=0.1.
Restricted training: separate 17-input/49-coefficient sensitivity, final C=10.
Original primary and first-round intervals use 1000 patient-cluster replicates;
later R1/R2, returning-extension and conditional-scenario intervals use 5000.
Slight differences in repeated baseline limits reflect run seeds/replicate counts;
they are not harmonised or substituted. Sharp bounds are not confidence intervals.
Original_v3/README.txt records seeds, endpoint definitions and source limitations.
The archive contains aggregate outputs, not patient-level data, bootstrap draws,
the formal model binary or a standalone private-source reconstruction pipeline.
NA is not zero; unknown is not negative. Do not add overlapping patient/specimen
cells or independently allocated populations to infer distinct expanded coverage.

REPRODUCTION AND CHECKSUMS
The complete formal D18 reported coefficient/preprocessing tables are in
Original_tables/S03/. Derived-model full-precision aggregate specifications already
present in version 3 remain in Original_v3/E/. Original_v3 manifests apply to paths
relative to Original_v3, unchanged. payload_manifest_v4.json hashes all v4 payload
files except itself. Source document SHA-256 values are in source_hashes.json.
