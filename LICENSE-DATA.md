# Licence for models, results and documentation

Except where noted otherwise, the contents of the following folders are licensed under the Creative Commons Attribution 4.0 International licence (CC BY 4.0): `models/`, `configuration/`, `examples/`, `results/`, `aggregate_results/`, `figures/`, `audit/` and `documentation/`, together with `README.md` and `PACKAGE_MANIFEST.json`.

Licence summary: https://creativecommons.org/licenses/by/4.0/
Legal code: https://creativecommons.org/licenses/by/4.0/legalcode

The code in `code/` and `original_scripts/` is licensed under the MIT License; see [LICENSE](LICENSE).

Notes:

- Results derived from MIMIC-IV are aggregate statistics. MIMIC-IV data themselves are not redistributed here and remain subject to the PhysioNet credentialed data-use agreement.
- The model files are Python objects saved with joblib. Loading such a file can execute code, so only load files whose SHA-256 matches `configuration/model_registry.json`.
