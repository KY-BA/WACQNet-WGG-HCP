# Data and weight access

The release plan uses one combined Zenodo record with three archive files.
Its verified DOI is not yet recorded here. A draft upload number is not a
substitute for a verified published DOI. Add the real DOI/link after the
author has checked the record and access status.

| Archive in the combined record | Contents | Rights boundary |
| --- | --- | --- |
| `NASA_OCO3_source_subset.zip` | 36 unchanged NASA OCO3_L2_Lite_FP v11r daily granules and provenance | Original NASA terms; not relicensed by this software license |
| `derived_backgrounds_v2.zip` | 36 existing 32 x 32 Development scene records | CC BY 4.0 for authors' derived contributions; NASA source rights retained |
| `wacqnet_weights_v2.zip` | 36 WACQ_FULL LOBO fold checkpoints and fold mapping | CC BY 4.0 for authors' authorized weight contributions |

The ZIP packaging suffix is not a new algorithm version. The 36 scene
records are not a claim of 36 independent geographical sites. Repeated
Level-B realizations are not independent satellite backgrounds.

The combined record's `README.md` and `LICENSE_MAPPING.md` explain the
mixed-license scopes. NASA's original collection must still be cited:
<https://doi.org/10.5067/8U0VGVQC7HZG>.

## Important exclusions

The HPR checkpoint and normalization layer, upstream model factories,
synthetic plume library, Level-B caches, and certain frozen prior
Development input tables are not bundled. Some require source/rights
clarification and some are outside the author's present release scope.

The author has deferred release of manuscript figure/table performance
CSV/JSON, predictions, and training histories. They are not placed in
GitHub to bypass that decision. This repository contains metadata CSV/JSON
and configuration/parameter objects, which are not the withheld results.

There is no all-36 final-refit model, saved ablation checkpoints, newly fitted
formal external `q_hat`, or new reserved Calibration/Test data in this release.
The existing source/configuration files alone do not yield complete
raw-to-results reproduction; read `REPRODUCTION_GAPS.md` and
`DEPENDENCY_ACTIONS.md` before running any scripts.
