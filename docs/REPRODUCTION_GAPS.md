# Reproduction gaps and explicit exclusions

This document prevents “files have been collected” from being misreported as
“the complete study has been independently reproduced.” It does not call for
new scientific experiments or opening Test data.

| Item | Current state | Consequence / next non-experimental action |
| --- | --- | --- |
| 36 derived Development scenes | Included in separate Zenodo draft | NPZ bytes and hashes preserved; nine legacy files use the documented mask fallback |
| 36 Full LOBO checkpoints | Included in separate Zenodo draft | Exact fold map supplied; no single final-refit model exists |
| Three ablation checkpoints | Not saved by the original runner | Do not claim their weights are provided; recipes are preserved; no retraining during packaging |
| HPR checkpoint and normalization layer | Preserved in the older LOCAL draft, omitted from revision-2 public-upload ZIP | Team/institution permission confirmed; exact upstream rights/provenance still require review |
| HPR architecture factory | Dynamically imported `models.reg` / `models.preprocessing` from the prior CO2 inversion repository; not included | Identify authoritative code/version and lawful access before end-to-end use |
| Synthetic plume library `dataset.nc` | Excluded; not NASA OCO-3 source observations | Frozen hash is in the protocol; determine lawful source/access and required rows before promising raw-to-results reproduction |
| Level-B tensor cache and metadata | Existing locally but not included in this scope | Their sharing decision is separate from 32 x 32 scene release; WACQ runner currently expects them |
| Frozen earlier Development prediction tables | Not included | Cache builder uses sample/condition IDs, true emission and frozen HPR predictions from them; WGG loader also uses prior data/role manifests |
| Figure/table result CSV/JSON and prediction outputs | Deliberately excluded at author's request | The plotting scripts cannot recreate manuscript plots from this draft alone |
| Historical split manifests with other roles | Not included | No reserved Test identities or values are copied merely to satisfy an old loader |
| A4 fitted object | Configuration-only extraction included | It is not byte-identical to the full historical A4 artifact; original full-file hash checks must not be bypassed |
| Absolute filesystem paths | Preserved in historical source/configs | A separately tracked portability pass is required; changing paths does not change the original freeze |
| Historical Python/TF/Keras/CUDA versions | Not established by this staging task | Do not manufacture a version lock from the current packaging environment |
| Original tests | Included selectively, not executed | Several depend on original paths/data and cannot be claimed to pass in a fresh checkout |

## Minimal workflow dependency map (not executed)

1. Official NASA source granules -> frozen scene extraction -> 32 x 32 scenes.
   This draft directly supplies the existing scenes; it does not rerun extraction.
2. Scenes + lawful plume-library inputs + condition/seeding rules + frozen
   HPR predictions -> `build_wacqnet_levelb_cache.py` inputs/cache.
3. Cache + protocol -> `run_wacqnet_36_lobo.py` -> fold weights/predictions.
   Existing Full weights are supplied separately. No training is needed just
   to archive the existing study.
4. Frozen A4 object + authorized Development feature inputs ->
   `run_bti_worst_group_guard_hcp_development.py` -> Development interval outputs.
5. Existing or independently reproduced outputs -> summary/plot scripts.
   Output tables themselves are not part of the current public-data plan.

Do not silently replace missing dependencies, substitute a different HPR,
derive a new calibration rule, use Test rows to complete a missing table, or
disable original integrity checks. Resolve release/portability issues first.

Revision-2 license selection is resolved for author-controlled contributions.
See `DEPENDENCY_ACTIONS.md` for the remaining concrete tasks. None of those
runtime or scientific tasks was executed in this packaging revision.
