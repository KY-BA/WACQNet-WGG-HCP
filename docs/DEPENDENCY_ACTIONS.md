# Reproduction completeness: remaining actions

License selection is resolved for team-owned contributions. Runtime
reproducibility is a separate matter and has NOT been established by this
packaging operation. Do not call this an end-to-end or one-command release.

| Dependency | Status and evidence | Next safe action |
| --- | --- | --- |
| 36 Development background NPZs | Included, original array hashes preserved | Use the separate licensed background ZIP and documented legacy mask fallback |
| 36 WACQ_FULL fold checkpoints | Included, existing fold mapping preserved | Preserve exact architecture/protocol association; do not present one fold as a final refit |
| Third-party HPR model factories | Dynamic imports in `legacy_support/src/models/backbones.py`; upstream files absent | Match local checkout files to an authoritative archived version and applicable notices before redistribution |
| HPR checkpoint and normalization layer | Withheld from the new public-upload package; hashes retained as dependency identifiers | Complete upstream provenance review; author/school consent alone does not settle third-party terms |
| Synthetic plume `dataset.nc` | Not NASA background data; not bundled | Establish exact source/version/license and byte identity with a lawful distribution |
| Level-B cache and row metadata | Not bundled | Make a separate sharing decision; 32 x 32 backgrounds alone are not the model input tensors |
| Frozen prior Development predictions | Required by cache-builder `load_rows`; not bundled | These contain result-derived inputs. Do not silently add them under a metadata label; obtain an explicit release decision |
| Legacy historical manifests | Some original loaders need broader manifests | Design a Development-only portable loader without copying reserved Test rows; version the adapter separately |
| A4 fitted object | Parameter-only extract included under CC BY 4.0 | Retain original full-file hash as historical evidence; never bypass its mismatch with the extract |
| Absolute paths | Historical paths preserved in source/configs | Version a portability-only patch; do not overwrite the frozen code or silently change scientific parameters |
| Environment | Unpinned discovery list only | Recover historical Python/TF/Keras/CUDA versions from existing records before claiming an exact lock |
| Figure/table CSV/JSON and predictions | Excluded by author decision | Keep that exclusion explicit; bundled plot scripts cannot recreate figures from this package alone |
| Ablation checkpoints / all-36 refit | Not saved / not performed | State this, rather than create new experiments just to fill an archive |

## Verification levels

1. **Completed in packaging:** copy/hash verification, ZIP CRC/payload checks,
   Python AST parsing, common credential-pattern checks, and dependency
   documentation. AST parsing does not import or execute project modules.
2. **Not completed:** clean-environment imports, model loading, training,
   inference, calibration, original pytest suite, or raw-to-figure replication.
3. **Not authorized by this release step:** opening new Calibration/Test,
   fitting a new `q_hat`, changing algorithms, publishing files remotely, or
   adding the withheld performance outputs.

The current milestone is a licensed, checksum-verified archive of specified
assets, not a verified complete reproduction environment.
