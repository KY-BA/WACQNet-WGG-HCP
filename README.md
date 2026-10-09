# WACQNet + HPR-centered A4/WorstGroupGuard-HCP

**Research source and frozen-configuration archive; not a turnkey
reproduction environment.** This directory contains the existing manuscript
implementation, not a newly trained model. Scientific source files and
configurations retain their historical bytes; only release documentation
and repository housekeeping have been updated for GitHub upload.

Start with [reproduction gaps](docs/REPRODUCTION_GAPS.md),
[dependency actions](docs/DEPENDENCY_ACTIONS.md), and
[data/weight access](docs/DATA_AND_WEIGHTS.md). A public source archive is
not evidence that clean-environment runtime reproduction has passed.

## Scientific scope

- Point/risk track: fold-specific BTI-WACQNet WACQ_FULL on 36 Development
  background records, with three frozen ablation recipes.
- Conservative-interval track: A4/WorstGroupGuard-HCP centered on the frozen
  **HPR** estimate, not the WACQNet estimate.
- Repeated realizations are not independent satellite backgrounds.
- No all-36 final refit and no formal new external-calibration `q_hat` are
  supplied. This archive authorizes no access to reserved Calibration/Test data.

## Contents

| Directory | Role |
| --- | --- |
| `src/` | Existing WACQNet, A4 and group-calibration implementation; some helper modules have historical names |
| `scripts/` | Existing cache, Development training/evaluation and summary scripts |
| `configs/` | Byte-identical historical configuration files, including required configuration ancestry |
| `legacy_support/` | Existing earlier-project helper code/configs required by static imports |
| `protocol_archive/` | Frozen protocol records and withdrawals, kept as historical evidence |
| `parameters/` | A4 fitted parameter object and WGG mathematical rule, not figure source data |
| `manifests/` | Code checksums, scene IDs and exact checkpoint/fold mapping |
| `plotting/` | The existing manuscript Results plotting script; its result-data inputs are excluded |
| `tests/` | Selected original tests, archived but not run during packaging |
| `docs/` | Scope, excluded inputs, unresolved dependencies and release checklist |

The three data/checkpoint ZIPs are intended for **one combined Zenodo record**:
`NASA_OCO3_source_subset.zip`, `derived_backgrounds_v2.zip`, and
`wacqnet_weights_v2.zip`. They are not included in this code repository.
No verified Zenodo DOI or public repository URL has been supplied to this
README yet. Add actual stable identifiers after verification; do not infer
a DOI from a draft upload number. See `docs/DATA_AND_WEIGHTS.md`.

## Important: original paths and hashes

Original source and configuration files are deliberately byte-identical.
They retain historical drive paths such as `D:/CO2/_third` and `D:/CO2_third`.
These paths are not installation instructions for a new user. Some scripts
also resolve paths relative to their own location. **Do not execute defaults
without a separate path/permission review.**

The code/config distribution has not silently rewritten paths or disabled
integrity checks. A future portability-only patch must be versioned, preserve
all scientific parameters, document changed hashes, and be verified separately.
The root-anchored `.gitignore` excludes data/output directories at repository
root without hiding `legacy_support/src/data/` source modules. The
`.gitattributes` file disables automatic line-ending conversion so that
historical source/config hashes can remain byte-exact when using Git.
The sanitized scene manifest is an index, not a hash-equivalent replacement
for the original frozen manifest. The original combined protocol hash remains
historical evidence, not a checksum of this new repository.

The mathematical protocol archive retains its original historical heading;
the authoritative training configuration is version **1.0.2**. The withdrawal
records explain superseded protocol versions. Legacy internal model-name
strings are implementation identifiers, not an instruction to expand HPR
incorrectly in the manuscript.

## Reproduction status

The staging audit checked file hashes, NPZ schema, checkpoint-to-fold mapping,
static local Python imports, syntax, and common credential patterns. It did
not run training, model loading, inference, calibration, pytest, or plots.
Therefore this release cannot claim a successful runtime reproduction test.

Read `docs/REPRODUCTION_GAPS.md` before trying to run anything. In particular,
the author has deferred release of figure/table CSV/JSON performance data;
this decision is not bypassed by putting them on GitHub instead of Zenodo.
Those files are absent from this archive. Some older prediction tables also
serve as frozen inputs to cache/calibration scripts, and are absent too.

## Licenses and citation

The authors have confirmed coauthor/institution approval for open release
and selected **MIT for project-owned code/configuration** and **CC BY 4.0
for authorized derived data/model parameters and weights**. This revision
implements that choice without changing any scientific source file or
configuration. It is a packaging revision, not WACQNet algorithm v2.

See `LICENSE` (MIT), `LICENSES/CC-BY-4.0.txt`, `LICENSE_SCOPE.md`,
`license_scope_manifest.csv`, and `THIRD_PARTY_NOTICES.md`. The learned A4
parameter object uses CC BY 4.0; experimental configurations, including the
fixed WGG rule, use MIT. Third-party
components and NASA observations are not relicensed by these files. The
legacy-support directory contains project wrappers/helpers; the dynamically
imported external HPR implementations are not supplied or licensed here.

The companion revision-2 weight ZIP deliberately omits the HPR anchor and
normalization binaries while their upstream provenance remains unresolved.
Do not upload the earlier mixed weight draft by mistake. The 36 WACQ_FULL
checkpoint files remain unchanged. Read `docs/DEPENDENCY_ACTIONS.md` for
specific remaining work; choosing a license does not fix missing runtime
inputs or establish independent reproduction.

NASA source collection: <https://doi.org/10.5067/8U0VGVQC7HZG>.
The separate source-data package retains exact product/build/granule identities.
After publication, tag the manuscript code version, record the commit SHA,
link the combined data/weight record, and preferably archive the code release
with a persistent identifier. No DOI is assigned by this preparation process.
