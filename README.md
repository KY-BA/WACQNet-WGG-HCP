# WACQNet + HPR-centered A4/WorstGroupGuard-HCP

**Research source and frozen-configuration archive; not a turnkey
reproduction environment.** This repository contains the existing manuscript
implementation, not a newly trained model. Scientific source files and
configurations retain their historical bytes; release documentation and
repository housekeeping are maintained separately.

Repository: [KY-BA/WACQNet-WGG-HCP](https://github.com/KY-BA/WACQNet-WGG-HCP).

Start with [reproduction gaps](docs/REPRODUCTION_GAPS.md),
[dependency actions](docs/DEPENDENCY_ACTIONS.md), and
[data/weight documentation](docs/DATA_AND_WEIGHTS.md). A public source archive
is not evidence that clean-environment runtime reproduction has passed.

## Scientific scope

- Point/risk track: fold-specific BTI-WACQNet WACQ_FULL on 36 Development
  background records, with three frozen ablation recipes.
- Conservative-interval track: A4/WorstGroupGuard-HCP centered on the frozen
  **HPR** estimate, not the WACQNet estimate.
- Repeated realizations are not independent satellite backgrounds.
- The 36 Development scene records must not be interpreted as 36 independent
  geographical sites or fresh external validation backgrounds.
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

## Associated data: ScienceDB submission

The following three archives have been uploaded together to one ScienceDB
dataset record and submitted for repository review. They are not included
in this code repository.

| Archive | Contents |
| --- | --- |
| `NASA_OCO3_source_subset.zip` | 36 unchanged NASA OCO3_L2_Lite_FP v11r daily granules, official metadata snapshots and scene-to-granule provenance mappings |
| `derived_backgrounds_v2.zip` | 36 existing Development scene containers with derived 32 × 32 backgrounds, sidecar metadata, schemas and source mappings |
| `wacqnet_weights_v2.zip` | 36 saved WACQ_FULL Development LOBO checkpoints and exact checkpoint-to-fold mappings |

The submitted ScienceDB record also contains `README.md`,
`LICENSE_MAPPING.md`, and `SHA256SUMS.txt`.

### Submission status and identifiers

**Status recorded on 2026-10-10: submitted for repository review.
Public release and identifier registration are pending.**

ScienceDB provided the following identifiers, both displayed as
**unregistered** at the time of submission:

- Dataset DOI:
  [10.57760/sciencedb.0157o](https://doi.org/10.57760/sciencedb.0157o)
- Dataset CSTR:
  [31253.11.sciencedb.0157o](https://cstr.cn/31253.11.sciencedb.0157o)

The final character in both identifiers is the lowercase letter `o`.

These identifiers refer to the associated **dataset**, not to this code
repository. Their presence here does not establish that the record has
passed review, that the identifiers resolve publicly, or that the files
are already available for public download.

This status statement should be updated after repository review,
identifier registration and public file access have been confirmed.

### Scope and historical packaging references

The `v2` suffix in the background and weight filenames denotes packaging
revision 2, not a new WACQNet algorithm version. The three ZIPs retain their
previously prepared bytes.

ScienceDB replaces the earlier planned Zenodo hosting arrangement for this
combined data/weight deposit. Earlier Zenodo references in archival release
documents or archive-internal directory names describe that historical plan.
They do not establish a published Zenodo record or require separate Zenodo
DOIs. This hosting update does not change scientific protocols, archive
contents or component-specific rights.

The deposited assets alone do not constitute a complete end-to-end
reproduction package. The exclusions and unresolved dependencies described
below remain applicable.

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
historical evidence, not a checksum of this repository.

The mathematical protocol archive retains its original historical heading;
the authoritative training configuration is version **1.0.2**. The withdrawal
records explain superseded protocol versions. Legacy internal model-name
strings are implementation identifiers, not an instruction to expand HPR
incorrectly in the manuscript.

Repository checksums and license-scope hash entries must track the exact
distributed file bytes. Documentation updates require corresponding checksum
updates; they do not justify changing historical scientific source files,
configuration files or frozen protocol evidence.

## Reproduction status

The staging audit checked file hashes, NPZ schema, checkpoint-to-fold mapping,
static local Python imports, syntax, and common credential patterns. It did
not run training, model loading, inference, calibration, pytest, or plots.
Therefore this release cannot claim a successful runtime reproduction test.

Read `docs/REPRODUCTION_GAPS.md` before trying to run anything. In particular:

- The author has deferred release of figure/table CSV/JSON performance data.
  Those files are absent from this code repository and the associated
  ScienceDB deposit. Changing the hosting platform does not change that scope.
- Some older prediction tables also serve as frozen inputs to
  cache/calibration scripts and are absent.
- Synthetic plume data, Level-B caches, and some frozen prior Development
  inputs are not included in the associated data deposit.
- The HPR anchor and normalization binaries remain excluded pending exact
  upstream-version and provenance clarification.
- The WACQNet checkpoints are weights-only files, not standalone serialized
  models. Their use requires the exact architecture, preprocessing,
  configuration and HPR-dependent inputs.
- No ablation checkpoints or all-36-background final-refit model are supplied.
- WGG is a calibration rule with frozen parameters, not an additional neural
  checkpoint in the weight archive.

Included CSV/JSON provenance, schema and fold metadata must not be confused
with the excluded manuscript figure/table performance data.

Neither repository availability nor completion of data upload establishes
independent scientific validation or successful raw-to-results reproduction.

## Licenses

The authors have confirmed coauthor/institution approval for open release
and selected **MIT for project-owned code/configuration** and **CC BY 4.0
for authorized derived data/model parameters and weights**.

These release-documentation updates do not change any scientific source file
or configuration. They are packaging and availability updates, not a new
scientific algorithm version.

See:

- [LICENSE](LICENSE)
- [CC BY 4.0 license text](LICENSES/CC-BY-4.0.txt)
- [License scope](LICENSE_SCOPE.md)
- [File-level license-scope manifest](license_scope_manifest.csv)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

The learned A4 parameter object uses CC BY 4.0; experimental configurations,
including the fixed WGG rule, use MIT.

Third-party components and NASA observations are not relicensed by these
files. The `legacy_support/` directory contains project wrappers/helpers;
the dynamically imported external HPR implementations are not supplied or
licensed here.

For the associated ScienceDB deposit, CC BY 4.0 applies only to the authors'
authorized contributions to derived Development backgrounds, WACQNet fold
weights, curation and documentation. Unchanged NASA observations and source
metadata retain their original source terms, including any marked exceptions.
No additional restrictions are imposed on public-domain source material or
facts. Consult the deposit's `LICENSE_MAPPING.md` for the component-specific
scope.

The companion revision-2 weight ZIP deliberately omits the HPR anchor and
normalization binaries while their upstream provenance remains unresolved.
Do not substitute or upload the earlier mixed weight draft. The 36 WACQ_FULL
checkpoint files remain unchanged.

Read `docs/DEPENDENCY_ACTIONS.md` for specific remaining work. Choosing a
license does not fix missing runtime inputs or establish independent
reproduction.

## Source attribution and citation

### Original NASA observations

Original source collection:

OCO-2/OCO-3 Science Team, Vivienne Payne, Abhishek Chatterjee (2024).
*OCO-3 Level 2 bias-corrected XCO2 and other select fields from the
full-physics retrieval aggregated as daily files, Retrospective processing
V11r*. Version 11r. Goddard Earth Sciences Data and Information Services
Center (GES DISC).

Source DOI:
[10.5067/8U0VGVQC7HZG](https://doi.org/10.5067/8U0VGVQC7HZG).

The source-data package retains exact product, processing-build and granule
identities. These are NASA Level-2 processed retrieval products, not Level-0
instrument measurements. The project authors curate the subset and produce
derived contributions; they do not claim to have produced the original
NASA observations. No endorsement by NASA is implied.

Please retain the original NASA dataset citation when using these materials.
A ScienceDB dataset identifier does not replace the original NASA source DOI.

### Associated dataset and code version

After the ScienceDB record is published, use its final dataset citation,
actual version and registered identifier when citing the curated data and
weight deposit.

For code reuse, identify the exact Git commit or release tag used. The
ScienceDB dataset DOI is not a software DOI for this repository.

A manuscript-associated code release should be tagged, its commit SHA
recorded, and preferably archived separately with a persistent identifier.
No software DOI is claimed in this README.
