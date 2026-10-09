# Third-party material and dependencies

This is a packaging revision, not a new experiment or model release. The
authors' permission to release their own work does not relicense other
creators' work. Dependencies are not bundled merely because the source
imports them. No external model code, synthetic plume library, or external
model checkpoint has been newly added in this revision.

## NASA OCO-3 source observations

Original collection: OCO3_L2_Lite_FP Version 11r, NASA GES DISC.
Source DOI: <https://doi.org/10.5067/8U0VGVQC7HZG>.
Provider policy: <https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy>.

Retain the collection citation and original product, build, granule and
checksum metadata. The project authors are creators of the derived grids
and curators of a source subset, not creators of the satellite observations.
NASA source terms and any marked exceptions continue to apply. CC BY 4.0
on project contributions does not add attribution restrictions to source
observations that are otherwise public domain/CC0. No NASA endorsement is
implied. The separate NASA source ZIP has not been changed or relicensed.

## Earlier CO2 inversion implementation and HPR anchor

Upstream project: Joffrey Dumont Le Brazidec,
<https://github.com/JoffreyDumontLeBrazidec/co2-oco3-inv-dl>.
Archived code v1.0.2: <https://doi.org/10.5281/zenodo.14013176>.
Related data/weights record: <https://doi.org/10.5281/zenodo.12788520>.

The archived code record identifies CC BY 4.0, while the current GitHub
landing page identifies MIT. These are different version contexts. Neither
is sufficient on its own to establish the terms of every file in the local
historical checkout identified by `bdf7f24`. Exact version/file provenance
has not yet been matched. The related data/weights record is a provenance
lead, not a verified license or byte match for the local plume dataset.

The project-authored adapter dynamically requires `models.reg` and
`models.preprocessing`; those external implementations are not bundled here.
The HPR checkpoint and normalization layer remain in the earlier LOCAL draft
but are intentionally absent from this revision's public-upload weight ZIP.
Author/institution permission for the team's own contributions is confirmed;
upstream rights and exact-version attribution remain unresolved. Their
omission is not a claim that redistribution is legally prohibited.

## Software dependencies

NumPy, pandas, SciPy, scikit-learn, TensorFlow/Keras, h5py, PyYAML, matplotlib
and pytest are external dependencies, not sublicensed by this package's MIT
file. Consult the licenses of the exact versions actually installed. The
discovery requirements list is not a historical environment lock and the
package has not been runtime-validated in a clean environment.

## Scope of this review

This notice records source evidence and current release boundaries. A static
header/import scan is not an exhaustive provenance or legal audit. Existing
third-party notices, if any, take precedence for their corresponding
material. No license is granted for excluded files. No new external
Calibration/Test data or manuscript performance tables are included.
