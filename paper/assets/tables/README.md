# Tables

保存由脚本导出或人工核验的表格数据。表中数值必须能追溯到 run ID 或源数据，不手工覆盖生成结果。

## Frozen Jev-PHM PU cohort

The following files were generated on 13 October 2026 from the saved inputs of
`src/outputs/jev-phm/run-20261013-pu-001`, without reading model predictions:

- `jev_cohort.csv`: the canonical 2,559 source recordings and their 32 physical
  bearing identities, original labels, declared class names, relative source
  filenames, operating-domain IDs, actual H5 lengths, metadata lengths, sampling
  rates, source channel counts, and selected zero-based channel index.
- `jev_unit_roles.csv`: the 224 bearing/protocol assignments for the primary split
  and six rotations, together with each bearing's original class and source-record
  count. The physical-unit strings are the saved `Label_Description` identities.
- `jev_protocol_20261013.json`: the frozen label order, probe options, full-coverage
  window plans, train/tune window selections, dependency revisions, and the public
  factory fields and explicit native split adapter used by the study. The code
  commit in `prepared_run_spec` identifies preparation of the inputs; later model
  and fitting commands are recorded with their own run artifacts.

`file_id` identifies the supplied processed-data H5 key; `source_filename` is the
relative PU source filename recorded in the frozen inventory. Reproduction must
select exactly these 2,559 records and retain their bearing assignments. A provider
snapshot containing additional recordings must not silently enlarge this cohort.
KA08 has 79 listed source records; every other bearing has 80. The tables preserve
the available cohort and do not infer absent filenames or exclusion causes from a
gap in a file sequence.

Independent raw-file checks recorded the unavailable
`KA08/N15_M01_F10_KA08_2.mat`: it exists locally but failed to parse under both
inspected SciPy versions and is absent from the supplied metadata/H5. The
historical cause is unknown. This independently verified exclusion is described
in the manuscript; the generated cohort table does not synthesize that record.

Actual H5 length controls the window boundaries. It differs from metadata length
for 1,517 listed recordings. The supplied lengths span 249,940–299,038 samples;
all saved nominal sampling rates are 64,000 Hz, all source arrays have three
channels, and the selected channel index is 2. These descriptors locate the
processed observations used by this run; they do not establish that a separately
downloaded or reprocessed signal has identical numerical values.

The protocol snapshot omits machine-private absolute paths, the GPU UUID, and
unrelated installed packages. The complete machine-local inputs and environment
remain in the ignored run directory. Native PHMFactory frozen manifests require
the upstream metadata checksum field; the study uses that existing contract and
adds no custom hash mechanism. The saved role table supplies the explicit
train/tune/cal/test assignment; the native validation partition combines tune and
calibration records. `tune_nll` in the window-selection snapshot is a development
selection score, not a held-out result.
