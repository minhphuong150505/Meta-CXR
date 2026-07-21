# Safe notebook handling

MIMIC-CXR is PhysioNet credentialed-access data. The DUA forbids
redistribution. Notebooks are the easiest way to breach it by accident: the
source looks harmless while the *executed outputs* silently embed patient
identifiers, report text and `findings_clean` values.

`.gitignore` covers the two known notebooks, but one `git add -f`, a rename, or
a newly created notebook defeats it. `scripts/check_notebook_privacy.py` is the
backstop.

## Usage

```bash
# What the pre-commit hook runs
python scripts/check_notebook_privacy.py --staged

# One file
python scripts/check_notebook_privacy.py --path notebooks/foo.ipynb

# Everything currently tracked
python scripts/check_notebook_privacy.py --all-tracked

# Write a cleaned copy (never edits the original without an explicit flag)
python scripts/check_notebook_privacy.py --sanitize dirty.ipynb --output clean.ipynb
```

Exit codes: `0` clean, `1` violations found, `2` usage or IO error.

Install the hook once per checkout:

```bash
pip install pre-commit
pre-commit install
```

## What it flags

| Check | Where | Rationale |
|---|---|---|
| `executed_output_present` | outputs | Any output at all. This is the blunt rule that catches everything else. |
| `execution_count_set` | cell | Marks a notebook that was run against real data. |
| `identifier_key:*` | outputs | `subject_id`, `study_id`, `dicom_id`, `patient_id` as bare words — a DataFrame print renders them as inline column headers. |
| `identifier_value_with_context` | outputs | 8-digit `1…`/`5…` numbers, **only** when the same output also carries an identifier key or a MIMIC marker. |
| `mimic_data_path` | outputs + source | `files/p1X/pNNNNNNNN/sNNNNNNN` layout. |
| `mimic_dataset_marker` | outputs | Literal `MIMIC-CXR` / `MIMIC-IV`. |
| `private_storage_uri` | outputs | `gs://`, `s3://`, `azure://`, `abfss://`. |
| credential kinds | outputs + source | HF tokens, AWS keys, GitHub tokens, signed URLs, private keys, bearer tokens, `api_key="…"`. |

### Two deliberate design choices

**Bare 8-digit numbers are not flagged on shape alone.** The tracked notebooks
carry legitimate Kaggle `datasetId`/`sourceId` values in exactly the same
numeric range. Flagging those would produce constant false positives and train
people to bypass the checker, which is worse than not having one. A number
counts only alongside an identifier key or a MIMIC marker in the same output.

**Cell source is not scanned for identifier keys.** `df["subject_id"]` in code
is a column reference, not a leak. Source *is* scanned for credentials and
MIMIC data paths, which are leaks whether or not the cell was ever run.

## Reporting never reproduces the value

Violations print a redacted form (`10****32`). Printing the identifier in order
to warn about the identifier would put it in your terminal scrollback, your CI
log and your shell history — turning the checker into another leak. Reports
give file, cell index, violation kind and location; that is enough to find the
cell without restating its contents.

## Sanitizing

`--sanitize` strips outputs and resets `execution_count`. It **does not touch
cell source**, so your code survives intact.

- It refuses to run without `--output`, unless you pass `--overwrite-in-place`.
- It refuses `--overwrite-in-place` on a **tracked** notebook outright: silently
  rewriting a file that is already in history is not something a privacy tool
  should do on your behalf.

Sanitize to a new path, review it, then stage the reviewed copy.

## The known private notebook

`preporcessing/kltn-data-preprocessing.ipynb` contains real MIMIC identifiers in
its executed outputs. It is git-ignored and untracked, and it must stay that
way. It has deliberately **not** been sanitized, deleted or modified — it is
your research artifact and it is safe where it is.

`tests/test_notebook_privacy.py` asserts on every run that the file is still
both ignored and untracked, without opening it.

## Fixtures

Everything under `tests/fixtures/notebooks/` is synthetic. No content was
copied from a real notebook and every identifier-shaped number was invented for
the fixture. Copying a real executed notebook in to test the leak detector
would commit the leak it is meant to prevent.
