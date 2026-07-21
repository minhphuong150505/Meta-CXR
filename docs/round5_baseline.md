# Round 5 baseline — measured at `05728bc`

Branch: `refactor/medgemma-real-runtime-validation` (from
`refactor/stage2-real-medgemma-runtime` @ `05728bc`)
Working tree clean, remote in sync, at time of measurement.
Interpreter: `/home/phuong/venv/bin/python`, Python 3.12.3.

## Results

| Check | Result |
|---|---|
| `python -m compileall .` | exit 0 |
| `pytest -q` | **275 passed, 1 failed** (276 collected) |
| `ruff check .` | 426 errors (372 auto-fixable) |
| `ruff format --check .` | 70 files would be reformatted, 26 already formatted |

`ruff` counts match the round-4 record (426). Test count grew 250 → 276; the
26 new tests came from `05728bc`, the notebook privacy guard.

## The brief's expectation was not met

The round-5 brief states "Kỳ vọng tối thiểu: 276 passed, 0 failed". The
starting commit does **not** satisfy that: 1 of the 276 fails, and it has never
passed. This is pre-existing and was not caused by any change in this round —
no source file was modified before the baseline ran.

### Failing test

`tests/test_notebook_privacy.py::test_all_tracked_notebooks_pass_the_guard`

The test asserts `check_notebook_privacy.py --all-tracked` exits 0. It cannot.
`--all-tracked` resolves targets via `tracked_notebooks()`
(`scripts/check_notebook_privacy.py:347`), which enumerates *every* tracked
`.ipynb` — including the three deliberately-dirty fixtures added by the same
commit to prove the guard detects violations:

```
tests/fixtures/notebooks/credential_like.ipynb
tests/fixtures/notebooks/executed_output.ipynb
tests/fixtures/notebooks/synthetic_identifier.ipynb
```

The guard correctly flags its own test fixtures, so `--all-tracked` always
exits non-zero and the test always fails.

**This is not a data leak.** The fixtures are synthetic by design — commit
`b03fbc9` replaced a MIMIC-shaped subject id in one of them precisely to keep
them free of real identifiers. The defect is in target enumeration, not in
detection.

## Fixed — baseline is now green

Resolved in `test: keep dirty notebook fixtures out of tracked notebook scan`.
The fix is in the fixtures and the test, **not** in the guard.

The five fixtures were renamed `*.ipynb` → `*.ipynb.fixture`:

```
tests/fixtures/notebooks/clean.ipynb.fixture
tests/fixtures/notebooks/credential_like.ipynb.fixture
tests/fixtures/notebooks/executed_output.ipynb.fixture
tests/fixtures/notebooks/kaggle_ids.ipynb.fixture
tests/fixtures/notebooks/synthetic_identifier.ipynb.fixture
```

`tests/test_notebook_privacy.py` gained a `notebook(tmp_path, name)` helper that
writes a fixture out as a real `.ipynb` under `tmp_path`; every assertion about
output detection, execution counts, identifier detection, redaction, credential
detection and sanitize is unchanged and still runs against an ordinary notebook
on disk. The clean fixtures go through the same helper, so all five load one
way.

### The guard was not weakened

- `scripts/check_notebook_privacy.py` is **unmodified**.
- `--all-tracked` keeps its exact meaning: it still checks *every* `.ipynb`
  tracked by Git, via `git ls-files "*.ipynb"`. It now scans the three real
  notebooks under `notebooks/` and reports them clean.
- No directory exclusion was added. Excluding `tests/` or `tests/fixtures/`
  would hide a genuinely leaky notebook parked there later.
- No filename-based forgiveness was added. The detector never learns to ignore
  a violation because of where it lives.
- Intentionally-dirty fixtures are simply no longer stored as tracked
  notebooks, so they are outside the notebook namespace rather than exempted
  from it. A future `.ipynb` added anywhere — `tests/` included — is still
  scanned.
- Fixtures remain fully synthetic; no real MIMIC identifier was introduced.

### Post-fix measurements

| Check | Result |
|---|---|
| `pytest -q tests/test_notebook_privacy.py` | 26 passed |
| `check_notebook_privacy.py --all-tracked` | exit 0 (CLEAN) |
| `pytest -q` | **276 passed, 0 failed, 0 skipped** |
| `python -m compileall scripts tests` | exit 0 |
| `ruff check` on the two touched files | All checks passed |

`ruff format --check` still reports both touched files as needing reformatting.
That is pre-existing — verified by running the check against their `HEAD`
versions, which report identically. Repo-wide counts are unchanged at 426 ruff
errors and 70 unformatted files; this commit neither added to them nor fixed
them, which was out of scope.

The private notebook `preporcessing/kltn-data-preprocessing.ipynb` was not
opened, read, or modified, and remains git-ignored (`.gitignore:62`) and
untracked.
