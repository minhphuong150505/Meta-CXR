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

**Not fixed in this round.** The fix is small (exclude the fixture directory
from `--all-tracked`, or assert the expected non-zero exit), but it is outside
the runtime-validation scope this round was authorised for, and the brief
directs stopping at the failing step rather than widening. Flagged for a
decision.
