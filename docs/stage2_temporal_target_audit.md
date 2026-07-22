# Stage-2 Temporal-Target Audit

## The problem
MIMIC-CXR FINDINGS frequently contain interval-change language — "unchanged",
"stable", "compared to the prior study", "new since …". Stage-2 inputs (a single
current study, native image or Q-Former features) carry **no prior**. Training on
those targets teaches the model to emit temporal claims it cannot ground, i.e.
confident temporal hallucination. This is audit finding #15.

## How to measure it
`scripts/audit_temporal_targets.py` scans a records JSONL and reports:

- `total_reports`
- `temporal_reports` — targets containing an interval-change cue
- `reports_with_prior_context` — records where `prior_available` is true
- `temporal_but_no_prior` — the harmful intersection
- `temporal_but_no_prior_rate`

```bash
# Real counts (restricted data; run on a private box):
python scripts/audit_temporal_targets.py \
  --input outputs/validation_records.jsonl \
  --output outputs/temporal_target_audit.json
```

The detector is the lexical regex in `stage2/prompts/policies.py::contains_temporal_language`
(hard cues like "compared to prior", plus soft cues like "stable"/"improved" matched
at clause level). It is a heuristic, **not** a clinical judgement.

### Counts on real MIMIC-CXR — NOT YET RUN
No real count is reported here. The split records this repo consumes do not currently
carry a `prior_available` flag (MIMIC-CXR-JPG metadata in this project has no prior
linkage), so both the numerator's prior side and the guard rely on that field being
threaded first. Running the script on synthetic fixtures is illustrative only.

## Policy options (`stage2/prompts/policies.py::apply_temporal_target_policy`)
| Policy | Effect on a temporal target with no prior |
|---|---|
| `keep` (default) | unchanged — backward compatible; the harmful signal remains |
| `remove_temporal_clauses` | drops whole sentences containing a temporal cue (lossy heuristic) |
| `exclude_sample` | drops the sample from training entirely |
| `require_prior_context` | signals the caller to supply prior context or drop the sample |

`keep` is the default **and is not silently overridden**. Any other choice is opt-in
via `temporal.target_policy` in the prompt config, and the choice is recorded in the
adapter manifest (`prompt.temporal_target_policy`).

Independently, the prompt itself always forbids temporal wording when no prior is
present (`forbid_comparison_without_prior`, on by default): the `NO_PRIOR_GUARD` line
is added to the instruction whenever `prior_available`/`comparison_available` are both
false. This addresses the inference side even while `keep` leaves targets untouched.

## Limitations of the heuristic
- Lexical, so it over-triggers on non-temporal "stable" (e.g. "stable cardiac
  silhouette") and can miss paraphrases.
- Sentence-level clause removal can leave a target shorter than intended or, rarely,
  drop a clinically relevant non-temporal clause that shares a sentence with a
  temporal one.
- `prior_available` must be populated upstream before `require_prior_context` /
  `exclude_sample` are meaningful; until then they behave conservatively (treat every
  study as prior-less).
- A radiologist has not reviewed the rewrites; `remove_temporal_clauses` is a research
  lever, not a validated preprocessing step.
