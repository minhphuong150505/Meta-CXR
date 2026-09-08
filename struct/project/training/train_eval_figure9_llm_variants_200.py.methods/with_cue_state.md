# with_cue_state(record, cue_rule)

Called by `build_stage1_records` for fresh records and cache hits. Returns a new
dictionary with `cue_rule` and `cue_state`, preserving the tensor objects.
`none` clears groups and records `not_provided`; other rules infer `predicted`
from nonempty P/N/U groups or `abstained` from empty groups. Unknown rules raise.

Consumer: `stage2.prompts.records.context_from_record` → `PromptContext` →
`PromptBuilder`. Withheld/abstained states do not emit a normal statement.

Tests: `tests/test_cue_contract.py`, including a legacy cache hit that refuses
any model loading and checks that cached dictionaries are not mutated.

Parent: [engine](../train_eval_figure9_llm_variants_200.py.doc.md).
