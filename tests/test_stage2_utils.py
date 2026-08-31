import json
import tempfile
import unittest
from pathlib import Path

from training.stage2_utils import (
    SCHEMA_VERSION,
    accumulation_window_size,
    adapter_is_complete,
    contains_sensitive_eval_fields,
    language_lora_target_names,
    masked_label_ids,
    native_findings_instruction,
    private_bucket_violations,
    safe_prediction_row,
    select_threshold_class,
    stable_fingerprint,
    validate_soft_token_batch,
)


class Stage2UtilsTest(unittest.TestCase):
    def test_threshold_falls_back_to_argmax(self):
        result = select_threshold_class(
            [0.2, 0.4, 0.3], {"negative": 0.9, "positive": 0.9, "uncertain": 0.9}
        )
        self.assertEqual(result, "positive")

    def test_fingerprint_is_order_stable_and_cohort_sensitive(self):
        self.assertEqual(stable_fingerprint({"b": 2, "a": 1}), stable_fingerprint({"a": 1, "b": 2}))
        self.assertNotEqual(stable_fingerprint({"split": "val"}), stable_fingerprint({"split": "test"}))

    def test_prompt_mask_requires_an_exact_prefix(self):
        self.assertEqual(masked_label_ids([1, 2, 3, 4], [1, 2]), [-100, -100, 3, 4])
        with self.assertRaises(ValueError):
            masked_label_ids([1, 9, 3], [1, 2])

    def test_tail_accumulation_is_scaled_by_actual_window(self):
        self.assertEqual([accumulation_window_size(i, 10, 4) for i in range(10)], [4] * 8 + [2] * 2)

    def test_default_prediction_artifact_is_deidentified(self):
        row = safe_prediction_row(sample_key="abc", index=2, prediction="clear lungs", generation_ok=True)
        self.assertFalse(contains_sensitive_eval_fields(row))

    def test_adapter_completion_checks_all_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in [
                "adapter_config.json",
                "adapter_model.safetensors",
                "meta.json",
                "img_proj.pt",
                "trainer_state.pt",
            ]:
                (root / name).write_bytes(b"x")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "complete",
                        "image_mode": "qformer",
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(adapter_is_complete(root, "qformer"))
            (root / "img_proj.pt").unlink()
            self.assertFalse(adapter_is_complete(root, "qformer"))

    def test_quantized_language_targets_are_selected_by_name(self):
        names = [
            "model.vision_tower.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.embed_tokens",
        ]
        self.assertEqual(
            language_lora_target_names(names),
            [
                "model.language_model.layers.0.mlp.down_proj",
                "model.language_model.layers.0.self_attn.q_proj",
            ],
        )

    def test_native_instruction_has_no_stage1_structured_context(self):
        instruction = native_findings_instruction()
        self.assertIn("provided chest radiograph", instruction)
        self.assertNotIn("Abnormality information", instruction)

    def test_private_bucket_requires_pap_and_no_public_principals(self):
        private_metadata = {
            "iamConfiguration": {"publicAccessPrevention": "enforced"}
        }
        self.assertEqual(private_bucket_violations(private_metadata, {"bindings": []}), [])
        self.assertTrue(
            private_bucket_violations(
                private_metadata,
                {
                    "bindings": [
                        {"role": "roles/storage.objectViewer", "members": ["allUsers"]}
                    ]
                },
            )
        )
        self.assertTrue(private_bucket_violations({}, {"bindings": []}))


class SoftTokenBatchValidation(unittest.TestCase):
    def test_matching_batch_passes(self):
        validate_soft_token_batch(4, 4, 32, 32)

    def test_fewer_embeddings_than_batch_is_rejected(self):
        # Previously clamped to the last embedding, silently pairing one study's
        # image with another study's report.
        with self.assertRaises(ValueError) as ctx:
            validate_soft_token_batch(1, 4, 32, 32)
        self.assertIn("do not match the batch", str(ctx.exception))

    def test_more_embeddings_than_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_soft_token_batch(8, 4, 32, 32)

    def test_wrong_soft_token_count_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_soft_token_batch(4, 4, 31, 32)
        self.assertIn("32 image tokens", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


# --------------------------------------------------------------------------
# Mid-epoch recovery checkpointing
# --------------------------------------------------------------------------


def test_train_fine_accepts_save_every_updates():
    """A full-cohort epoch is ~70 h and save_adapter otherwise runs only after
    the batch loop, so a hang late in the epoch loses everything."""
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "training" / "train_eval_figure9_llm_variants_200.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "train_fine"
    )
    names = [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]
    assert "save_every_updates" in names


def test_the_recovery_save_is_marked_in_progress():
    """A partial adapter must never be mistaken for a finished run."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1]
        / "training" / "train_eval_figure9_llm_variants_200.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("if save_every_updates and global_step"):]
    block = block[: block.index("print(")]
    assert 'status="in_progress"' in block


def test_the_default_keeps_the_old_behaviour():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "training" / "run_medgemma_qlora.py"
    ).read_text(encoding="utf-8")
    assert '"--save-every-updates", type=int, default=0' in source
