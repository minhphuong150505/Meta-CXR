"""Stage1Context must be a real snapshot, not a mutable global in disguise."""

import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.run_context import Stage1Context  # noqa: E402


class TestImmutability(unittest.TestCase):
    def test_fields_cannot_be_reassigned(self):
        context = Stage1Context(run_name="run_a")
        with self.assertRaises(FrozenInstanceError):
            context.run_name = "run_b"

    def test_thresholds_cannot_be_mutated_through_the_snapshot(self):
        """The old code handed out a dict any caller could edit in place."""
        source = {"Edema": {"positive": 0.4}}
        context = Stage1Context(run_name="run_a", thresholds=source)
        with self.assertRaises(TypeError):
            context.thresholds["Edema"] = {"positive": 0.9}
        with self.assertRaises(TypeError):
            context.thresholds["Edema"]["positive"] = 0.9

    def test_mutating_the_source_dict_does_not_change_the_snapshot(self):
        source = {"Edema": {"positive": 0.4}}
        context = Stage1Context(run_name="run_a", thresholds=source)
        source["Edema"]["positive"] = 0.99
        source["Cardiomegaly"] = {"positive": 0.5}
        self.assertEqual(context.thresholds["Edema"]["positive"], 0.4)
        self.assertNotIn("Cardiomegaly", context.thresholds)

    def test_two_contexts_are_independent(self):
        """Two runs in one process must not interfere."""
        a = Stage1Context(run_name="run_a", thresholds={"Edema": {"positive": 0.3}})
        b = Stage1Context(run_name="run_b", thresholds={"Edema": {"positive": 0.7}})
        self.assertEqual(a.run_name, "run_a")
        self.assertEqual(b.run_name, "run_b")
        self.assertEqual(a.threshold_for("Edema")["positive"], 0.3)
        self.assertEqual(b.threshold_for("Edema")["positive"], 0.7)


class TestResolution(unittest.TestCase):
    def test_explicit_config_path_wins_over_default(self):
        context = Stage1Context(run_name="r", config_path=Path("/explicit/cfg.yaml"))
        self.assertEqual(
            context.resolve_config_path(Path("/default/cfg.yaml")),
            Path("/explicit/cfg.yaml"),
        )

    def test_default_config_path_used_when_unset(self):
        context = Stage1Context(run_name="r")
        self.assertEqual(
            context.resolve_config_path(Path("/default/cfg.yaml")),
            Path("/default/cfg.yaml"),
        )

    def test_checkpoint_falls_back_to_run_scoped_best(self):
        context = Stage1Context(run_name="my_run")
        self.assertEqual(
            context.resolve_checkpoint_path(Path("/ckpt")),
            Path("/ckpt/my_run/checkpoint_best.pth"),
        )

    def test_explicit_checkpoint_wins(self):
        context = Stage1Context(run_name="my_run", checkpoint_path=Path("/x/best.pth"))
        self.assertEqual(
            context.resolve_checkpoint_path(Path("/ckpt")), Path("/x/best.pth")
        )

    def test_missing_threshold_returns_empty_mapping_not_none(self):
        context = Stage1Context(run_name="r")
        self.assertEqual(dict(context.threshold_for("Nonexistent")), {})

    def test_empty_run_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "run_name"):
            Stage1Context(run_name="   ")


class TestFingerprint(unittest.TestCase):
    def test_payload_is_json_safe_and_stable(self):
        import json

        context = Stage1Context(
            run_name="r",
            config_path=Path("/cfg.yaml"),
            thresholds={"Edema": {"positive": 0.4}},
        )
        payload = context.fingerprint_payload()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["run_name"], "r")
        self.assertEqual(payload["checkpoint_path"], None)

    def test_different_thresholds_give_different_payloads(self):
        a = Stage1Context(run_name="r", thresholds={"Edema": {"positive": 0.4}})
        b = Stage1Context(run_name="r", thresholds={"Edema": {"positive": 0.6}})
        self.assertNotEqual(a.fingerprint_payload(), b.fingerprint_payload())


if __name__ == "__main__":
    unittest.main()
