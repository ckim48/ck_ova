import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from run_comparisons import read_rounds, same_protocol, summarize, wait_for_reference


class ComparisonQueueTests(unittest.TestCase):
    def test_failed_reference_does_not_block_comparators(self):
        manifest = {"status": "failed", "error": "No feasible design"}
        with patch("run_comparisons.read_json", return_value=manifest), \
                patch("run_comparisons.validate_reference"), \
                patch("run_comparisons.read_rounds", return_value=[{"round": 90}]):
            self.assertEqual(wait_for_reference(), manifest)

    def test_partial_round_is_not_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "rounds.jsonl").write_text('{"round": 1}\n{"round":')
            self.assertEqual(read_rounds(directory), [{"round": 1}])

    def test_protocol_must_match_and_use_cuda(self):
        reference = {"config": {"seed": 0}, "source_sha256": {"model": "abc"}, "device": "cuda"}
        self.assertTrue(same_protocol(reference, dict(reference)))
        self.assertFalse(same_protocol(reference, dict(reference, device="cpu")))
        self.assertFalse(same_protocol(reference, dict(reference, config={"seed": 1})))
        self.assertFalse(same_protocol(reference, dict(reference, source_sha256={})))

    def test_summary_keeps_rounds_and_missing_results_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "base"
            base.mkdir()
            (base / "manifest.json").write_text(json.dumps({"status": "running"}))
            (base / "rounds.jsonl").write_text(json.dumps({
                "round": 10, "validation": {"loss": 1.2, "accuracy": 0.7, "samples": 1000},
                "transmission": {"energy_bound_j": 0.002},
                "cumulative_energy_bound_j": 0.02,
            }) + "\n")
            suite = Path(temporary) / "suite"
            summarize(base, suite)
            with (suite / "comparison.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["round"], "10")
            self.assertEqual(rows[0]["validation_loss"], "1.2")
            self.assertEqual(rows[0]["cumulative_energy_bound_j"], "0.02")
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["status"] == "not_started" for row in rows[1:]))
            self.assertTrue(all(row["validation_loss"] == "" for row in rows[1:]))


if __name__ == "__main__":
    unittest.main()
