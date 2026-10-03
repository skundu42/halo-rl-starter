import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from halo_demo.__main__ import train
from halo_demo.io import write_json, write_jsonl


class OrchestrationTests(unittest.TestCase):
    def test_updates_use_latest_adapter_and_no_signal_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = (Path(directory) / "run").resolve()
            calls = []

            def stage(name, output, index=0, adapter=None):
                calls.append((name, index, adapter))
                if name == "evaluate":
                    label = "before" if index == 0 else "after"
                    write_json(output / f"eval-{label}.json", {"accuracy": 0.5})
                elif name == "collect":
                    round_dir = output / f"round-{index:03d}"
                    round_dir.mkdir()
                    write_jsonl(
                        round_dir / "rollouts.jsonl",
                        [
                            {
                                "prompt": "test",
                                "completions": ["right", "wrong"],
                                "rewards": [1.0, 1.0] if index == 2 else [1.0, 0.0],
                            }
                        ],
                    )
                else:
                    (output / f"round-{index:03d}" / "adapter").mkdir()

            args = argparse.Namespace(config="configs/default.toml", smoke=False, output=str(run_dir))
            with patch("halo_demo.__main__.subprocess.run"), patch("halo_demo.__main__.run_stage", side_effect=stage):
                train(args)
            first = run_dir / "round-001" / "adapter"
            self.assertIn(("collect", 2, first), calls)
            self.assertIn(("collect", 3, first), calls)
            self.assertNotIn("update", [name for name, index, _ in calls if index == 2])
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary["updated_rounds"], 4)
            self.assertEqual(summary["adapter"], "round-005/adapter")
            self.assertEqual(calls[-1], ("evaluate", 6, run_dir / "round-005" / "adapter"))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory, patch("halo_demo.__main__.subprocess.run"):
            args = argparse.Namespace(config="configs/default.toml", smoke=True, output=directory)
            with self.assertRaises(FileExistsError):
                train(args)

    def test_all_tied_run_reports_failure_and_no_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"

            def stage(name, run_dir, index=0, adapter=None):
                if name == "evaluate":
                    write_json(run_dir / f"eval-{'before' if index == 0 else 'after'}.json", {"accuracy": 1.0})
                else:
                    self.assertEqual(name, "collect")
                    round_dir = run_dir / f"round-{index:03d}"
                    round_dir.mkdir()
                    write_jsonl(round_dir / "rollouts.jsonl", [{"completions": ["a", "b"], "rewards": [1, 1]}])

            args = argparse.Namespace(config="configs/default.toml", smoke=True, output=str(output))
            with patch("halo_demo.__main__.subprocess.run"), patch("halo_demo.__main__.run_stage", side_effect=stage):
                with self.assertRaisesRegex(SystemExit, "No learning signal"):
                    train(args)
            summary = json.loads((output / "summary.json").read_text())
            self.assertIsNone(summary["adapter"])
            self.assertEqual(summary["updated_rounds"], 0)


if __name__ == "__main__":
    unittest.main()
