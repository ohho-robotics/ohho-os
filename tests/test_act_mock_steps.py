from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ohho.train.act as act


class TestActMockSteps(unittest.TestCase):
    def test_mock_logged_steps_are_strictly_increasing(self) -> None:
        for train_steps in (1, 100, 300):
            with self.subTest(train_steps=train_steps):
                with tempfile.TemporaryDirectory() as tmp:
                    output_dir = Path(tmp) / str(train_steps)
                    logged_steps: list[int] = []

                    def capture(step: int, _metrics: dict[str, object]) -> None:
                        logged_steps.append(step)

                    with (
                        patch.object(
                            act,
                            "DatasetReader",
                            return_value=SimpleNamespace(
                                state_dim=9,
                                action_dim=9,
                            ),
                        ),
                        patch.object(
                            act,
                            "open_metrics",
                            return_value=nullcontext(),
                        ),
                        patch.object(act, "log", side_effect=capture),
                    ):
                        act.train_act(
                            dataset="unused",
                            output_dir=str(output_dir),
                            train_steps=train_steps,
                            mock=True,
                        )

                    self.assertTrue(logged_steps)
                    self.assertTrue(
                        all(step <= train_steps for step in logged_steps),
                        f"mock trainer logged a step beyond train_steps: {logged_steps}",
                    )
                    self.assertTrue(
                        all(
                            previous < current
                            for previous, current in zip(
                                logged_steps, logged_steps[1:]
                            )
                        ),
                        f"mock trainer logged non-increasing steps: {logged_steps}",
                    )
                    self.assertEqual(logged_steps[-1], train_steps)


if __name__ == "__main__":
    unittest.main()
