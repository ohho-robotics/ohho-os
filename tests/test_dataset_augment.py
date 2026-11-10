"""Train-time augmentation is deterministic and preview returns before/after."""

from __future__ import annotations

import copy
import sys
import unittest
from unittest.mock import patch

from ohho.datasets import (
    PREVIEW_SCHEMA,
    Augmentation,
    AugmentationError,
    CameraToggle,
    ColorJitter,
    Crop,
    DatasetBuildError,
    apply_augmentation,
    materialised_apply_config,
    preview_augmentation,
)
from ohho.datasets.augment import prepare_augmentation


def _pixels(height: int, width: int, fill: tuple[int, int, int]) -> list[int]:
    return list(fill) * (height * width)


def _image(height: int, width: int, fill: tuple[int, int, int]) -> dict:
    return {"height": height, "width": width, "pixels": _pixels(height, width, fill)}


def _impulse() -> dict:
    pixels = _pixels(3, 3, (0, 0, 0))
    pixels[4 * 3 : 4 * 3 + 3] = [255, 255, 255]
    return {"height": 3, "width": 3, "pixels": pixels}


def _sample(frames: int = 1, cameras: dict | None = None) -> dict:
    if cameras is None:
        cameras = {"front": _image(2, 2, (100, 40, 10))}
    return {
        "episode_id": "ep-1",
        "frames": [
            {
                "frame_index": index,
                "timestamp": index / 10.0,
                "observation_state": [1.0, -2.0],
                "action": [0.25],
                "cameras": cameras,
            }
            for index in range(frames)
        ],
    }


class TestApplyAugmentation(unittest.TestCase):
    def test_default_config_is_identity_and_leaves_the_input_alone(self) -> None:
        sample = _sample()
        original = copy.deepcopy(sample)
        out = apply_augmentation(sample, Augmentation(), seed=4)
        self.assertEqual(sample, original)
        frame = out["frames"][0]
        self.assertEqual(frame["observation_state"], [1.0, -2.0])
        self.assertEqual(frame["action"], [0.25])
        self.assertEqual(frame["cameras"][0]["pixels"], _pixels(2, 2, (100, 40, 10)))
        self.assertEqual(
            len(apply_augmentation(sample, Augmentation(copies=4), seed=1)["frames"]), 1
        )

    def test_scale_and_offset_are_exact(self) -> None:
        out = apply_augmentation(
            _sample(),
            Augmentation(
                state_scale=2.0, state_offset=0.5, action_scale=4.0, action_offset=-1.0
            ),
            seed=3,
        )
        self.assertEqual(out["frames"][0]["observation_state"], [2.5, -3.5])
        self.assertEqual(out["frames"][0]["action"], [0.0])

    def test_noise_is_deterministic_and_stays_inside_the_magnitude(self) -> None:
        config = Augmentation(state_noise=0.1, action_noise=0.2)
        sample = _sample(cameras={})
        first = apply_augmentation(sample, config, seed=0)
        second = apply_augmentation(sample, config, seed=0)
        other_seeds = [
            apply_augmentation(sample, config, seed=seed) for seed in range(1, 6)
        ]
        self.assertEqual(first, second)
        self.assertGreater(len({json_key(item) for item in [first, *other_seeds]}), 1)
        for item in (first, *other_seeds):
            state = item["frames"][0]["observation_state"]
            action = item["frames"][0]["action"]
            self.assertLessEqual(abs(state[0] - 1.0), 0.1 + 1e-12)
            self.assertLessEqual(abs(state[1] - (-2.0)), 0.1 + 1e-12)
            self.assertLessEqual(abs(action[0] - 0.25), 0.2 + 1e-12)
        quiet = Augmentation()
        self.assertEqual(
            apply_augmentation(sample, quiet, seed=0)["frames"][0]["observation_state"],
            apply_augmentation(sample, quiet, seed=9)["frames"][0]["observation_state"],
        )

    def test_colour_crop_and_temporal_depend_on_the_seed(self) -> None:
        jitter = Augmentation(color_jitter=ColorJitter(brightness=0.8, hue=0.25))
        colour = []
        for seed in range(6):
            out = apply_augmentation(_sample(), jitter, seed=seed)
            colour.append(tuple(out["frames"][0]["cameras"][0]["pixels"]))
            self.assertEqual(
                colour[-1],
                tuple(
                    apply_augmentation(_sample(), jitter, seed=seed)["frames"][0][
                        "cameras"
                    ][0]["pixels"]
                ),
            )
        self.assertGreater(len(set(colour)), 1)

        image = _checkerboard()
        crop = Augmentation(crop=Crop(height=2, width=2))
        sample = _sample(cameras={"front": image})
        crops = []
        for seed in range(6):
            out = apply_augmentation(sample, crop, seed=seed)
            camera = out["frames"][0]["cameras"][0]
            self.assertEqual((camera["height"], camera["width"]), (2, 2))
            crops.append(tuple(camera["pixels"]))
        self.assertGreater(len(set(crops)), 1)

        frames = [
            {
                "frame_index": index,
                "observation_state": [float(index)],
                "action": [0.0],
            }
            for index in range(4)
        ]
        stride = Augmentation(temporal_stride=2)
        patterns = set()
        for seed in range(24):
            out = apply_augmentation({"frames": frames}, stride, seed=seed)
            indices = tuple(frame["frame_index"] for frame in out["frames"])
            self.assertIn(indices, {(0, 2), (1, 3)})
            self.assertEqual(
                indices,
                tuple(
                    frame["frame_index"]
                    for frame in apply_augmentation(
                        {"frames": frames}, stride, seed=seed
                    )["frames"]
                ),
            )
            patterns.add(indices)
        self.assertGreater(len(patterns), 1)

    def test_resize_blur_and_camera_toggle(self) -> None:
        source = {
            "height": 2,
            "width": 2,
            "pixels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
        resized = apply_augmentation(
            _sample(cameras={"front": source}),
            Augmentation(resize=Crop(height=1, width=1)),
            seed=0,
        )
        self.assertEqual(resized["frames"][0]["cameras"][0]["pixels"], [1, 2, 3])

        blurred = apply_augmentation(
            _sample(cameras={"front": _impulse()}),
            Augmentation(blur=1.0),
            seed=0,
        )
        pixels = blurred["frames"][0]["cameras"][0]["pixels"]
        self.assertEqual(
            pixels,
            apply_augmentation(
                _sample(cameras={"front": _impulse()}),
                Augmentation(blur=1.0),
                seed=2,
            )["frames"][0]["cameras"][0]["pixels"],
        )
        self.assertLess(pixels[4 * 3], 255)
        self.assertGreater(pixels[0], 0)

        toggled = apply_augmentation(
            _sample(cameras={"wrist": _impulse(), "front": _impulse()}),
            Augmentation(blur=1.0, cameras=(CameraToggle("wrist", False),)),
            seed=0,
        )
        by_name = {camera["name"]: camera for camera in toggled["frames"][0]["cameras"]}
        self.assertEqual(list(by_name), ["front", "wrist"])
        self.assertEqual(by_name["wrist"]["pixels"], _impulse()["pixels"])
        self.assertNotEqual(by_name["front"]["pixels"], _impulse()["pixels"])

    def test_materialised_view_keeps_image_settings_and_skips_baked_noise(self) -> None:
        config = Augmentation(
            state_scale=2.0, state_noise=0.4, action_noise=0.4, blur=1.0
        )
        view = materialised_apply_config(config)
        self.assertEqual(view.state_scale, 1.0)
        self.assertEqual(view.state_noise, 0.0)
        self.assertEqual(view.action_noise, 0.0)
        self.assertEqual(view.blur, 1.0)
        self.assertEqual(view.copies, 1)
        out = apply_augmentation(_sample(cameras={"front": _impulse()}), view, seed=0)
        self.assertEqual(out["frames"][0]["observation_state"], [1.0, -2.0])
        self.assertNotEqual(
            out["frames"][0]["cameras"][0]["pixels"], _impulse()["pixels"]
        )

    def test_prepare_normalises_camera_order_and_rejects_bad_ranges(self) -> None:
        left = Augmentation(
            cameras=(CameraToggle("wrist", False), CameraToggle("front", True))
        )
        right = Augmentation(
            cameras=(CameraToggle("front", True), CameraToggle("wrist", False))
        )
        self.assertEqual(prepare_augmentation(left), prepare_augmentation(right))
        with self.assertRaises(DatasetBuildError):
            prepare_augmentation(
                Augmentation(
                    cameras=(CameraToggle("front", True), CameraToggle("front", False))
                )
            )
        with self.assertRaises(DatasetBuildError):
            prepare_augmentation(Augmentation(temporal_stride=0))
        with self.assertRaises(DatasetBuildError):
            prepare_augmentation(Augmentation(color_jitter=ColorJitter(hue=0.6)))
        with self.assertRaises(AugmentationError):
            apply_augmentation(
                _sample(), Augmentation(crop=Crop(height=8, width=8)), seed=0
            )
        with self.assertRaises(AugmentationError):
            apply_augmentation(
                {
                    "frames": [
                        {
                            "observation_state": [0.0],
                            "action": [0.0],
                            "cameras": {
                                "front": {"height": 1, "width": 1, "pixels": [1, 2]}
                            },
                        }
                    ]
                },
                Augmentation(),
                seed=0,
            )

    def test_preview_shape(self) -> None:
        sample = {
            "episode_id": "ep-7",
            "frames": [
                {
                    "frame_index": index,
                    "timestamp": float(index),
                    "observation_state": [float(index)],
                    "action": [0.0],
                    "cameras": {
                        "wrist": _image(1, 1, (1, 2, 3)),
                        "front": _image(1, 1, (4, 5, 6)),
                    },
                }
                for index in range(4)
            ],
        }
        config = Augmentation(temporal_stride=2, blur=0.0)
        preview = preview_augmentation(sample, config, seed=1)
        again = preview_augmentation(sample, config, seed=1)
        self.assertEqual(preview, again)
        self.assertEqual(preview["schema"], PREVIEW_SCHEMA)
        self.assertEqual(preview["seed"], 1)
        self.assertEqual(
            set(preview), {"schema", "seed", "augmentation", "before", "after"}
        )
        self.assertEqual(preview["augmentation"]["temporal_stride"], 2)
        before = preview["before"]["frames"][0]
        self.assertEqual(
            set(before),
            {"frame_index", "timestamp", "observation_state", "action", "cameras"},
        )
        self.assertEqual(
            [camera["name"] for camera in before["cameras"]], ["front", "wrist"]
        )
        self.assertEqual(
            set(before["cameras"][0]), {"name", "height", "width", "pixels"}
        )
        self.assertEqual(len(before["cameras"][0]["pixels"]), 3)
        self.assertEqual(len(preview["before"]["frames"]), 4)
        after_index = [frame["frame_index"] for frame in preview["after"]["frames"]]
        self.assertIn(tuple(after_index), {(0, 2), (1, 3)})
        self.assertEqual(preview["before"]["episode_id"], "ep-7")

    def test_missing_pillow_names_the_install(self) -> None:
        class FakeImage:
            pass

        FakeImage.__module__ = "PIL.Image"
        sample = _sample(cameras={"front": FakeImage()})
        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            with self.assertRaises(AugmentationError) as caught:
                apply_augmentation(sample, Augmentation(), seed=0)
        self.assertIn("pillow", str(caught.exception).lower())

    def test_numpy_image_round_trip(self) -> None:
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy not installed")
        image = numpy.zeros((2, 2, 3), dtype=numpy.uint8)
        image[0, 0] = (10, 20, 30)
        sample = _sample(cameras={"front": image})
        out = apply_augmentation(sample, Augmentation(), seed=0)
        pixels = out["frames"][0]["cameras"][0]["pixels"]
        self.assertEqual(type(pixels).__module__, "numpy")
        self.assertEqual(tuple(pixels.shape), (2, 2, 3))
        self.assertEqual(int(pixels[0, 0, 0]), 10)
        self.assertTrue(
            numpy.array_equal(
                image,
                numpy.array(
                    [[[10, 20, 30], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]],
                    dtype=numpy.uint8,
                ),
            )
        )
        preview = preview_augmentation(sample, Augmentation(blur=1.0), seed=0)
        self.assertIsInstance(
            preview["after"]["frames"][0]["cameras"][0]["pixels"], list
        )


def _checkerboard() -> dict:
    pixels: list[int] = []
    for y in range(4):
        for x in range(4):
            tone = y * 4 + x
            pixels.extend((tone, tone * 2, 200 - tone))
    return {"height": 4, "width": 4, "pixels": pixels}


def json_key(payload: dict) -> str:
    frame = payload["frames"][0]
    return repr((frame["observation_state"], frame["action"]))


if __name__ == "__main__":
    unittest.main()
