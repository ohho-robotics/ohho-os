"""Train-time augmentation for dataset versions.

:func:`apply_augmentation` is the function ACT and Diffusion Policy data
loaders call. The arithmetic is pure Python, so the same seed returns the
same sample with or without numpy. A numpy array of shape ``(height, width,
3)`` or a PIL image is accepted at the boundary. Pillow is imported only
when a PIL image is passed. Numpy is imported only when a numpy image is
returned.

Draws use :class:`random.Random` streams. The stream seed is the first 8
bytes of SHA-256 over ``ohho.augment.v1:{seed}:{label}``. Labels are
``color``, ``crop``, ``temporal``, ``state``, and ``action``. A magnitude of
zero does not draw.

Colour factors are drawn once per call, in this order: brightness, contrast,
saturation, hue. Each of brightness, contrast, and saturation is uniform on
``[max(0, 1 - magnitude), 1 + magnitude]``. Hue is uniform on
``[-magnitude, magnitude]``.

When ``temporal_stride`` is greater than 1, one draw sets
``phase = int(random() * stride)``. Frame ``i`` is kept when
``i % stride == phase``. An empty keep falls back to phase 0.

Crop draws, state noise, and action noise then run on the kept frames only.
Cameras are visited in name order. A disabled camera is copied unchanged and
consumes no crop draws. Image operations run in this order: crop, nearest
resize, colour, Gaussian blur. State and action use
``value * scale + offset + jitter``, with
``jitter = (random() * 2 - 1) * noise`` when the noise magnitude is positive.

``copies`` is applied by the version builder as repeated episodes. This
function leaves a sample's frame count unchanged except for temporal
sub-sampling.

Loaders that read a materialised version call
:func:`materialised_apply_config` first. Scale, offset, noise, and copies
are already stored in those frames; crop, resize, colour, blur, temporal
stride, and camera toggles still run here. Loaders that read source
episodes pass the stored :class:`~ohho.datasets.model.Augmentation` through
so scale, offset, and noise run in this function.

:func:`preview_augmentation` returns the before/after object described in
``docs/dataset-versions.md``.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .model import (
    Augmentation,
    CameraToggle,
    ColorJitter,
    Crop,
    DatasetBuildError,
    DatasetVersionError,
    json_number,
)

PREVIEW_SCHEMA = "ohho.augmentation_preview.v1"


class AugmentationError(DatasetVersionError):
    """A sample or augmentation config cannot be applied."""


@dataclass(frozen=True)
class _Factors:
    brightness: float
    contrast: float
    saturation: float
    hue: float


@dataclass(frozen=True)
class _ParsedCamera:
    name: str
    height: int
    width: int
    pixels: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class _ParsedFrame:
    observation_state: tuple[float, ...]
    action: tuple[float, ...]
    timestamp: float
    frame_index: int
    cameras: tuple[_ParsedCamera, ...]


@dataclass(frozen=True)
class _ParsedSample:
    episode_id: str
    frames: tuple[_ParsedFrame, ...]


def prepare_augmentation(augmentation: Augmentation) -> Augmentation:
    """Normalise an augmentation so the same config always hashes the same way."""
    copies = augmentation.copies
    if not isinstance(copies, int) or isinstance(copies, bool) or copies < 1:
        raise DatasetBuildError("augmentation copies must be an integer >= 1")
    noise = {
        "action_noise": json_number(augmentation.action_noise),
        "state_noise": json_number(augmentation.state_noise),
    }
    for name, number in noise.items():
        if number < 0:
            raise DatasetBuildError(f"{name} cannot be negative")
    blur = json_number(augmentation.blur)
    if blur < 0:
        raise DatasetBuildError("blur cannot be negative")
    stride = augmentation.temporal_stride
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise DatasetBuildError("temporal_stride must be an integer >= 1")
    return Augmentation(
        action_scale=json_number(augmentation.action_scale),
        state_scale=json_number(augmentation.state_scale),
        action_offset=json_number(augmentation.action_offset),
        state_offset=json_number(augmentation.state_offset),
        action_noise=noise["action_noise"],
        state_noise=noise["state_noise"],
        copies=copies,
        crop=_prepare_size(augmentation.crop, "crop"),
        resize=_prepare_size(augmentation.resize, "resize"),
        color_jitter=_prepare_jitter(augmentation.color_jitter),
        blur=blur,
        temporal_stride=stride,
        cameras=_prepare_cameras(augmentation.cameras),
    )


def materialised_apply_config(augmentation: Augmentation) -> Augmentation:
    """Config for a loader that reads a built dataset version.

    Scale, offset, noise, and copies were written into the stored frames.
    Crop, resize, colour jitter, blur, temporal stride, and camera toggles
    still run at train time.
    """
    config = prepare_augmentation(augmentation)
    return replace(
        config,
        action_scale=1.0,
        state_scale=1.0,
        action_offset=0.0,
        state_offset=0.0,
        action_noise=0.0,
        state_noise=0.0,
        copies=1,
    )


def apply_augmentation(
    sample: Mapping[str, Any],
    augmentation: Augmentation,
    *,
    seed: int,
    json_images: bool = False,
) -> dict[str, Any]:
    """Apply ``augmentation`` to one sample.

    ``seed`` selects the random streams documented on this module. The same
    sample, config, and seed return the same object. The input is not
    mutated. Numpy images come back as ``uint8`` arrays of shape
    ``(height, width, 3)`` unless ``json_images`` is true, in which case
    every camera's ``pixels`` is a list of RGB ints.
    """
    _require_seed(seed)
    config = prepare_augmentation(augmentation)
    parsed = _parse_sample(sample)
    kept = _keep_frames(parsed.frames, config.temporal_stride, seed)
    factors = _draw_factors(config.color_jitter, seed)
    crop_rng = _rng(seed, "crop")
    state_rng = _rng(seed, "state")
    action_rng = _rng(seed, "action")
    frames: list[dict[str, Any]] = []
    for frame in kept:
        cameras: list[dict[str, Any]] = []
        for camera in frame.cameras:
            height, width, pixels = _transform_camera(camera, config, factors, crop_rng)
            cameras.append(
                _render_camera(
                    camera.name, height, width, pixels, camera.kind, json_images
                )
            )
        frames.append(
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "observation_state": _affine(
                    frame.observation_state,
                    config.state_scale,
                    config.state_offset,
                    config.state_noise,
                    state_rng,
                ),
                "action": _affine(
                    frame.action,
                    config.action_scale,
                    config.action_offset,
                    config.action_noise,
                    action_rng,
                ),
                "cameras": cameras,
            }
        )
    return {"episode_id": parsed.episode_id, "frames": frames}


def preview_augmentation(
    sample: Mapping[str, Any],
    augmentation: Augmentation,
    *,
    seed: int,
) -> dict[str, Any]:
    """Return the before/after payload the console preview renders.

    ``schema`` is ``ohho.augmentation_preview.v1``. ``augmentation`` is the
    normalised config. ``before`` and ``after`` each have ``episode_id`` and
    ``frames``. A frame has ``frame_index``, ``timestamp``,
    ``observation_state``, ``action``, and ``cameras``. A camera has
    ``name``, ``height``, ``width``, and ``pixels`` (row-major RGB ints,
    length ``height * width * 3``). Cameras are sorted by name.
    ``after.frames`` keeps the source ``frame_index`` and omits frames
    removed by temporal sub-sampling. Pixels are JSON ints so the console
    can reshape them without numpy.
    """
    _require_seed(seed)
    config = prepare_augmentation(augmentation)
    parsed = _parse_sample(sample)
    return {
        "schema": PREVIEW_SCHEMA,
        "seed": seed,
        "augmentation": config.to_dict(),
        "before": _render_parsed(parsed),
        "after": apply_augmentation(sample, config, seed=seed, json_images=True),
    }


def _require_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise AugmentationError("seed must be an integer")


def _prepare_size(size: Crop | None, label: str) -> Crop | None:
    if size is None:
        return None
    return Crop(
        height=_positive_int(size.height, f"{label} height"),
        width=_positive_int(size.width, f"{label} width"),
    )


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DatasetBuildError(f"{label} must be an integer >= 1")
    return value


def _prepare_jitter(jitter: ColorJitter) -> ColorJitter:
    hue = _non_negative(jitter.hue, "hue")
    if hue > 0.5:
        raise DatasetBuildError("hue must be between 0 and 0.5")
    return ColorJitter(
        brightness=_non_negative(jitter.brightness, "brightness"),
        contrast=_non_negative(jitter.contrast, "contrast"),
        saturation=_non_negative(jitter.saturation, "saturation"),
        hue=hue,
    )


def _non_negative(value: float, label: str) -> float:
    number = json_number(value)
    if number < 0:
        raise DatasetBuildError(f"{label} cannot be negative")
    return number


def _prepare_cameras(cameras: Sequence[CameraToggle]) -> tuple[CameraToggle, ...]:
    found: dict[str, bool] = {}
    for camera in cameras:
        if not isinstance(camera.name, str) or not camera.name.strip():
            raise DatasetBuildError("camera names must be non-empty strings")
        if any(ord(char) < 32 for char in camera.name):
            raise DatasetBuildError("camera names must not contain control characters")
        name = camera.name.strip()
        if name in found:
            raise DatasetBuildError(f"duplicate camera toggle: {name}")
        if not isinstance(camera.enabled, bool):
            raise DatasetBuildError("camera enabled must be true or false")
        found[name] = camera.enabled
    return tuple(CameraToggle(name, found[name]) for name in sorted(found))


def _rng(seed: int, label: str) -> random.Random:
    text = f"ohho.augment.v1:{seed}:{label}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _span(rng: random.Random, magnitude: float) -> float:
    if magnitude == 0.0:
        return 1.0
    low = max(0.0, 1.0 - magnitude)
    high = 1.0 + magnitude
    return low + rng.random() * (high - low)


def _signed(rng: random.Random, magnitude: float) -> float:
    if magnitude == 0.0:
        return 0.0
    return (rng.random() * 2.0 - 1.0) * magnitude


def _draw_factors(jitter: ColorJitter, seed: int) -> _Factors:
    rng = _rng(seed, "color")
    return _Factors(
        brightness=_span(rng, jitter.brightness),
        contrast=_span(rng, jitter.contrast),
        saturation=_span(rng, jitter.saturation),
        hue=_signed(rng, jitter.hue),
    )


def _keep_frames(
    frames: Sequence[_ParsedFrame], stride: int, seed: int
) -> tuple[_ParsedFrame, ...]:
    if stride == 1:
        return tuple(frames)
    phase = int(_rng(seed, "temporal").random() * stride)
    kept = tuple(frame for index, frame in enumerate(frames) if index % stride == phase)
    if kept:
        return kept
    return tuple(frame for index, frame in enumerate(frames) if index % stride == 0)


def _affine(
    values: Sequence[float],
    scale: float,
    offset: float,
    noise: float,
    rng: random.Random,
) -> list[float]:
    rendered: list[float] = []
    for value in values:
        jitter = 0.0 if noise == 0.0 else (rng.random() * 2.0 - 1.0) * noise
        number = float(value) * scale + offset + jitter
        if not math.isfinite(number):
            raise AugmentationError("augmented values must be finite")
        rendered.append(number)
    return rendered


def _camera_enabled(name: str, toggles: Sequence[CameraToggle]) -> bool:
    for toggle in toggles:
        if toggle.name == name:
            return toggle.enabled
    return True


def _transform_camera(
    camera: _ParsedCamera,
    config: Augmentation,
    factors: _Factors,
    crop_rng: random.Random,
) -> tuple[int, int, tuple[int, ...]]:
    if not _camera_enabled(camera.name, config.cameras):
        return camera.height, camera.width, camera.pixels
    pixels = camera.pixels
    height = camera.height
    width = camera.width
    if config.crop is not None:
        pixels, height, width = _crop_image(
            pixels, height, width, config.crop, crop_rng, camera.name
        )
    if config.resize is not None:
        pixels, height, width = _resize_image(
            pixels, height, width, config.resize.height, config.resize.width
        )
    pixels = _recolor(pixels, factors)
    pixels = _blur_image(pixels, height, width, config.blur)
    return height, width, pixels


def _crop_image(
    pixels: tuple[int, ...],
    height: int,
    width: int,
    crop: Crop,
    rng: random.Random,
    name: str,
) -> tuple[tuple[int, ...], int, int]:
    if crop.height > height or crop.width > width:
        raise AugmentationError(
            f"crop is larger than camera {name!r} ({height}x{width})"
        )
    max_top = height - crop.height
    max_left = width - crop.width
    top = 0 if max_top == 0 else int(rng.random() * (max_top + 1))
    left = 0 if max_left == 0 else int(rng.random() * (max_left + 1))
    out: list[int] = []
    for y in range(top, top + crop.height):
        start = (y * width + left) * 3
        out.extend(pixels[start : start + crop.width * 3])
    return tuple(out), crop.height, crop.width


def _resize_image(
    pixels: tuple[int, ...],
    height: int,
    width: int,
    new_height: int,
    new_width: int,
) -> tuple[tuple[int, ...], int, int]:
    out: list[int] = []
    for y in range(new_height):
        src_y = min(height - 1, (y * height) // new_height)
        for x in range(new_width):
            src_x = min(width - 1, (x * width) // new_width)
            index = (src_y * width + src_x) * 3
            out.extend(pixels[index : index + 3])
    return tuple(out), new_height, new_width


def _recolor(pixels: tuple[int, ...], factors: _Factors) -> tuple[int, ...]:
    if (
        factors.brightness == 1.0
        and factors.contrast == 1.0
        and factors.saturation == 1.0
        and factors.hue == 0.0
    ):
        return pixels
    values = [component * factors.brightness for component in pixels]
    if factors.contrast != 1.0:
        mean = sum(values) / float(len(values))
        values = [(component - mean) * factors.contrast + mean for component in values]
    if factors.saturation != 1.0 or factors.hue != 0.0:
        adjusted: list[float] = []
        for index in range(0, len(values), 3):
            red, green, blue = (
                _clamp_unit(values[index + offset]) for offset in range(3)
            )
            gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue
            red = gray + (red - gray) * factors.saturation
            green = gray + (green - gray) * factors.saturation
            blue = gray + (blue - gray) * factors.saturation
            hue, saturation, value = _rgb_to_hsv(
                _clamp_unit(red), _clamp_unit(green), _clamp_unit(blue)
            )
            red, green, blue = _hsv_to_rgb((hue + factors.hue) % 1.0, saturation, value)
            adjusted.extend((red, green, blue))
        values = adjusted
    return tuple(_quantize(component) for component in values)


def _blur_image(
    pixels: tuple[int, ...], height: int, width: int, sigma: float
) -> tuple[int, ...]:
    if sigma <= 0.0:
        return pixels
    radius = max(1, int(round(sigma * 2.0)))
    offsets = range(-radius, radius + 1)
    weights = [math.exp(-0.5 * (offset / sigma) ** 2) for offset in offsets]
    total = sum(weights)
    weights = [weight / total for weight in weights]
    horizontal = [0.0] * (height * width * 3)
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                acc = 0.0
                for weight, offset in zip(weights, offsets):
                    xx = min(max(x + offset, 0), width - 1)
                    acc += pixels[(y * width + xx) * 3 + channel] * weight
                horizontal[(y * width + x) * 3 + channel] = acc
    out: list[int] = []
    for y in range(height):
        for x in range(width):
            for channel in range(3):
                acc = 0.0
                for weight, offset in zip(weights, offsets):
                    yy = min(max(y + offset, 0), height - 1)
                    acc += horizontal[(yy * width + x) * 3 + channel] * weight
                out.append(_quantize(acc))
    return tuple(out)


def _rgb_to_hsv(red: float, green: float, blue: float) -> tuple[float, float, float]:
    red /= 255.0
    green /= 255.0
    blue /= 255.0
    maximum = max(red, green, blue)
    minimum = min(red, green, blue)
    delta = maximum - minimum
    if delta == 0.0:
        hue = 0.0
    elif maximum == red:
        hue = ((green - blue) / delta) % 6.0
    elif maximum == green:
        hue = (blue - red) / delta + 2.0
    else:
        hue = (red - green) / delta + 4.0
    hue /= 6.0
    saturation = 0.0 if maximum == 0.0 else delta / maximum
    return hue, saturation, maximum


def _hsv_to_rgb(
    hue: float, saturation: float, value: float
) -> tuple[float, float, float]:
    hue = hue % 1.0
    sector = int(hue * 6.0) % 6
    fraction = hue * 6.0 - int(hue * 6.0)
    peak = value * (1.0 - saturation)
    fade_down = value * (1.0 - fraction * saturation)
    fade_up = value * (1.0 - (1.0 - fraction) * saturation)
    if sector == 0:
        red, green, blue = value, fade_up, peak
    elif sector == 1:
        red, green, blue = fade_down, value, peak
    elif sector == 2:
        red, green, blue = peak, value, fade_up
    elif sector == 3:
        red, green, blue = peak, fade_down, value
    elif sector == 4:
        red, green, blue = fade_up, peak, value
    else:
        red, green, blue = value, peak, fade_down
    return red * 255.0, green * 255.0, blue * 255.0


def _clamp_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 255.0:
        return 255.0
    return value


def _quantize(value: float) -> int:
    if value < 0.0:
        return 0
    if value > 255.0:
        return 255
    return int(math.floor(value + 0.5))


def _parse_sample(sample: Mapping[str, Any]) -> _ParsedSample:
    if not isinstance(sample, Mapping):
        raise AugmentationError("sample must be an object")
    episode_id = ""
    if "episode_id" in sample and sample["episode_id"] is not None:
        if not isinstance(sample["episode_id"], str):
            raise AugmentationError("episode_id must be a string")
        episode_id = sample["episode_id"]
    frames = sample.get("frames")
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise AugmentationError("sample frames must be a list")
    if not frames:
        raise AugmentationError("sample has no frames")
    return _ParsedSample(
        episode_id=episode_id,
        frames=tuple(_parse_frame(item, index) for index, item in enumerate(frames)),
    )


def _parse_frame(item: Any, index: int) -> _ParsedFrame:
    if not isinstance(item, Mapping):
        raise AugmentationError("each frame must be an object")
    if "frame_index" in item:
        frame_index = item["frame_index"]
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise AugmentationError("frame_index must be an integer")
    else:
        frame_index = index
    if "timestamp" in item:
        timestamp = _finite(item["timestamp"], "timestamp")
    else:
        timestamp = 0.0
    if "observation_state" not in item:
        raise AugmentationError("observation_state must be a list of numbers")
    if "action" not in item:
        raise AugmentationError("action must be a list of numbers")
    return _ParsedFrame(
        observation_state=_vector(item["observation_state"], "observation_state"),
        action=_vector(item["action"], "action"),
        timestamp=timestamp,
        frame_index=frame_index,
        cameras=_parse_cameras(item.get("cameras")),
    )


def _parse_cameras(value: Any) -> tuple[_ParsedCamera, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        parsed = [_parse_named(name, image) for name, image in value.items()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parsed = [_parse_camera_entry(item) for item in value]
    else:
        raise AugmentationError("cameras must be a mapping or a list")
    names = [camera.name for camera in parsed]
    if len(names) != len(set(names)):
        raise AugmentationError("camera names in a sample must be unique")
    return tuple(sorted(parsed, key=lambda camera: camera.name))


def _parse_named(name: Any, image: Any) -> _ParsedCamera:
    if not isinstance(name, str) or not name.strip():
        raise AugmentationError("camera names must be non-empty strings")
    height, width, pixels, kind = _parse_image(image)
    return _ParsedCamera(name.strip(), height, width, pixels, kind)


def _parse_camera_entry(item: Any) -> _ParsedCamera:
    if not isinstance(item, Mapping) or "name" not in item:
        raise AugmentationError("each camera needs a name")
    if "pixels" in item:
        return _parse_named(item["name"], item)
    if "image" in item:
        return _parse_named(item["name"], item["image"])
    raise AugmentationError("each camera needs pixels or an image")


def _parse_image(image: Any) -> tuple[int, int, tuple[int, ...], str]:
    if _is_ndarray(image):
        return _parse_ndarray(image)
    if type(image).__module__.startswith("PIL."):
        return _parse_pil(image)
    if isinstance(image, Mapping):
        return _parse_image_mapping(image)
    raise AugmentationError(
        "a camera image must be RGB pixels, a numpy array of shape "
        "(height, width, 3), or a PIL image"
    )


def _parse_ndarray(value: Any) -> tuple[int, int, tuple[int, ...], str]:
    shape = tuple(value.shape)
    if len(shape) != 3 or int(shape[2]) != 3:
        raise AugmentationError(
            "a numpy camera image must have shape (height, width, 3)"
        )
    height = int(shape[0])
    width = int(shape[1])
    if height < 1 or width < 1:
        raise AugmentationError("camera image is empty")
    flat = value.reshape(-1).tolist()
    if len(flat) != height * width * 3:
        raise AugmentationError("pixels length must be height * width * 3")
    return height, width, tuple(_pixel(item) for item in flat), "numpy"


def _parse_pil(value: Any) -> tuple[int, int, tuple[int, ...], str]:
    image_module = _load_pil()
    if not isinstance(value, image_module.Image):
        raise AugmentationError(
            "Pillow is required to apply an augmentation to a PIL image. "
            "Install it with pip install pillow, or pass RGB pixels as a list of ints."
        )
    rgb = value.convert("RGB")
    width, height = rgb.size
    if height < 1 or width < 1:
        raise AugmentationError("camera image is empty")
    pixels = tuple(rgb.tobytes())
    if len(pixels) != height * width * 3:
        raise AugmentationError("pixels length must be height * width * 3")
    return height, width, pixels, "list"


def _parse_image_mapping(
    image: Mapping[str, Any],
) -> tuple[int, int, tuple[int, ...], str]:
    if "height" not in image or "width" not in image or "pixels" not in image:
        raise AugmentationError("camera image needs height, width, and pixels")
    height = _image_dim(image["height"], "height")
    width = _image_dim(image["width"], "width")
    raw = image["pixels"]
    kind = "list"
    if _is_ndarray(raw):
        raw = raw.reshape(-1).tolist()
        kind = "numpy"
    elif isinstance(raw, (bytes, bytearray)):
        raw = list(raw)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise AugmentationError("pixels must be a list of RGB ints")
    if len(raw) != height * width * 3:
        raise AugmentationError("pixels length must be height * width * 3")
    return height, width, tuple(_pixel(item) for item in raw), kind


def _image_dim(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AugmentationError(f"camera {label} must be an integer >= 1")
    return value


def _vector(value: Any, label: str) -> tuple[float, ...]:
    if _is_ndarray(value):
        value = value.reshape(-1).tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AugmentationError(f"{label} must be a list of numbers")
    return tuple(_finite(item, label) for item in value)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise AugmentationError(f"{label} values must be finite numbers")
    if isinstance(value, (int, float)):
        number = float(value)
    elif type(value).__module__ == "numpy":
        number = float(value)
    else:
        raise AugmentationError(f"{label} values must be finite numbers")
    if not math.isfinite(number):
        raise AugmentationError(f"{label} values must be finite numbers")
    return number


def _pixel(value: Any) -> int:
    if isinstance(value, bool):
        raise AugmentationError("pixel values must be integers from 0 to 255")
    if isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise AugmentationError("pixel values must be integers from 0 to 255")
        number = int(value)
    elif type(value).__module__ == "numpy":
        number_f = float(value)
        if not math.isfinite(number_f) or not number_f.is_integer():
            raise AugmentationError("pixel values must be integers from 0 to 255")
        number = int(number_f)
    else:
        raise AugmentationError("pixel values must be integers from 0 to 255")
    if number < 0 or number > 255:
        raise AugmentationError("pixel values must be integers from 0 to 255")
    return number


def _is_ndarray(value: Any) -> bool:
    return type(value).__module__ == "numpy" and hasattr(value, "shape")


def _load_pil() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise AugmentationError(
            "Pillow is required to apply an augmentation to a PIL image. "
            "Install it with pip install pillow, or pass RGB pixels as a list of ints."
        ) from exc
    return Image


def _load_numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise AugmentationError(
            "numpy is required to return a numpy camera image. "
            "Install it with pip install numpy, or pass RGB pixels as a list of ints."
        ) from exc
    return numpy


def _render_camera(
    name: str,
    height: int,
    width: int,
    pixels: tuple[int, ...],
    kind: str,
    json_images: bool,
) -> dict[str, Any]:
    if kind == "numpy" and not json_images:
        numpy = _load_numpy()
        emitted: Any = numpy.asarray(pixels, dtype=numpy.uint8).reshape(
            height, width, 3
        )
    else:
        emitted = list(pixels)
    return {"name": name, "height": height, "width": width, "pixels": emitted}


def _render_parsed(parsed: _ParsedSample) -> dict[str, Any]:
    frames = []
    for frame in parsed.frames:
        cameras = [
            _render_camera(
                camera.name,
                camera.height,
                camera.width,
                camera.pixels,
                "list",
                True,
            )
            for camera in frame.cameras
        ]
        frames.append(
            {
                "frame_index": frame.frame_index,
                "timestamp": frame.timestamp,
                "observation_state": list(frame.observation_state),
                "action": list(frame.action),
                "cameras": cameras,
            }
        )
    return {"episode_id": parsed.episode_id, "frames": frames}
