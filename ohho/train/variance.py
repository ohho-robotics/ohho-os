"""Published run-to-run variance for each shipped recipe.

``expected_variance`` and ``tolerance`` are the figures a comparison view
uses. ``tolerance`` is the absolute ``final_loss`` band that still counts as
within noise. Deterministic mode publishes ``0`` because the CPU trainers
are required to match exactly. The other mode publishes the grain of the
recorded loss (``1e-4``) as the expected variance and ``1e-3`` as the hint
band. Those default figures are a published hint, not a statistic computed
from a sample of runs.

Families with no trainer in this build publish no number. A re-run cannot
produce a final loss for them.

The console view that would show the hint is not in this build. Call
:func:`comparison_hint` from that view when it exists.
"""

from __future__ import annotations

from dataclasses import dataclass

CONSOLE_FOLLOWUP = (
    "A console comparison view can call comparison_hint and read tolerance "
    "to show a within-noise hint. That view is not in this build."
)

_METRIC = "final_loss"
_TRAINED = ("act", "diffusion")
_UNTRAINED = ("smolvla", "openvla", "ppo")
_DEFAULT_VARIANCE = 1e-4
_DEFAULT_TOLERANCE = 1e-3

_DETERMINISTIC_NOTE = (
    "Deterministic CPU training must match final_loss exactly. "
    "Expected variance and tolerance are 0."
)
_DEFAULT_NOTE = (
    "Published hint for a run that did not set deterministic. "
    "final_loss is recorded to 1e-4. This is not a sample variance. "
    "A comparison view can treat an absolute gap at or below 1e-3 as within noise."
)
_UNTRAINED_NOTE = (
    "This build has no trainer for this recipe, so a re-run cannot produce final_loss."
)


@dataclass(frozen=True)
class RecipeNoise:
    """One published variance row."""

    recipe: str
    metric: str
    deterministic: bool
    expected_variance: float | None
    tolerance: float | None
    note: str


def recipe_noise(recipe: str, *, deterministic: bool) -> RecipeNoise:
    """Return the published row for ``recipe`` in the requested mode."""
    for row in all_recipe_noise():
        if row.recipe == recipe and row.deterministic is deterministic:
            return row
    return RecipeNoise(
        recipe=recipe,
        metric=_METRIC,
        deterministic=deterministic,
        expected_variance=None,
        tolerance=None,
        note="No published variance for this recipe.",
    )


def all_recipe_noise() -> tuple[RecipeNoise, ...]:
    """Every shipped recipe, deterministic mode and the default mode."""
    rows: list[RecipeNoise] = []
    for recipe in _TRAINED:
        rows.append(
            RecipeNoise(
                recipe=recipe,
                metric=_METRIC,
                deterministic=True,
                expected_variance=0.0,
                tolerance=0.0,
                note=_DETERMINISTIC_NOTE,
            )
        )
        rows.append(
            RecipeNoise(
                recipe=recipe,
                metric=_METRIC,
                deterministic=False,
                expected_variance=_DEFAULT_VARIANCE,
                tolerance=_DEFAULT_TOLERANCE,
                note=_DEFAULT_NOTE,
            )
        )
    for recipe in _UNTRAINED:
        for deterministic in (True, False):
            rows.append(
                RecipeNoise(
                    recipe=recipe,
                    metric=_METRIC,
                    deterministic=deterministic,
                    expected_variance=None,
                    tolerance=None,
                    note=_UNTRAINED_NOTE,
                )
            )
    return tuple(rows)


def within_noise(
    recipe: str,
    original: float,
    candidate: float,
    *,
    deterministic: bool,
) -> bool:
    """True when ``candidate`` is inside the published tolerance of ``original``.

    A tolerance of 0 requires an exact numeric match. A recipe with no
    published tolerance is not within noise.
    """
    hint = comparison_hint(recipe, original, candidate, deterministic=deterministic)
    return bool(hint["within_noise"])


def comparison_hint(
    recipe: str,
    original: float | None,
    candidate: float | None,
    *,
    deterministic: bool,
) -> dict[str, object]:
    """Fields a comparison view needs for the within-noise hint.

    ``console_followup`` records that the view itself is not in this build.
    """
    spec = recipe_noise(recipe, deterministic=deterministic)
    exact = _numbers_equal(original, candidate)
    delta: float | None = None
    if _is_number(original) and _is_number(candidate):
        delta = abs(float(original) - float(candidate))
    if spec.tolerance is None or delta is None:
        noisy = False
    elif spec.tolerance == 0:
        noisy = exact
    else:
        noisy = delta <= float(spec.tolerance)
    return {
        "recipe": spec.recipe,
        "metric": spec.metric,
        "deterministic": spec.deterministic,
        "original": original,
        "candidate": candidate,
        "abs_delta": delta,
        "exact": exact,
        "expected_variance": spec.expected_variance,
        "tolerance": spec.tolerance,
        "within_noise": noisy,
        "note": spec.note,
        "console_followup": CONSOLE_FOLLOWUP,
    }


def format_noise(value: float | None) -> str:
    """Render a published variance or tolerance for the docs table."""
    if value is None:
        return "n/a"
    if value == 0:
        return "0"
    text = f"{float(value):.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e+")


def doc_table_rows() -> tuple[str, ...]:
    """Markdown rows for :func:`all_recipe_noise`, one recipe per line."""
    families: list[str] = []
    for row in all_recipe_noise():
        if row.recipe not in families:
            families.append(row.recipe)
    lines: list[str] = []
    for family in families:
        det = recipe_noise(family, deterministic=True)
        default = recipe_noise(family, deterministic=False)
        lines.append(
            f"| `{family}` | `{det.metric}` | {format_noise(det.expected_variance)} "
            f"| {format_noise(det.tolerance)} "
            f"| {format_noise(default.expected_variance)} "
            f"| {format_noise(default.tolerance)} |"
        )
    return tuple(lines)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numbers_equal(left: object, right: object) -> bool:
    if not _is_number(left) or not _is_number(right):
        return False
    return float(left) == float(right)
