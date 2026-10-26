# Robot plugins and RobotSpec v2

Status: Proposed — awaiting Varun's review

OHH-309. This ADR records how OhhO OS stays robot-agnostic across every robot
class: mobile base, mobile manipulator, arm, quadruped, aerial and humanoid.
Booster T2 is only the first humanoid adapter. Switching robots is a config
change, not a code change.

This change adds the spec fields and writes the rule down. It does not discover
plugins, extract safety policies, or edit the open drone branches.

## The six modularity rules

1. **Core code never names a vendor or a model.** `ohho/` (outside
   `ohho/robots/<plugin>/` and the out-of-tree plugin packages) must not
   contain `booster`, `t2`, `unitree`, `go2`, `px4`, `crazyflie`, `yahboom` or
   `omnibot` once the M0R refactors land. A CI grep test enforces this for new
   names from day one (`booster`, `t2`) and for the existing names once each
   has moved. That grep is C-07. It is not added here, and the built-in specs
   in `ohho/registry.py` keep their current ids until M0R.

2. **A robot is a plugin.** A plugin declares specs, capabilities with
   parameters, transports (URI schemes), sim backends, a safety policy (or uses
   its class default), a descriptor for the console and VR, asset locations and
   an optional doctor. Plugins are found through the Python entry-point group
   `ohho.robots`. Built-ins use that path (C-02). Spec data stays in
   `ohho/registry.py` and `ohho.robots.builtin` registers it, with
   `source="builtin"`. A checkout that has not been reinstalled loads the same
   module when the entry point is absent.

3. **Behaviour asks for capabilities, never for robot ids.**
   `robot.has("humanoid.locomotion")`, not `if robot_id == "booster-t2"`.

4. **Safety is chosen by robot class.** `SafetyGate` delegates to a
   `SafetyPolicy` picked from `spec.robot_class` (ground, legged, arm, aerial,
   humanoid). A plugin may make its class policy stricter, never looser. That
   delegation is C-05. This change does not touch `ohho/safety.py`, safety
   caps, watchdogs, arming, or the e-stop latch.

5. **One switch point.** The robot id is resolved once: `--robot` on the CLI,
   then `OHHO_ROBOT`, then `~/.ohho/config.toml`, then `sim`. sim-serve, MCP,
   the recorder, the console (`/sim/[robotId]`) and VR all pass that id and
   nothing else robot-specific. C-06 (`ohho/config.py`, OHH-314) is that
   resolver. Per-robot files are `~/.ohho/robots/<id>.toml` or a repo-local
   `ohho.toml`. Their safety overrides may only tighten caps.

6. **Sims, CI and the console are parameterised by robot.** One CI matrix over
   registered robots, one scenario per robot class, one console sim page driven
   by the descriptor. `RobotSpec.ci_sim` is the opt-in (`enabled` plus a
   scenario name). The matrix workflow is C-08.

## Where plugins live

Plugins live in this repo at `ohho-os/plugins/<name>`, each as its own
distribution, installed with an extra. The first humanoid plugin is
`ohho-os/plugins/ohho-booster`, extra `ohho-os[booster]`. An out-of-tree
package uses the same `ohho.robots` entry-point group, so moving a plugin to
its own repo later does not change core.

Core does not import a plugin. A plugin must not import its vendor SDK at
module import time; only the transport factory does, so `pip install ohho-os`
without extras still lists every spec. Scheme collisions are an error at load,
naming both plugins. `dds://` stays the Unitree scheme already on main; a
humanoid plugin uses its own scheme.

C-02 implements discovery. This ADR only fixes the location and the group name.

## RobotSpec v2

`RobotSpec` keeps every existing field. Nothing is renamed or removed.

| Field | Meaning | Safe default |
|---|---|---|
| `robot_class` | `RobotClass` enum: `mobile_base`, `mobile_manipulator`, `arm`, `quadruped`, `aerial`, `humanoid` | derived from `category` |
| `joint_groups` | name → ordered joint names | empty mapping |
| `limits` | one typed object for the class | derived from `max_lin` / `max_ang` |
| `assets` | URDF, MJCF, meshes, licence | empty paths, empty licence |
| `ci_sim` | `enabled` bool plus scenario name | disabled, scenario `""` |
| `source` | which plugin registered the spec | `""`; built-ins set `"builtin"` |
| `capability_params` | JSON-like parameters keyed by capability name | empty mapping |

`category` stays a free string so old manifests load. `limits` is a description
of the class. `Robot.drive` still clamps with `max_lin` and `max_ang`.
`SafetyGate` still uses `SafetyProfile`. The new objects are not consulted by
the gate, and this change does not copy safety-profile numbers into them.

| `robot_class` | Limits type | What it holds |
|---|---|---|
| `mobile_base` | `GroundLimits` | `max_vx`, `max_vy`, `max_wz` |
| `mobile_manipulator` | `MobileManipulatorLimits` | `ground` + `arm` |
| `arm` | `ArmLimits` | `max_joint_step`, `max_tau` (`None` if undeclared) |
| `quadruped` | `LeggedLimits` | `max_vx`, `max_vy`, `max_wz` |
| `aerial` | `AerialLimits` | `max_climb`, `gps_fix`, `requires_hardware_arming`, `experimental`, `flight_ready` |
| `humanoid` | `HumanoidLimits` | `max_vx`, `max_vy`, `max_wz`, `max_joint_step`, `max_kp`, `max_tau` |

Velocity fields default from the flat clamps (`max_vx` and `max_vy` from
`max_lin`, `max_wz` from `max_ang`). Arm and humanoid joint fields default to
`None` (not declared). Aerial extras default to the drone drafts' ground-robot
defaults: climb `0.0`, `gps_fix` `""`, flags false.

`capability_params` is a JSON object keyed by capability name. `has()` stays a
string check. The names that carry parameters are `base.drive`,
`base.holonomic_drive`, `legged.pose`, `aerial`, `humanoid.locomotion`,
`humanoid.posture`, `whole_body.joint`, `manipulation`, `manipulation.dual_arm`,
`end_effector.gripper`, `end_effector.dex_hand` and `head.gaze`. Perception
capabilities stay strings with no parameter schema. When the field is omitted,
parameters are filled from the spec's limits and joint groups.
They describe the robot. They are not safety caps, and `SafetyGate` does not
read them. `register()` rejects a capability name that is not in the
vocabulary. `RobotSpec(...)` still accepts an explicit mapping, so a document
that already stored one round-trips.

`CiSim.enabled` defaults to false so existing robots do not join a matrix that
does not exist yet. Scenario ids, when a
plugin opts in, come from `CI_SCENARIOS`:

| Class | Scenario |
|---|---|
| `mobile_base`, `mobile_manipulator` | `drive_square` |
| `arm` | `arm_home` |
| `quadruped` | `quadruped_walk` |
| `aerial` | `aerial_square` |
| `humanoid` | `humanoid_walk` |

`RobotSpec.to_dict()` is JSON, and `spec_from_dict` accepts it. The `limits`
object always includes `max_lin` and `max_ang`, so a reader that still does
`limits.get("max_lin")` keeps working. `limits.kind` is the `robot_class`
value. Old files that have no `kind` still load.

## category → robot_class

| Category string | `robot_class` |
|---|---|
| `wheeled`, `mobile-base`, `mobile_base` | `mobile_base` |
| `mobile-manipulator`, `mobile_manipulator` | `mobile_manipulator` |
| `arm`, `manipulator` | `arm` |
| `legged`, `quadruped` | `quadruped` |
| `aerial`, `drone` | `aerial` |
| `humanoid`, `biped` | `humanoid` |

Lookup strips whitespace and ignores case. The stored `category` string is not
rewritten. A manifest that sets `robot_class` uses that value; `category` is
only the fallback when `robot_class` is omitted.

Built-ins today:

| Robot id | `category` | `robot_class` |
|---|---|---|
| `sim` | `wheeled` | `mobile_base` |
| `omnibot` | `mobile-manipulator` | `mobile_manipulator` |
| `unitree-go2` | `legged` | `quadruped` |

An unknown category, including a missing category (stored as the existing
default `"unknown"`), becomes `mobile_base`. If that manifest also omits
`capabilities`, it receives the generic mobile-base set, which is the built-in
`sim` set: `base.drive`, `base.holonomic_drive`, `perception.rgb`,
`perception.scan`. A known category that omits capabilities stays empty, as it
does today. Explicit capabilities are kept either way.

`ohho/robots/example.json` is a known `wheeled` manifest and loads unchanged.

## Aerial fields during the drone-PR transition

Open drafts ohho-os #57, #60, #62, #63 and #65 add these **flat** fields on
`RobotSpec`:

| Field | Type | Default | Role |
|---|---|---|---|
| `max_climb` | float | `0.0` | climb cap, m/s |
| `gps_fix` | str | `""` | `"3d"`, `"none"`, or unset |
| `requires_hardware_arming` | bool | `False` | a non-loopback URI needs a human present |
| `experimental` | bool | `False` | integration is not finished |
| `flight_ready` | bool | `False` | nothing marks a robot ready to fly |

They are not dataclass fields on this branch. Adding them would collide with
those diffs, and this issue does not edit those branches. They already live on
`AerialLimits`. `RobotSpec` exposes them as **read-only properties**:

- `spec.max_climb`
- `spec.gps_fix`
- `spec.requires_hardware_arming`
- `spec.experimental`
- `spec.flight_ready`

On an aerial spec the property reads `AerialLimits`. On every other class it
returns the draft's default (`0.0`, `""`, or `False`), which is what those
drafts store for ground robots.

`spec_from_dict` copies the flat keys into `AerialLimits` when `robot_class` is
`aerial`, from `limits.max_climb` or a top-level `max_climb`, and the same for
`gps_fix`, `requires_hardware_arming` and `experimental`. Only a real boolean
`True` sets a flag. A flat manifest (no `limits.kind`) cannot set
`flight_ready`; it stays false even if the file says true. A typed document
with `limits.kind` of `aerial` may carry `flight_ready` so `to_dict` round-trips.

When those PRs merge, R-03 resolves `ohho/registry.py` like this and does not
leave both a field and a property of the same name:

1. Delete the flat dataclass fields the drafts added.
2. Keep these properties, so `spec.max_climb` and the other four names still read.
3. Keep `spec_from_dict` writing the flat keys into `AerialLimits`.
4. Change constructors that pass `max_climb=` (and the other four) to
   `limits=AerialLimits(...)`.
5. Leave `limits.max_lin`, `limits.max_ang` and, for aerial specs,
   `limits.max_climb` in the serialized `limits` object so `limits.get(...)`
   in the draft loaders still works through the overlap.

`requires_hardware_arming` on the spec is this aerial flag only. It is not the
`SafetyGate` arming check, and it does not arm hardware.

## Consequences

- Old manifests and `RobotSpec(...)` calls keep working. New fields have defaults.
- `example.json` still loads. Built-in `max_lin` / `max_ang` are unchanged.
- Safety caps, watchdogs, arming and the e-stop path are unchanged.
- Plugin discovery is C-02 (`ohho/plugins.py`, group `ohho.robots`).
  C-03 (OHH-311) asks those plugins for a transport scheme before the scheme
  dict in `ohho/adapters/__init__.py`. The dict and its entries stay, so a
  scheme only the dict knows still resolves. `RobotSpec.sim_backend` (empty by
  default) selects a plugin sim backend for `sim://` and for auto; an empty
  name keeps today's simulator return. `dds://` stays the Unitree scheme.
  Per-class `SafetyPolicy`, the robot-id resolver and the CI matrix are later
  issues (C-05, C-06, C-08). They should follow this ADR rather than grow more
  fields on the flat spec. `SafetyGate` profile lookup stays as it is until
  C-05.
