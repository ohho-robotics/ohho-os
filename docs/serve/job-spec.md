# Job spec `ohho.job/v1`

One document describes an optimize, parity, and bench request. The CLI,
the console, workers, and CI all read this schema. This package is the
spec and the local commands. The console form and the job API are
separate.

The JSON Schema file is the only source:

`ohho/serve/job/schema.json`

`$id`: `https://ohho-robotics.com/schemas/ohho.job/v1.json`

`schema_version` is required and must be the string `ohho.job/v1`.
Every job that validates has that value on the object. A missing field
is a `required` error at `$.schema_version`. Any other value is a
`const` or `type` error.

Python types (`ohho/serve/job/generated.py`) and TypeScript types
(`schemas/ts/job.v1.d.ts`) are generated from the schema:

```bash
python scripts/generate_job_types.py
python scripts/generate_job_types.py --check
```

Target profile ids are free strings. This schema does not catalog
profiles.

## Document

| Field | Required | Meaning |
| --- | --- | --- |
| `schema_version` | yes | Constant `ohho.job/v1`. |
| `model.version` | yes | Model version reference: a repository id, a digest, or any other non-empty string. |
| `targets` | yes | One or more target profile ids. |
| `steps` | yes | `optimize`, `parity`, and/or `bench`, in execution order, with no duplicates. |
| `precision` | no | `fp32`, `fp16`, `bf16`, or `int8`. |
| `backend` | no | `onnx`, `torchscript`, or `cpu`. |
| `parity` | no | `max_abs_error` and `max_rel_error` (>= 0), `min_cosine` (-1 to 1), `samples` (integer >= 1). |
| `bench` | no | `iterations` (integer >= 1), `warmup` (integer >= 0), `batch_size` (integer >= 1), `timeout_s` (> 0). |
| `budget` | no | `max_seconds` (> 0), `max_usd` (>= 0), `max_gpu_seconds` (>= 0). |
| `outputs.wandb` | no | `enabled` (boolean), optional `project` and `entity`. |
| `outputs.prometheus` | no | `enabled` (boolean), optional `port` (1–65535) and `job_name`. |

Unknown properties are rejected. Omitted optional sections stay unset.
The validator does not invent thresholds, budgets, or export settings.

Checked-in examples:

- `ohho/serve/job/examples/full.yaml`
- `ohho/serve/job/examples/bench.yaml`
- `ohho/serve/job/examples/parity.yaml`

Files are YAML or JSON. `.json`, and any body whose first non-space
character is `{` or `[`, is parsed as JSON. Other files use a stdlib
YAML subset: comments, space indentation, block mappings and sequences,
plain and quoted scalars, `true` / `false` / `null` / `~`, numbers, and
one-line flow collections. Tabs in indentation, multiline `|` / `>`
strings, anchors, aliases, tags, merge keys, and multiple documents are
parse errors. Quote a scalar that would look like one of those forms.
PyYAML is not required.

## Validation messages

Each error is one line. The console and the server show this line
unchanged:

```text
<path>: <code>: <message>
```

Split on the first two `": "` separators (`path`, `code`, `message`).
The message may contain `": "`. The path and the code do not.

Path grammar, starting at `$`:

- `.name` when the key matches `[A-Za-z_][A-Za-z0-9_]*`
- `["..."]` for any other key, with the key encoded by `json.dumps(..., ensure_ascii=True)`
- `[index]` for an array index (decimal, no leading zeros)

Codes and messages:

| Code | Message |
| --- | --- |
| `required` | `missing required property ` plus the JSON string of the name |
| `type` | `expected <json type>, got <json type>` |
| `enum` | `value must be one of ` plus the JSON strings of the allowed values, comma-separated |
| `const` | `value must be ` plus the JSON encoding of the constant |
| `additional_property` | `unknown property ` plus the JSON string of the name |
| `minimum` | `value must be >= ` plus the JSON number |
| `maximum` | `value must be <= ` plus the JSON number |
| `exclusive_minimum` | `value must be > ` plus the JSON number |
| `exclusive_maximum` | `value must be < ` plus the JSON number |
| `min_length` | `string length must be >= <n>` |
| `max_length` | `string length must be <= <n>` |
| `min_items` | `array length must be >= <n>` |
| `max_items` | `array length must be <= <n>` |
| `unique_items` | `duplicate item` |
| `parse` | `line <n>: <reason>` |
| `io` | `cannot read ` plus the JSON string of the path, then `: ` and the reason |

JSON types in `type` messages are `object`, `array`, `string`, `number`,
`integer`, `boolean`, and `null`. Booleans are not integers. A float is
a `number`, not an `integer`.

Examples:

```text
$.steps: required: missing required property "steps"
$.steps[0]: enum: value must be one of "optimize", "parity", "bench"
$.schema_version: const: value must be "ohho.job/v1"
$.budget.max_usd: minimum: value must be >= 0
```

`ohho job validate SPEC --json` and `ohho job run SPEC --json` print the
same fields on stdout when the document is rejected:

```json
{
  "errors": [
    {
      "code": "enum",
      "message": "value must be one of \"optimize\", \"parity\", \"bench\"",
      "path": "$.steps[0]"
    }
  ],
  "ok": false,
  "schema_version": "ohho.job/v1"
}
```

`schema_version` on an error result is the schema being applied. The
line form is `path + ": " + code + ": " + message`.

## CLI

```bash
ohho job validate spec.yaml
ohho job run spec.yaml
```

`--json` prints a JSON object on stdout and does not also print the text
form.

Validate success (text):

```text
ok
schema_version: ohho.job/v1
```

Validate success (`--json`) is `{"ok": true, "schema_version": "ohho.job/v1", "job": { ... }}`.
The `job` object is the validated document, including `schema_version`.

Exit status:

| Code | Meaning |
| --- | --- |
| 0 | The spec validated, and for `run`, every step returned `ok`. |
| 1 | Parse, read, or schema failure, or a step returned `error`. |
| 2 | The spec validated, and at least one step is not available in this build. |

## Local run

`ohho job run` validates first. It then runs steps in order, and only
calls a step engine that this build registers. `optimize`, `parity`,
and `bench` are not registered. Each of those steps prints:

```text
<step> is not available in this build
```

The process exits 2. A result has `step`, `status`, and `message` only.
It does not include timings, scores, or export ids. Nothing in this
build invents those values.

Text output also prints `schema_version: ohho.job/v1` before the step
lines. `--json` is:

```json
{
  "ok": false,
  "schema_version": "ohho.job/v1",
  "steps": [
    {
      "message": "optimize is not available in this build",
      "status": "not_available",
      "step": "optimize"
    }
  ]
}
```

`status` is `ok`, `not_available`, or `error`.
