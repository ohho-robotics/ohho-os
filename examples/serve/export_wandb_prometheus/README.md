# Export bench results to W&B and Prometheus

Placeholder. This tree has no bench report exporter. There is no W&B logger
and no Prometheus text renderer on `ohho.bench`. This script checks for those
functions. It does not import W&B, does not open a push gateway, does not
write a metrics file, and prints no metric values.

When the exporter is on main, this example has to be pointed at it. It will
not invent a sample series before then. Documents that exporters read are
`ohho.bench/v1` (`ohho bench validate`).

## Hardware

CPU. No GPU and no network.

## Runtime

Under 5 seconds. Nothing is exported.

## Expected output

```text
example: export-wandb-prometheus
status: placeholder
hardware: cpu
pending: W&B and Prometheus export from the bench report
runtime: not-run
result: none
```

## Pending

Weights & Biases logging and Prometheus exposition from a bench report.
Both are absent on main.
