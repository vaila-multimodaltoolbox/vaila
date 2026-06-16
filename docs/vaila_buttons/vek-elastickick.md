# VEK ElasticKick Button

The **VEK ElasticKick** button inside **Soccer Tools** launches
`vaila.vek`, the vailá-ElasticKick biomechanical assessment module.

## Purpose

Analyze a soccer or futsal kick performed with elastic-band resistance using:

- markerless lower-limb pose CSV;
- optional ball tracking CSV;
- elastic-band force-length calibration CSV;
- TOML configuration.

## GUI Path

`vaila.py -> Soccer Tools -> VEK ElasticKick`

## What It Calculates

- Elastic-band length, extension, tension, power and work.
- Hip, knee, ankle and foot velocities.
- Knee, hip and ankle angles, angular velocities and ROM.
- Ball launch velocity, launch angle, kinetic energy and impulse.
- Contact event and quality-control indicators.

## Configuration

Create a starter TOML:

```bash
python -m vaila.vek --write-default-config vek_config.toml
```

Then edit athlete metadata, frame rate, kicking limb, band anchor, calibration
path, filters and contact settings.

## CLI Equivalent

```bash
python -m vaila.vek \
  --input pose.csv \
  --ball ball.csv \
  --band-calibration vek_band_calibration.csv \
  --config vek_config.toml \
  --output results
```

## Reports

VEK writes processed CSV files, summary CSV, PNG plots and a dashboard-style
HTML report.

---
See also: [VEK workflow](../vek.md), [Help page](../../vaila/help/vek.md).
