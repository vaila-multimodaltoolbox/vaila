# Vertical Jump — Frame B, B3_r3_c3

Version: 0.4.0 · Updated: 14 September 2026

The **Vertical Jump** button calls `Vaila.vailajump()` and opens
`vaila.vaila_and_jump.vaila_and_jump()`.

Choose an input directory and one of four modes:

1. Time of Flight: CSV with mass and flight time, optional contact time.
2. Jump Height: CSV with mass and measured height, optional contact time.
3. MediaPipe: landmark CSV plus mass, capture FPS and measured shank length.
4. Team Batch: a parent directory of athlete folders, each with CSVs and its own
   `vaila_and_jump_config.toml`.

MediaPipe and Team Batch produce **CMJ Performance Profile — PODS**:
Person (mass), Outcome (height, explicitly sourced velocity/momentum, mRSI),
Driver (estimated braking/bottom/propulsive force and power), and Strategy
(time to takeoff, countermovement depth, phase durations).

**Markerless force and power estimates are not force-platform measurements.**
Read the phase/kinetic QC, then compare event lines with the recorded movement.
Missing or invalid new metrics remain unavailable. No normative strength groups,
fatigue or injury diagnosis is inferred from the PODS profile.

## CLI

```sh
python -m vaila.vaila_and_jump -i "athlete/jump.csv" -c "athlete/config.toml" -o "results"
python -m vaila.vaila_and_jump -i "team athletes" --batch -o "team results"
python -m vaila.vaila_and_jump -i "flight-time CSVs" -d 1 -o "results"
```

Mode 3 uses `[jump_context]` and `[jump_phase]` from the exact `-c` file.
Configure a quiet standing baseline before the countermovement. Use capture FPS
for slow-motion video. The generated template documents onset persistence/noise,
minimum phase duration and takeoff agreement.

## Results and compatibility

Existing output names remain: scalar `*_jump_results_*.csv`, frame-wise
`*_calibrated_*.csv`, plots and HTML. MediaPipe adds a four-panel
`*_cmj_pods_*.png`, scalar PODS metrics/QC and frame `cmj_phase`/estimated kinetics
aliases. Team Batch adds descriptive statistics, PODS charts and z-scores.

Legacy `takeoff_frame` means the upward CoM standing-height crossing, not foot-off.
Legacy `propulsion_time_s` ends there and is never used for mRSI.
New `time_to_takeoff_s` runs from persistent movement onset to selected takeoff.
Flight-time height uses `g*T*T/8`; height-derived velocity uses `sqrt(2*g*h)`.
Potential energy at apex equals the height-derived kinetic energy at takeoff:
the current code does not sum them as independent energies.

See [module help](../../vaila/help/vaila_and_jump.md) and the
[complete PODS variable/event dictionary, QC and validation limitations](../cmj_pods.md).
