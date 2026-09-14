# Markerless CMJ Performance Profile — PODS

Version: 0.4.0 · Updated: 14 September 2026

## Architecture and scope

`vaila/vaila_and_jump.py` retains calibration, anthropometric CoM reconstruction,
12 Hz zero-lag Butterworth filtering, derivatives, legacy events, height/gravity
QC and old outputs. `vaila/cmj_pods.py` consumes those signals after the existing
foot and kinetic candidates are available. It returns one event dictionary, one
scalar dictionary and a frame table. CSV, HTML and the new four-panel plot use
these results without detecting events again. No dependencies were added.

The Person/Outcome/Driver/Strategy presentation is conceptually informed by
[Ripley et al., Biomechanics 2025;5(2):32](https://doi.org/10.3390/biomechanics5020032).
That study used force platforms. Its sample-specific strength split and group
values are **not** vailá thresholds. PODS here is an interpretive organization,
not a reproduction of its force-platform algorithms.

**Force and power are markerless estimates, not measured force-platform kinetics.**
The new module does not calculate RFD, classify athletes as strong/weak, diagnose
fatigue/injury, or infer a causal deficit from one trial.

## Running and configuration

GUI: Vertical Jump → MediaPipe (mode 3), or Team Batch (mode 4). Use a quiet standing
segment before the countermovement; check the baseline window in the TOML. Team
Batch reads each athlete folder's own `vaila_and_jump_config.toml`.

```sh
python -m vaila.vaila_and_jump -i "athlete/jump.csv" -c "athlete/config.toml" -o "results"
python -m vaila.vaila_and_jump -i "team athletes" --batch -o "team results"
```

Mode 3 honors `[jump_phase]` in the exact `-c` file, even outside the CSV folder.
Existing mode 1 (flight time) and mode 2 (height) do not have a CoM time series and
therefore do not produce PODS phases or force-phase estimates.

```toml
[jump_context]
mass_kg = 80.0
fps = 240.0 # capture FPS, not slow-motion playback FPS
shank_length_m = 0.40

[jump_phase]
baseline_rectify_start = true
baseline_start_frame = 10
baseline_end_frame = 20 # exclusive; quiet stance must end before movement onset
phase_smoothing_window_s = 0.06 # legacy median event signal only
baseline_tolerance_m = 0.02
anchor_events_to_com_peak = true
movement_onset_velocity_threshold_m_s = 0.05
movement_onset_displacement_threshold_m = 0.005
movement_onset_noise_multiplier = 3.0
movement_onset_min_duration_s = 0.03
phase_min_duration_s = 0.03
takeoff_agreement_tolerance_s = 0.04
```

PODS thresholds must be finite and positive. The onset uses baseline median and
1.4826 × MAD of filtered position and velocity. Both downward velocity and downward
displacement must exceed the larger of their configured minimum and noise-scaled
threshold, for `ceil(duration × fps)` consecutive samples (at least two). Search
starts after the configured baseline and ends before the existing CoM bottom.
An insufficient or already-moving baseline is rejected, not silently relocated.
There is no additional smoothing. Thresholds are operational defaults requiring
validation, not physiological norms or sub-frame onset estimates.

## Canonical events

Frame values are **zero-based CSV row positions**, matching the existing processing
pipeline. They are not arbitrary original video frame labels after a cropped CSV.
Event times are frame/FPS. Intervals are half-open `[start, end)`; the boundary
sample starts the following phase. Frame differences give durations.

| Event | Frame field | Definition / source |
|---|---|---|
| Onset | `movement_onset_frame` | First persistent filtered CoM departure after the quiet baseline |
| Downward velocity peak | `peak_downward_velocity_frame` | Minimum `cg_vy` from onset through bottom; also `braking_start_frame` |
| Bottom | `minimum_displacement_frame` | Alias of existing `propulsion_start_frame`, detected before apex; no second minimum |
| Historical reference crossing | `takeoff_frame` | Upward CoM standing-height crossing, **not true takeoff** |
| Foot takeoff | `takeoff_frame_foot_contact` | Existing last detected foot marker exceeding standing height + 2 cm |
| Kinetic candidate | `takeoff_frame_kinetic` | Existing last positive modelled-force sample up to the foot/reference event |
| Selected takeoff | `takeoff_frame_selected` | QC-accepted kinetic candidate, otherwise ordered foot event, otherwise explicit baseline fallback |
| Apex | `max_height_frame` | Existing maximum CoM height |
| Foot landing | `landing_frame_foot_contact` | Existing first foot marker returning to standing threshold |
| Selected landing | `landing_frame_selected` | First foot contact; explicit legacy CoM fallback if missing |

Required order: onset < downward peak ≤ bottom < takeoff < apex < landing.
Downward peak equal to bottom is retained as a sampling limitation and produces
no braking force aggregate. Invalid sequences retain raw candidates but no valid
PODS phase assignment or index. Baseline fallbacks never produce a seemingly valid
mRSI or flight duration. Raw foot/kinetic disagreement is retained as signed
kinetic-minus-foot difference and described in QC notes.

Kinetic selection requires successful gravity QC, chronological validity, positive
takeoff velocity, force at the candidate in `(0, 0.25 BW]`, then at least two
samples / 0.02 s within ±0.1 BW of zero before apex. Its bottom-to-candidate force
must be finite and within the gross artefact bounds. When both candidates exist,
they must agree within 0.04 s by default. These checks use **unmasked** `cg_ay + g`;
flight zeros inserted for legacy plotting cannot validate a kinetic event. A
rejected kinetic candidate remains exported with a QC note. The last-positive
kinetic frame and first-visible foot-off differ in definition by at least sampling
uncertainty; no sub-frame correction is fabricated.

## Variable dictionary

Notation: `m` = mass, `g` = 9.81 m/s², `h` = `height_qc_recommended_m`, `v` = existing
filtered `cg_vy`, `a` = existing `cg_ay`, `F` = existing `force_vertical`, `P` =
existing `power`, `O/D/B/T/A/L` = onset/downpeak/bottom/selected takeoff/apex/landing.
New aliases preserve the historical contact force `m(a+g)` and power `Fv`, including
legacy flight masking. These series are diagnostic even when scalar QC fails.

QC shorthand:

- **E**: finite, in-range event; raw candidate can remain visible after QC failure.
- **C**: positive finite mass/FPS and no gravity scale/FPS failure.
- **H**: C plus positive recommended height with `plausible` or `very_high_plausible` QC.
- **S**: ordered complete phases and actual selected takeoff/landing (no baseline fallback).
- **K**: C + S + gravity `ok` + finite plausible contact acceleration; each aggregate
  also needs at least three finite contact samples and the configured minimum
  duration. Limited temporal resolution is visibly flagged; undersampled phase
  aggregates are NaN.
- **D**: diagnostic series/metadata, not a validated measurement; inspect scalar QC.

### Person, outcomes and strategy

| Variable | Equation / definition | Unit | Source / phase | QC |
|---|---|---|---|---|
| `mass_kg` | Configured body mass (retained) | kg | Person / trial | positive finite input |
| `mrsi_AU` | h / ((T−O)/fps); conventional index, dimensionally m/s | AU | Height QC + onset/takeoff | H + S |
| `mrsi_height_source` | Recommended height's source label | text | Height QC | D |
| `mrsi_takeoff_source` | Copy of `takeoff_source` | text | Selected takeoff | D |
| `mrsi_qc_status` | ok, limited_temporal_resolution, height_qc_failed, phase_detection_failed, calibration_failed, insufficient_data | text | Combined numerator/denominator QC | D |
| `takeoff_velocity_height_derived_m_s` | √(2gh) | m/s | Recommended height; takeoff | H |
| `takeoff_velocity_com_est_m_s` | v[T] if positive | m/s | Filtered CoM; takeoff | C + S |
| `jump_momentum_height_derived_kg_m_s` | m√(2gh) | kg·m/s | Height-derived takeoff velocity | H |
| `jump_momentum_com_est_kg_m_s` | m v[T] | kg·m/s | Filtered CoM; takeoff | C + S |
| `jump_momentum_selected_kg_m_s` | Prefer valid height-derived momentum, otherwise valid CoM estimate | kg·m/s | Explicitly selected source | Corresponding velocity QC |
| `jump_momentum_source` | height_derived, com_est, unavailable | text | Selected outcome | D |
| `movement_onset_frame` | O | frame | Persistent filtered CoM departure | E |
| `movement_onset_time_s` | O/fps | s | Onset | E + valid FPS |
| `peak_downward_velocity_frame` | D = argmin(v[O:B+1]) | frame | Unweighting → braking | E |
| `peak_downward_velocity_time_s` | D/fps | s | Braking start | E + valid FPS |
| `peak_downward_velocity_m_s` | v[D], negative downward | m/s | Filtered CoM; downpeak | E; inspect calibration |
| `braking_start_frame` | Alias of D | frame | Braking start | E |
| `braking_start_time_s` | D/fps | s | Braking start | E + valid FPS |
| `minimum_displacement_frame` | B = propulsion_start_frame | frame | Canonical legacy bottom | E |
| `minimum_displacement_time_s` | B/fps | s | Bottom | E + valid FPS |
| `takeoff_frame_selected` | T selected from existing candidates | frame | Takeoff | E; source/QC required |
| `takeoff_time_selected_s` | T/fps | s | Takeoff | E + valid FPS |
| `takeoff_source` | kinetic_estimated_grf, foot_contact_marker, fallback_com_reference | text | Selected event | D |
| `landing_frame_selected` | L | frame | First contact, otherwise CoM fallback | E |
| `landing_time_selected_s` | L/fps | s | Landing | E + valid FPS |
| `landing_source` | foot_contact_marker or fallback_com_reference | text | Selected event | D |
| `unweighting_duration_s` | (D−O)/fps | s | Unweighting | Ordered events |
| `braking_duration_s` | (B−D)/fps | s | Braking | Ordered events; zero flags insufficient phase sampling |
| `propulsive_duration_s` | (T−B)/fps | s | Propulsion | Ordered events; no takeoff fallback |
| `time_to_takeoff_s` | (T−O)/fps | s | Complete movement to takeoff | Ordered events; no takeoff fallback |
| `flight_duration_selected_s` | (L−T)/fps | s | Flight | S |
| `countermovement_depth_m` | abs(squat_depth_m) | m | Legacy canonical bottom/standing reference | C |

`squat_depth_m` is retained unchanged: the current detector already exports its
absolute magnitude. The new alias makes the sign convention explicit and also
accepts negative legacy input without changing it. Legacy `propulsion_time_s`
remains bottom-to-CoM-baseline crossing and is **never** the mRSI denominator.

### Driver scalars

| Variable | Equation | Unit | Source / phase | QC |
|---|---|---|---|---|
| `avg_braking_force_est_N` | mean F[D:B] | N | Filtered model; braking | K |
| `avg_braking_force_est_N_per_kg` | mean F[D:B] / m | N/kg | Braking | K |
| `peak_braking_force_est_N` | max F[D:B] | N | Braking | K |
| `peak_braking_force_est_N_per_kg` | max F[D:B] / m | N/kg | Braking | K |
| `force_at_min_displacement_est_N` | F[B] | N | Canonical bottom | K, finite sample |
| `force_at_min_displacement_est_N_per_kg` | F[B] / m | N/kg | Bottom | K, finite sample |
| `avg_propulsive_force_est_N` | mean F[B:T] | N | Propulsion | K |
| `avg_propulsive_force_est_N_per_kg` | mean F[B:T] / m | N/kg | Propulsion | K |
| `peak_propulsive_force_est_N` | max F[B:T] | N | Propulsion | K |
| `peak_propulsive_force_est_N_per_kg` | max F[B:T] / m | N/kg | Propulsion | K |
| `avg_braking_power_est_W` | mean P[D:B] | W | Braking; signed eccentric power | K |
| `avg_braking_power_est_W_per_kg` | mean P[D:B] / m | W/kg | Braking | K |
| `peak_braking_power_est_W` | max P[D:B], signed maximum, not eccentric magnitude | W | Braking | K |
| `peak_braking_power_est_W_per_kg` | max P[D:B] / m | W/kg | Braking | K |
| `avg_propulsive_power_est_W` | mean P[B:T] | W | Propulsion | K |
| `avg_propulsive_power_est_W_per_kg` | mean P[B:T] / m | W/kg | Propulsion | K |
| `peak_propulsive_power_est_W` | max P[B:T] | W | Propulsion | K |
| `peak_propulsive_power_est_W_per_kg` | max P[B:T] / m | W/kg | Propulsion | K |

### QC and frame columns

| Variable | Definition | Unit | Source / phase | QC |
|---|---|---|---|---|
| `cmj_chronology_valid` | Required event sequence holds | boolean | All events | D |
| `cmj_phase_qc_status` | ok, limited_temporal_resolution, phase_detection_failed | text | Detection/order/fallback/sampling | D |
| `cmj_phase_qc_note` | Detection/fallback/disagreement explanations | text | All events | D |
| `kinetic_metrics_qc_status` | ok, limited_temporal_resolution, calibration_failed, phase_detection_failed, insufficient_data, force_plausibility_failed | text | Contact/phase/gravity checks | D |
| `kinetic_metrics_qc_note` | Calibration, sample count, force and velocity consistency explanations | text | Trial | D |
| `takeoff_kinetic_vs_foot_diff_frames` | kinetic frame − foot frame | frame | Raw candidate comparison | Both candidates finite |
| `takeoff_kinetic_vs_foot_diff_s` | Frame difference / fps | s | Raw candidate comparison | Both candidates + valid FPS |
| `force_vertical_est_N` | Exact alias of legacy force_vertical (m(a+g), flight masked) | N | Existing filtered kinetics / all frames | D |
| `force_vertical_est_N_per_kg` | force_vertical_est_N / m | N/kg | Same | D + positive finite mass |
| `power_est_W` | Exact alias of legacy power = Fv, flight masked | W | Existing filtered kinetics / all frames | D |
| `power_est_W_per_kg` | power_est_W / m | W/kg | Same | D + positive finite mass |
| `cmj_phase` | standing, unweighting, braking, propulsion, flight, landing_recovery; empty if chronology invalid | category | Same event dictionary / all frames | Ordered events; inspect fallback source |

Existing `cg_y_m_filtered`, `cg_vy`, `cg_ay`, `force_vertical`, `power` and all
legacy scalar names remain. No raw-coordinate differentiation or second filter is
introduced. The new force/power phase means and peaks use identical intervals.
No airborne or touchdown samples are included in the propulsion interval.

## QC and scientific limitations

- `suspect_fps_or_scale` invalidates new kinetics and mRSI. A missing gravity check
  also withholds kinetic scalars; outcome/strategy may still be reported with
  explicit QC context. Height QC failure produces NaN for mRSI/height-derived momentum.
- At 30/60 FPS the result is `limited_temporal_resolution`; it is not blanket-rejected.
  Phase aggregates require three samples and 0.03 s by default. Decimal formatting
  is storage/display precision, not measurement uncertainty.
- Gross contact estimates outside −0.5 to 10 body weights cause
  `force_plausibility_failed`. These conservative artefact guards are not normative
  thresholds; values are not clipped. Diagnostic aliases preserve the original
  estimates for investigation even when scalar Drivers are withheld.
- A velocity difference exceeding max(0.5 m/s, 25% of height-derived velocity)
  receives a note. Height-derived momentum remains transparently preferred; it is
  not an independent outcome and is not force-platform impulse-derived momentum.
- The fixed 12 Hz filter is unchanged, including the legacy fallback to unfiltered
  input for short sequences or insufficient sampling rate. Differentiation and
  filtering can shift/blur onset and kinetic takeoff. Phase QC and waveform review
  are necessary; these internal checks do not establish external validity.
- Foot events are marker-height threshold events, affected by foot posture,
  occlusion, camera perspective, segmentation and tracking. A kinetic fallback
  cannot rescue a poor trajectory automatically.
- Trials should contain one complete CMJ with quiet stance. Baseline contamination,
  mistimed crops and multi-jump recordings can prevent valid detection. Do not
  substitute zero for an unavailable result.

## Reports and outputs

The existing filenames remain. `*_jump_results_*.csv` gains the scalar columns;
`*_calibrated_*.csv` gains preferred aliases and `cmj_phase` with unique headers.
`*_cmj_pods_*.png` adds aligned displacement, velocity, estimated GRF and estimated
power panels, shared phase shading, and selected/raw event markers.
The individual HTML adds Person, Outcome, Driver and Strategy tables, explicit
sources/QC, and the event-definition table, retaining detailed height/gravity sections.

Team Batch adds all PODS team metrics to descriptive summaries and separate
within-team z-scores. Charts include height, mRSI, TTTO, depth and braking/propulsive
relative-force estimates. A PODS overview includes per-trial QC. New z-scores use
neutral styling and do not enter the legacy composite. mRSI gets a ranking; higher
force/depth or lower TTTO are not automatically classified as better. Missing
values remain missing even when the only valid observations are equal.

New Team CSV fields (dimensionless) are `mrsi_AU_zscore`, `time_to_takeoff_s_zscore`,
`countermovement_depth_m_zscore`, `avg_braking_force_est_N_per_kg_zscore`,
`avg_propulsive_force_est_N_per_kg_zscore`, `peak_propulsive_force_est_N_per_kg_zscore`,
and `jump_momentum_selected_kg_m_s_zscore`: each is `(value − team mean)/population SD`
over available trials; zero only for valid observations with zero variance, NaN
for missing observations. These are descriptive, not normative quality scores.

## Verification and future validation

Implementation files:

- `vaila/cmj_pods.py` (new event/metric/QC/report helper)
- `vaila/vaila_and_jump.py` (pipeline, configuration, export and Team Batch integration)
- `tests/test_cmj_pods.py` (new regression tests)
- `docs/cmj_pods.md` (this complete dictionary and validation record)
- `docs/vaila_buttons/vertical-jump.md`
- `docs/vaila_buttons/vailajump.md`
- `vaila/help/vaila_and_jump.md`
- `vaila/help/vaila_and_jump.html`
- `vaila/help/vaila_and_jump_help.md`
- `vaila/help/vaila_and_jump_help.html`
- `vaila/help/index.md`
- `vaila/help/index.html`
- `README.md`

All edited/new Python headers and module help match the existing global banner
version 0.4.0, dated 14 September 2026. No installer or dependency changes are
needed. Concurrent pre-existing GetPixelVideo/main-window changes are outside this
extension.

```sh
MPLBACKEND=Agg .venv/bin/python -m pytest tests/test_cmj_pods.py tests/test_vaila_and_jump.py tests/test_vaila_and_jump_integration.py -q
.venv/bin/ruff check vaila/cmj_pods.py vaila/vaila_and_jump.py tests/test_cmj_pods.py
.venv/bin/ty check vaila/cmj_pods.py
```

Tests cover formulas and phase boundaries, isolated baseline spikes, mild noise,
kinetic/foot/fallback selection, unmasked transition QC, calibration/height/order
failures, low FPS, short phases, NaNs, full landmark CSV export, HTML/diagnostic
plot generation, TOML/CLI parity, and Team Batch reporting. Existing tests bracket
the extension. No force-platform comparison is claimed.

Verification on 14 September 2026: **109 tests passed (26 new + 83 existing)**;
Ruff lint/format and `git diff --check` passed. The new helper passed Ty.
The main module retains the same seven Ty diagnostics observed in its pre-change
version (six errors and one warning in existing path/Tk/numeric/animation code).
Two complete CLI runs with `DISPLAY` unset generated CSV, HTML, PNG and GIF outputs:
the real landmark fixture at 30 FPS reported limited temporal resolution; an
intentional 240 FPS mismatch triggered calibration failure and withheld new kinetic
scalars/mRSI. Golden values captured before the change verify ten historical core
metrics on the existing fixture. The new diagnostic plot was visually inspected.

Before external kinetic interpretation, collect synchronized video and force
platform CMJs spanning FPS, cameras, body sizes and jump strategies. Independently
annotate foot-off/contact; compare event errors in frames/ms, TTTO/mRSI/height,
relative force means/peaks, momentum sources and power with bias, limits of
agreement and repeatability. Quantify cutoff sensitivity without selecting a
filter merely to improve agreement. Validate threshold choices on held-out trials
and report failed detections, not only successful cases. RFD remains out of scope.
