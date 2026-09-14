"""Markerless CMJ events, PODS metrics and their shared report definitions.

Version: 0.3.141
Update Date: 14 September 2026

Consumes the existing filtered CoM and derivatives. No filtering, differentiation
or replacement of legacy events occurs here. Frames are zero-based row positions.
"""

from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GRAVITY = 9.81
PODS_DEFAULTS = {
    "movement_onset_velocity_threshold_m_s": 0.05,
    "movement_onset_displacement_threshold_m": 0.005,
    "movement_onset_noise_multiplier": 3.0,
    "movement_onset_min_duration_s": 0.03,
    "phase_min_duration_s": 0.03,
    "takeoff_agreement_tolerance_s": 0.04,
}
# Conservative artefact guards, not athlete performance thresholds.
MAX_CONTACT_FORCE_BW = 10.0
MIN_CONTACT_FORCE_BW = -0.5
MIN_PHASE_SAMPLES = 3
KINETIC_ZERO_BW = 0.1
KINETIC_EDGE_BW = 0.25
KINETIC_PERSISTENCE_S = 0.02

PODS_TEAM_METRICS = {
    "mrsi_AU": "mRSI [AU; dimensionally m/s]",
    "time_to_takeoff_s": "Time to takeoff [s]",
    "countermovement_depth_m": "Countermovement depth [m]",
    "avg_braking_force_est_N_per_kg": "Average braking force estimate [N/kg]",
    "avg_propulsive_force_est_N_per_kg": "Average propulsive force estimate [N/kg]",
    "peak_propulsive_force_est_N_per_kg": "Peak propulsive force estimate [N/kg]",
    "jump_momentum_selected_kg_m_s": "Selected jump momentum estimate [kg m/s]",
}

EVENT_DEFINITIONS = (
    (
        "movement_onset_frame",
        "Movement onset",
        "Persistent downward filtered CoM velocity and displacement beyond baseline noise",
    ),
    (
        "peak_downward_velocity_frame",
        "Peak downward CoM velocity / braking start",
        "Minimum filtered velocity between onset and the canonical bottom",
    ),
    (
        "minimum_displacement_frame",
        "Minimum CoM displacement / propulsion start",
        "Existing propulsion_start_frame; no second bottom detection",
    ),
    (
        "takeoff_frame",
        "CoM standing-height crossing (legacy)",
        "Upward baseline crossing; NOT actual foot-off",
    ),
    (
        "takeoff_frame_foot_contact",
        "Takeoff — last foot-off",
        "Last detected foot marker above its standing threshold",
    ),
    (
        "takeoff_frame_kinetic",
        "Takeoff — kinetic estimate",
        "Legacy last positive modelled GRF candidate; acceptance requires additional QC",
    ),
    (
        "takeoff_frame_selected",
        "Takeoff — selected",
        "QC-checked kinetic candidate, foot marker, or explicitly labelled CoM fallback",
    ),
    ("max_height_frame", "CoM apex", "Existing maximum CoM height event"),
    (
        "landing_frame_foot_contact",
        "Landing — first foot contact",
        "First detected foot marker returning to its standing threshold",
    ),
    (
        "landing_frame_selected",
        "Landing — selected",
        "First foot contact, otherwise labelled historical CoM fallback",
    ),
)

PHASES = ("standing", "unweighting", "braking", "propulsion", "flight", "landing_recovery")
PHASE_START_KEYS = (
    "movement_onset_frame",
    "peak_downward_velocity_frame",
    "minimum_displacement_frame",
    "takeoff_frame_selected",
    "landing_frame_selected",
)

DRIVER_WARNING = (
    "Force and power variables labelled 'estimated' are reconstructed from markerless "
    "center-of-mass kinematics using F = m(a + g). They are not direct force-platform "
    "measurements and are not interchangeable with force-plate kinetics without validation."
)


def _number(value: object) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _frame(value: object, n: int) -> int | None:
    number = _number(value)
    return int(number) if np.isfinite(number) and number.is_integer() and 0 <= number < n else None


def _robust_location_noise(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < MIN_PHASE_SAMPLES:
        return float("nan"), float("nan")
    median = float(np.median(values))
    return median, float(1.4826 * np.median(np.abs(values - median)))


def detect_movement_onset(
    y: np.ndarray, velocity: np.ndarray, bottom: int | None, fps: float, options: dict
) -> int | None:
    """First persistent velocity AND displacement departure after quiet stance.

    The complete configured baseline must precede the bottom. A moving baseline
    is rejected; silently relocating it would change the operator's calibration.
    """
    start = max(0, int(options.get("baseline_start_frame", 10)))
    end = int(options.get("baseline_end_frame", 20))
    if bottom is None or not (start + MIN_PHASE_SAMPLES <= end < bottom) or fps <= 0:
        return None
    position, y_noise = _robust_location_noise(y[start:end])
    speed, v_noise = _robust_location_noise(velocity[start:end])
    min_speed = options["movement_onset_velocity_threshold_m_s"]
    multiplier = options["movement_onset_noise_multiplier"]
    if not np.isfinite(position + speed) or abs(speed) > min_speed:
        return None
    v_threshold = max(min_speed, multiplier * v_noise)
    y_threshold = max(options["movement_onset_displacement_threshold_m"], multiplier * y_noise)
    required = max(2, int(np.ceil(fps * options["movement_onset_min_duration_s"])))
    mask = (velocity < speed - v_threshold) & (y < position - y_threshold)
    count = 0
    for i in range(end, bottom):
        count = count + 1 if mask[i] else 0
        if count >= required:
            return i - required + 1
    return None


def select_takeoff_event(
    events: dict,
    raw_force_relative: np.ndarray,
    velocity: np.ndarray,
    fps: float,
    gravity_status: str,
    tolerance_s: float,
) -> tuple[int | None, str, list[str]]:
    """Validate an existing kinetic candidate against unmasked model residuals.

    A positive force sample alone is not a zero-force transition. Require a small
    positive edge followed by persistent near-zero force, plus foot agreement
    when both candidates exist. Masked flight zeros must never validate it.
    """
    n = len(velocity)
    bottom = _frame(events.get("propulsion_start_frame"), n)
    apex = _frame(events.get("max_height_frame"), n)
    foot = _frame(events.get("takeoff_frame_foot_contact"), n)
    kinetic = _frame(events.get("takeoff_frame_kinetic"), n)
    notes = []

    def ordered(f: int | None) -> bool:
        return f is not None and bottom is not None and apex is not None and bottom < f < apex

    agreement = (
        foot is None
        or not ordered(foot)
        or kinetic is None
        or abs(kinetic - foot) / fps <= tolerance_s
    )
    if not agreement:
        notes.append("Kinetic and foot takeoff disagree beyond tolerance; review both events.")
    edge_ok = False
    if ordered(kinetic) and kinetic is not None and bottom is not None and apex is not None:
        persistence = max(2, int(np.ceil(fps * KINETIC_PERSISTENCE_S)))
        after = raw_force_relative[kinetic + 1 : kinetic + 1 + persistence] / GRAVITY
        contact = raw_force_relative[bottom : kinetic + 1] / GRAVITY
        edge_ok = (
            kinetic + persistence < apex
            and len(after) == persistence
            and np.all(np.isfinite(after))
            and np.all(np.abs(after) <= KINETIC_ZERO_BW)
            and 0 < raw_force_relative[kinetic] / GRAVITY <= KINETIC_EDGE_BW
            and np.all(np.isfinite(contact))
            and np.all((contact >= MIN_CONTACT_FORCE_BW) & (contact <= MAX_CONTACT_FORCE_BW))
            and velocity[kinetic] > 0
        )
    if gravity_status == "ok" and edge_ok and agreement:
        return kinetic, "kinetic_estimated_grf", notes
    if kinetic is not None:
        notes.append(
            "Kinetic candidate not accepted: zero-transition, calibration or agreement QC."
        )
    if ordered(foot):
        return foot, "foot_contact_marker", notes
    reference = _frame(events.get("takeoff_frame"), n)
    notes.append("No accepted true takeoff; CoM baseline crossing is only an explicit fallback.")
    return reference if ordered(reference) else None, "fallback_com_reference", notes


def assign_phase_labels(n: int, events: dict) -> np.ndarray:
    """Half-open intervals; invalid chronology remains unclassified (empty)."""
    labels = np.full(n, "", dtype=object)
    if not events.get("cmj_chronology_valid", False):
        return labels
    labels[:] = PHASES[0]
    for key, label in zip(PHASE_START_KEYS, PHASES[1:], strict=True):
        labels[int(events[key]) :] = label
    return labels


def _ordered_events(frames: tuple) -> bool:
    if any(f is None for f in frames):
        return False
    onset, downpeak, bottom, takeoff, apex, landing = (int(f) for f in frames)
    return onset < downpeak <= bottom < takeoff < apex < landing


def _enough_phase_samples(start: int | None, end: int | None, fps: float, duration: float) -> bool:
    return (
        start is not None
        and end is not None
        and fps > 0
        and end - start >= MIN_PHASE_SAMPLES
        and (end - start) / fps >= duration
    )


def calculate_pods(
    data: pd.DataFrame, legacy: dict, mass: float, fps: float, options: dict | None = None
) -> tuple[dict, dict, pd.DataFrame]:
    """Return canonical events, scalar metrics and preferred frame aliases.

    Legacy input is never mutated. Invalid new scalars are NaN. Durations use
    frame differences; averages use [start, end), excluding flight/touchdown.
    """
    opts = {**PODS_DEFAULTS, **(options or {})}
    n = len(data)
    events: dict[str, Any] = {
        key: _frame(legacy.get(key), n)
        for key in (
            "propulsion_start_frame",
            "takeoff_frame",
            "takeoff_frame_kinetic",
            "takeoff_frame_foot_contact",
            "max_height_frame",
            "landing_frame_foot_contact",
        )
    }
    mass_ok = np.isfinite(mass) and mass > 0
    fps_ok = np.isfinite(fps) and fps > 0
    sample_fps = fps if fps_ok else 1.0

    def signal(key: str) -> np.ndarray:
        return pd.to_numeric(
            data.get(key, pd.Series(np.nan, index=data.index)), errors="coerce"
        ).to_numpy(dtype=float)

    y, velocity, acceleration = (signal(k) for k in ("cg_y_m_filtered", "cg_vy", "cg_ay"))
    force, power = signal("force_vertical"), signal("power")
    airborne = data.get("airborne", pd.Series(False, index=data.index)).to_numpy(dtype=bool)
    relative_force = force / mass if mass_ok else np.full(n, np.nan)
    # Only algebra on the existing acceleration, for QC of unmasked flight residuals.
    raw_relative = acceleration + GRAVITY
    gravity_status = str(legacy.get("gravity_check_status", "insufficient_data"))
    calibrated = mass_ok and fps_ok and gravity_status != "suspect_fps_or_scale"
    bottom = events["propulsion_start_frame"]
    onset = detect_movement_onset(y, velocity, bottom, sample_fps, opts) if fps_ok else None
    downpeak = None
    if onset is not None and bottom is not None:
        segment = velocity[onset : bottom + 1]
        if np.isfinite(segment).all():
            downpeak = onset + int(np.argmin(segment))
    selected, source, notes = select_takeoff_event(
        events,
        raw_relative,
        velocity,
        sample_fps,
        gravity_status,
        opts["takeoff_agreement_tolerance_s"],
    )
    landing = events["landing_frame_foot_contact"]
    landing_source = "foot_contact_marker"
    if landing is None:
        landing = _frame(legacy.get("landing_frame"), n)
        landing_source = "fallback_com_reference"
        notes.append("Landing uses historical CoM crossing; true flight duration unavailable.")
    events.update(
        {
            "movement_onset_frame": onset,
            "peak_downward_velocity_frame": downpeak,
            "braking_start_frame": downpeak,
            "minimum_displacement_frame": bottom,
            "takeoff_frame_selected": selected,
            "takeoff_source": source,
            "landing_frame_selected": landing,
            "landing_source": landing_source,
        }
    )
    apex = events["max_height_frame"]
    order = (onset, downpeak, bottom, selected, apex, landing)
    chronology = _ordered_events(order) and fps_ok
    events["cmj_chronology_valid"] = chronology
    if not chronology:
        notes.append(
            "Missing onset or invalid chronology: onset < downpeak <= bottom < takeoff < apex < landing required."
        )
    phase_status = "ok" if chronology else "phase_detection_failed"
    too_short = chronology and any(
        not _enough_phase_samples(start, end, fps, opts["phase_min_duration_s"])
        for start, end in ((downpeak, bottom), (bottom, selected))
    )
    if chronology and (fps <= 60 or too_short):
        phase_status = "limited_temporal_resolution"
        notes.append(
            "Temporal resolution is limited; report frame quantization, not sub-frame precision."
        )
    if source == "fallback_com_reference" or landing_source == "fallback_com_reference":
        phase_status = "phase_detection_failed"
    metrics = dict(events)
    for key, time_key in (
        ("movement_onset_frame", "movement_onset_time_s"),
        ("peak_downward_velocity_frame", "peak_downward_velocity_time_s"),
        ("braking_start_frame", "braking_start_time_s"),
        ("minimum_displacement_frame", "minimum_displacement_time_s"),
        ("takeoff_frame_selected", "takeoff_time_selected_s"),
        ("landing_frame_selected", "landing_time_selected_s"),
    ):
        metrics[time_key] = events[key] / fps if events[key] is not None and fps_ok else np.nan
    metrics["mass_kg"] = mass
    metrics["peak_downward_velocity_m_s"] = velocity[downpeak] if downpeak is not None else np.nan
    metrics["countermovement_depth_m"] = (
        abs(_number(legacy.get("squat_depth_m"))) if calibrated else np.nan
    )
    for key, start, end in (
        ("unweighting_duration_s", onset, downpeak),
        ("braking_duration_s", downpeak, bottom),
        ("propulsive_duration_s", bottom, selected),
        ("time_to_takeoff_s", onset, selected),
        ("flight_duration_selected_s", selected, landing),
    ):
        metrics[key] = (
            (end - start) / fps if chronology and end is not None and start is not None else np.nan
        )
    if source == "fallback_com_reference":
        metrics["time_to_takeoff_s"] = np.nan
        metrics["propulsive_duration_s"] = np.nan
    if source == "fallback_com_reference" or landing_source == "fallback_com_reference":
        metrics["flight_duration_selected_s"] = np.nan
    kinetic, foot = events["takeoff_frame_kinetic"], events["takeoff_frame_foot_contact"]
    diff = kinetic - foot if kinetic is not None and foot is not None else np.nan
    metrics["takeoff_kinetic_vs_foot_diff_frames"] = diff
    metrics["takeoff_kinetic_vs_foot_diff_s"] = diff / fps if fps_ok else np.nan
    metrics["cmj_phase_qc_status"] = phase_status
    metrics["cmj_phase_qc_note"] = " ".join(notes) or "Canonical events are ordered."

    height = _number(legacy.get("height_qc_recommended_m"))
    height_ok = height > 0 and legacy.get("height_qc_recommended_status") in {
        "plausible",
        "very_high_plausible",
        "ok",
    }
    phase_ok = phase_status in {"ok", "limited_temporal_resolution"}
    metrics["mrsi_height_source"] = legacy.get("height_qc_recommended_source", "unavailable")
    metrics["mrsi_takeoff_source"] = source
    mrsi_status = phase_status
    if not height_ok:
        mrsi_status = "height_qc_failed"
    if not calibrated:
        mrsi_status = "calibration_failed"
    if n < MIN_PHASE_SAMPLES:
        mrsi_status = "insufficient_data"
    metrics["mrsi_qc_status"] = mrsi_status
    metrics["mrsi_AU"] = (
        height / metrics["time_to_takeoff_s"] if height_ok and calibrated and phase_ok else np.nan
    )
    height_velocity = float(np.sqrt(2 * GRAVITY * height)) if height_ok and calibrated else np.nan
    com_velocity = (
        velocity[selected]
        if calibrated and phase_ok and selected is not None and velocity[selected] > 0
        else np.nan
    )
    metrics["takeoff_velocity_height_derived_m_s"] = height_velocity
    metrics["takeoff_velocity_com_est_m_s"] = com_velocity
    metrics["jump_momentum_height_derived_kg_m_s"] = mass * height_velocity if mass_ok else np.nan
    metrics["jump_momentum_com_est_kg_m_s"] = mass * com_velocity if mass_ok else np.nan
    # Height-derived is intentionally preferred and labelled; never imply it is independent of h.
    momentum_source = (
        "height_derived"
        if np.isfinite(height_velocity)
        else "com_est"
        if np.isfinite(com_velocity)
        else "unavailable"
    )
    metrics["jump_momentum_source"] = momentum_source
    metrics["jump_momentum_selected_kg_m_s"] = metrics.get(
        f"jump_momentum_{momentum_source}_kg_m_s", np.nan
    )

    kinetic_notes = []
    kinetic_status = "ok"
    if not calibrated:
        kinetic_status = "calibration_failed"
    elif not phase_ok:
        kinetic_status = "phase_detection_failed"
    elif gravity_status != "ok":
        kinetic_status = "insufficient_data"
        kinetic_notes.append("Flight gravity check unavailable; kinetic scale is unverified.")
    if chronology and calibrated:
        contact = raw_relative[onset:selected] / GRAVITY
        if not np.isfinite(contact).all():
            kinetic_status = "insufficient_data"
        elif np.any((contact < MIN_CONTACT_FORCE_BW) | (contact > MAX_CONTACT_FORCE_BW)):
            kinetic_status = "force_plausibility_failed"
            kinetic_notes.append(
                "Contact estimate outside -0.5 to 10 body weights; inspect tracking. Values are not clipped."
            )
    if kinetic_status == "ok" and (fps <= 60 or too_short):
        kinetic_status = "limited_temporal_resolution"
        kinetic_notes.append(
            "30/60 Hz or short phases: temporal resolution limits force-phase estimates."
        )
    if too_short:
        kinetic_notes.append(
            "A braking or propulsion phase has too few samples/duration; its aggregates are unavailable."
        )
    if np.isfinite(height_velocity + com_velocity) and abs(com_velocity - height_velocity) > max(
        0.5, 0.25 * height_velocity
    ):
        kinetic_notes.append(
            "CoM and height-derived takeoff velocity disagree (>0.5 m/s or 25%, whichever is larger); height-derived momentum remains explicitly selected."
        )
    metrics["kinetic_metrics_qc_status"] = kinetic_status
    metrics["kinetic_metrics_qc_note"] = (
        " ".join(kinetic_notes + notes)
        or "Scale, phase sampling and gross force plausibility checks passed; markerless estimates only."
    )
    kinetics_ok = kinetic_status in {"ok", "limited_temporal_resolution"}
    for phase, start, end in (("braking", downpeak, bottom), ("propulsive", bottom, selected)):
        enough = chronology and _enough_phase_samples(start, end, fps, opts["phase_min_duration_s"])
        for signal_name, values, unit in (("force", force, "N"), ("power", power, "W")):
            segment = values[start:end] if chronology else np.array([])
            if chronology:
                segment = segment[~airborne[start:end]]
            valid = kinetics_ok and enough and np.isfinite(segment).sum() >= MIN_PHASE_SAMPLES
            for statistic, reducer in (("avg", np.nanmean), ("peak", np.nanmax)):
                key = f"{statistic}_{phase}_{signal_name}_est_{unit}"
                metrics[key] = float(reducer(segment)) if valid else np.nan
                metrics[f"{key}_per_kg"] = metrics[key] / mass if mass_ok else np.nan
    metrics["force_at_min_displacement_est_N"] = (
        force[bottom] if kinetics_ok and bottom is not None else np.nan
    )
    metrics["force_at_min_displacement_est_N_per_kg"] = (
        metrics["force_at_min_displacement_est_N"] / mass if mass_ok else np.nan
    )
    frames = pd.DataFrame(
        {
            "force_vertical_est_N": force,
            "force_vertical_est_N_per_kg": relative_force,
            "power_est_W": power,
            "power_est_W_per_kg": power / mass if mass_ok else np.full(n, np.nan),
            "cmj_phase": assign_phase_labels(n, events),
        },
        index=data.index,
    )
    return events, metrics, frames


def event_rows_html(events: dict, fps: float) -> str:
    rows = []
    for key, label, definition in EVENT_DEFINITIONS:
        frame = _number(events.get(key))
        frame_text = f"{frame:.0f}" if np.isfinite(frame) else "N/A"
        time = f"{frame / fps:.3f}" if np.isfinite(frame) and fps > 0 else "N/A"
        if key == "takeoff_frame_selected":
            definition += f" ({events.get('takeoff_source', 'unavailable')})"
        if key == "landing_frame_selected":
            definition += f" ({events.get('landing_source', 'unavailable')})"
        rows.append(
            f"<tr><td>{escape(label)}</td><td>{frame_text}</td><td>{time}</td><td>{escape(definition)}</td></tr>"
        )
    return "".join(rows)


def pods_report_html(metrics: dict) -> str:
    groups = {
        "Person": [("mass_kg", "Body mass [kg]")],
        "Outcome": [
            ("height_qc_recommended_m", "Recommended height [m]"),
            ("takeoff_velocity_height_derived_m_s", "Height-derived takeoff velocity [m/s]"),
            ("takeoff_velocity_com_est_m_s", "CoM takeoff velocity estimate [m/s]"),
            ("jump_momentum_selected_kg_m_s", "Selected momentum estimate [kg m/s]"),
            ("mrsi_AU", "mRSI [AU; dimensionally m/s]"),
        ],
        "Driver — markerless estimates": [
            ("avg_braking_force_est_N_per_kg", "Average braking force estimate [N/kg]"),
            ("force_at_min_displacement_est_N_per_kg", "Bottom force estimate [N/kg]"),
            ("avg_propulsive_force_est_N_per_kg", "Average propulsive force estimate [N/kg]"),
            ("peak_propulsive_force_est_N_per_kg", "Peak propulsive force estimate [N/kg]"),
            ("avg_propulsive_power_est_W_per_kg", "Average propulsive power estimate [W/kg]"),
            ("peak_propulsive_power_est_W_per_kg", "Peak propulsive power estimate [W/kg]"),
            ("max_power_W_per_kg", "Legacy peak power estimate [W/kg; foot-reference interval]"),
            (
                "power_avg_propulsion_W",
                "Legacy energy/time power estimate [W; baseline-crossing duration]",
            ),
        ],
        "Strategy": [
            ("time_to_takeoff_s", "Time to takeoff [s]"),
            ("countermovement_depth_m", "Countermovement depth [m; positive]"),
            ("unweighting_duration_s", "Unweighting [s]"),
            ("braking_duration_s", "Braking [s]"),
            ("propulsive_duration_s", "Propulsion [s]"),
        ],
    }
    html = '<section id="cmj-pods"><h2>CMJ Performance Profile — PODS</h2>'
    for group, fields in groups.items():
        html += f"<h3>{group}</h3>"
        if group.startswith("Driver"):
            html += f'<p class="qc-note"><strong>{escape(DRIVER_WARNING)}</strong></p>'
        html += "<table><tr><th>Metric</th><th>Value</th></tr>"
        for key, label in fields:
            number = _number(metrics.get(key))
            text = f"{number:.3f}" if np.isfinite(number) else "N/A"
            if group.startswith("Driver") and metrics.get("kinetic_metrics_qc_status") not in {
                "ok",
                "limited_temporal_resolution",
            }:
                text = "N/A (kinetic QC)"
            html += f"<tr><td>{label}</td><td>{text}</td></tr>"
        html += "</table>"
    for key in (
        "takeoff_source",
        "jump_momentum_source",
        "mrsi_height_source",
        "mrsi_qc_status",
        "cmj_phase_qc_status",
        "cmj_phase_qc_note",
        "kinetic_metrics_qc_status",
        "kinetic_metrics_qc_note",
    ):
        html += f"<p><strong>{key}:</strong> {escape(str(metrics.get(key, 'unavailable')))}</p>"
    return html + (
        "<p>Jump height describes the outcome; mRSI combines height with movement-onset-to-takeoff time. "
        "Similar heights can arise from different timing and countermovement strategies. "
        "Height-derived velocity and momentum are not independent of height, nor impulse-derived measurements. "
        "Force uses second derivatives and depends strongly on the existing 12 Hz filter; height and time are generally less derivative-sensitive. "
        "Cross-system comparisons require validation. These results do not diagnose weakness, fatigue, injury risk or neuromuscular deficits from a single jump.</p>"
        '<p>Conceptual reference: <a href="https://doi.org/10.3390/biomechanics5020032">Ripley et al. (2025)</a>; '
        "their force-platform study supplies no normative thresholds for this markerless report.</p></section>"
    )


def plot_pods_diagnostic(data: pd.DataFrame, events: dict, fps: float, output: Path) -> str:
    """Four aligned panels, driven exclusively by the exported event dictionary."""
    import matplotlib.pyplot as plt

    t = np.arange(len(data)) / fps
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, key, label in zip(
        axes,
        ("cg_y_m_filtered", "cg_vy", "force_vertical_est_N", "power_est_W"),
        ("CoM displacement [m]", "CoM velocity [m/s]", "Estimated GRF [N]", "Estimated power [W]"),
        strict=True,
    ):
        values = data[key].to_numpy(dtype=float).copy()
        if key == "cg_y_m_filtered":
            values -= (
                float(data["reference_cg_y"].iloc[0]) if "reference_cg_y" in data else values[0]
            )
        ax.plot(t, values, color="#245879", linewidth=1)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        labels = assign_phase_labels(len(data), events)
        for phase, color in zip(
            PHASES[1:5], ("#f5ce83", "#9abdeb", "#b0d6bd", "#d7b8df"), strict=True
        ):
            indices = np.flatnonzero(labels == phase)
            if len(indices):
                ax.axvspan(
                    indices[0] / fps, (indices[-1] + 1) / fps, color=color, alpha=0.4, label=phase
                )
        for key, label, _ in EVENT_DEFINITIONS:
            if key in {"takeoff_frame", "landing_frame_foot_contact"}:
                continue
            frame = events.get(key)
            if frame is not None:
                ax.axvline(
                    frame / fps,
                    linestyle=":" if "kinetic" in key or "foot_contact" in key else "--",
                    alpha=0.6,
                    linewidth=0.8,
                    label=label,
                )
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(
        "CMJ phases — markerless force/power estimates; inspect scalar QC before interpretation"
    )
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(output)
