"""
===============================================================================
vaila_deadlift_imu.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 June 2026
Update Date: 09 June 2026
Version: 0.3.50
Python Version: 3.12.x

Description:
------------
Standalone IMU-only Deadlift / RDL analysis pipeline using AHRS sensor fusion.

This module is a dedicated companion to :mod:`vaila.vaila_deadlift` focused on
barbell-mounted IMU streams (3-axis accelerometer + 3-axis gyroscope, with an
optional 3-axis magnetometer). It replaces the simplified "project the raw
accelerometer onto the static gravity axis" approach in ``vaila_deadlift.py``
with a proper orientation tracker so that lift velocity / power / work survive
arbitrary barbell rotations during the pull.

Two industry-standard AHRS filters are ported from the x-io Technologies
open-source C reference:

* **Madgwick** - gradient-descent quaternion update (Madgwick S.O.H., 2010,
  *An efficient orientation filter for inertial and inertial/magnetic sensor
  arrays*).
* **Mahony** - complementary-filter style proportional+integral quaternion
  update (Mahony R., Hamel T., Pflimlin J.-M., 2008).

References
----------
* x-io Open-Source IMU and AHRS Algorithms:
  https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
* Reference C code (vendored under
  ``tests/Deadlift/madgwick_algorithm_c/MadgwickAHRS`` /
  ``MahonyAHRS``):
  https://github.com/xioTechnologies/Fusion/tree/main
* Medium tutorial used as cross-check:
  https://medium.com/@k66115704/imu-madgwick-filter-explanation-556fbe7f02e3

Quaternion modeling - publication history & tribute
---------------------------------------------------
The quaternion-based 3D kinematic modeling used here traces back to a close
academic collaboration with Prof. René Jean Brenzikofer (UNICAMP), which led to
the first publication of quaternion-based three-dimensional modeling data in the
book *"Modelos Matemáticos nas Ciências Não-Exatas, vol. 1"* (Editora Blucher,
2007). This module is dedicated to his memory.

BibTeX references
^^^^^^^^^^^^^^^^^

.. code-block:: bibtex

    @book{nogueira2007modelos,
      title     = {Modelos matem\'aticos nas ci\\^encias n\\~ao-exatas - vol. 1},
      author    = {Nogueira, Eduardo Arantes and Martins, Luiz Eduardo Barreto
                   and Brenzikofer, Ren\'e},
      year      = {2007},
      publisher = {Editora Blucher},
      isbn      = {9788521204190},
      url       = {https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190}
    }

    @phdthesis{santiago2009rotaccoes,
      title    = {Rota\\c{c}\\~oes tridimensionais em biomec\\^anica via
                  quat\'ernions: aplica\\c{c}\\~oes na an\'alise dos movimentos
                  esportivos},
      author   = {Santiago, Paulo Roberto Pereira},
      school   = {Universidade Estadual Paulista (Unesp),
                  Instituto de Bioci\\^encias de Rio Claro},
      year     = {2009},
      url      = {http://hdl.handle.net/11449/100404}
    }

Tribute & social media coverage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Instagram post (tribute & publication history):
  https://www.instagram.com/p/DZLogIJoFbF/
* LinkedIn post (publication history & tribute to Prof. Brenzikofer):
  https://www.linkedin.com/posts/paulo-roberto-pereira-santiago-132619112_hist%C3%B3rico-de-publica%C3%A7%C3%A3o-e-homenagem-ao-prof-ugcPost-7468670091845472257-3tzD/
* Doctoral thesis PDF (open access, UNESP repository):
  https://repositorio.unesp.br/server/api/core/bitstreams/41603fa7-545b-4e74-a045-57ce94885e0c/content
* Reference book (Editora Blucher, 2007):
  https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190

Quaternion convention
---------------------
``q = [q0, q1, q2, q3] = [w, x, y, z]`` is the rotation that takes a vector
expressed in the **sensor body frame** into the **Earth (navigation) frame**::

    v_earth = R(q) @ v_sensor

where ``R(q)`` is the standard Hamilton rotation matrix (the same convention
used by the x-io C reference). When the sensor is at rest with the body Z axis
pointing up, the accelerometer reads roughly ``[0, 0, +g]`` in the body frame
and the rotated reading is ``[0, 0, +g]`` in the Earth frame.

Pipeline
--------
1. Detect a static window at the start of the recording, estimate initial
   pitch / roll from the gravity vector (yaw is left at 0 - magnetometer-free
   IMUs cannot recover absolute heading).
2. Run Madgwick (or Mahony) sample-by-sample to track the quaternion.
3. Rotate the raw accelerometer to the Earth frame, subtract the
   measured static gravity to obtain the **linear acceleration** vector and,
   in particular, the linear vertical acceleration ``a_v(t)``.
4. Segment repetitions from the filtered vertical acceleration / velocity
   envelope (zero-velocity update style drift correction per rep).
5. Per-rep metrics: peak / mean velocity, peak / mean power, work, ROM,
   peak / mean force - using either the barbell weight only or the
   total system mass.
6. Generate PNG plots and a self-contained HTML report.

Inputs
------
CSV with at least the columns::

    acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z

Accelerometer is expected in m/s^2 (gravity included). Gyroscope is accepted
in deg/s by default and converted to rad/s; pass ``--gyro-units rad`` if
the file is already in rad/s. Optional magnetometer columns ``mag_x``,
``mag_y``, ``mag_z`` (units arbitrary, any consistent unit) enable the full
AHRS branch with magnetic heading correction.

Run modes
---------
* GUI (single Tkinter form, batches every ``*.csv`` in a directory):
  ``uv run python vaila/vaila_deadlift_imu.py``
* CLI - single file (runs Madgwick + Mahony by default):
  ``uv run python vaila/vaila_deadlift_imu.py -i deadlift_imu.csv``
* CLI - single file with explicit params file:
  ``uv run python vaila/vaila_deadlift_imu.py -i deadlift_imu.csv -p deadlift_parameters.txt``
* CLI - whole directory (batch, mirrors the GUI, runs both filters):
  ``uv run python vaila/vaila_deadlift_imu.py -d /path/to/csv_dir -o /path/to/out -p /path/to/deadlift_parameters.txt``
* Show all CLI flags:
  ``uv run python vaila/vaila_deadlift_imu.py --help``
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import tkinter as tk
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, ttk

import matplotlib

matplotlib.use("Agg")  # avoid Qt backend issues in CLI mode

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import butter, filtfilt, find_peaks  # noqa: E402

try:
    import tomllib

    _toml_reader: object | None = tomllib
except Exception:  # pragma: no cover - Python <3.11 fallback
    _toml_reader = None

# Global vailá version (keep in sync with vaila.py banner / module header).
VAILA_VERSION = "0.3.50"
DEFAULT_IMU_CUTOFF_HZ = 4.0


# ---------------------------------------------------------------------------
# AHRS quaternion filters (Madgwick and Mahony)
# ---------------------------------------------------------------------------


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (sensor frame -> Earth frame) for quaternion [w, x, y, z]."""
    q0, q1, q2, q3 = q
    return np.array(
        [
            [
                q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
            ],
        ]
    )


def _quat_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion [w,x,y,z] to roll / pitch / yaw (rad), aerospace ZYX."""
    q0, q1, q2, q3 = q
    roll = math.atan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1 * q1 + q2 * q2))
    sinp = 2.0 * (q0 * q2 - q3 * q1)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2 * q2 + q3 * q3))
    return roll, pitch, yaw


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Aerospace ZYX (yaw-pitch-roll) Euler -> quaternion [w,x,y,z]."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy
    return np.array([q0, q1, q2, q3], dtype=float)


def estimate_initial_quaternion(acc_static: np.ndarray) -> np.ndarray:
    """Initial quaternion from a static accelerometer reading (gravity).

    Uses ``roll = atan2(ay, az)``, ``pitch = atan2(-ax, sqrt(ay^2 + az^2))``,
    ``yaw = 0``. ``acc_static`` is the mean accelerometer vector over a static
    window in **sensor frame**, with gravity included.
    """
    ax, ay, az = acc_static
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az) + 1e-12)
    return _euler_to_quat(roll, pitch, 0.0)


@dataclass
class MadgwickAHRS:
    """Madgwick AHRS filter ported from the x-io C reference (IMU + AHRS)."""

    sample_period: float
    beta: float = 0.1
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    def update_imu(self, gx: float, gy: float, gz: float, ax: float, ay: float, az: float) -> None:
        """Gyro+accelerometer update (no magnetometer)."""
        q0, q1, q2, q3 = self.q
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        if not (ax == 0.0 and ay == 0.0 and az == 0.0):
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm > 0:
                ax /= norm
                ay /= norm
                az /= norm

                _2q0 = 2.0 * q0
                _2q1 = 2.0 * q1
                _2q2 = 2.0 * q2
                _2q3 = 2.0 * q3
                _4q0 = 4.0 * q0
                _4q1 = 4.0 * q1
                _4q2 = 4.0 * q2
                _8q1 = 8.0 * q1
                _8q2 = 8.0 * q2
                q0q0 = q0 * q0
                q1q1 = q1 * q1
                q2q2 = q2 * q2
                q3q3 = q3 * q3

                s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay
                s1 = (
                    _4q1 * q3q3
                    - _2q3 * ax
                    + 4.0 * q0q0 * q1
                    - _2q0 * ay
                    - _4q1
                    + _8q1 * q1q1
                    + _8q1 * q2q2
                    + _4q1 * az
                )
                s2 = (
                    4.0 * q0q0 * q2
                    + _2q0 * ax
                    + _4q2 * q3q3
                    - _2q3 * ay
                    - _4q2
                    + _8q2 * q1q1
                    + _8q2 * q2q2
                    + _4q2 * az
                )
                s3 = 4.0 * q1q1 * q3 - _2q1 * ax + 4.0 * q2q2 * q3 - _2q2 * ay
                s_norm = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
                if s_norm > 0:
                    s0 /= s_norm
                    s1 /= s_norm
                    s2 /= s_norm
                    s3 /= s_norm

                    qDot1 -= self.beta * s0
                    qDot2 -= self.beta * s1
                    qDot3 -= self.beta * s2
                    qDot4 -= self.beta * s3

        q0 += qDot1 * self.sample_period
        q1 += qDot2 * self.sample_period
        q2 += qDot3 * self.sample_period
        q3 += qDot4 * self.sample_period

        n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if n > 0:
            self.q = np.array([q0 / n, q1 / n, q2 / n, q3 / n])

    def update(
        self,
        gx: float,
        gy: float,
        gz: float,
        ax: float,
        ay: float,
        az: float,
        mx: float,
        my: float,
        mz: float,
    ) -> None:
        """Full AHRS update (gyro + acc + mag). Falls back to IMU if mag is zero."""
        if mx == 0.0 and my == 0.0 and mz == 0.0:
            self.update_imu(gx, gy, gz, ax, ay, az)
            return

        q0, q1, q2, q3 = self.q
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        if not (ax == 0.0 and ay == 0.0 and az == 0.0):
            na = math.sqrt(ax * ax + ay * ay + az * az)
            nm = math.sqrt(mx * mx + my * my + mz * mz)
            if na > 0 and nm > 0:
                ax /= na
                ay /= na
                az /= na
                mx /= nm
                my /= nm
                mz /= nm

                _2q0mx = 2.0 * q0 * mx
                _2q0my = 2.0 * q0 * my
                _2q0mz = 2.0 * q0 * mz
                _2q1mx = 2.0 * q1 * mx
                _2q0 = 2.0 * q0
                _2q1 = 2.0 * q1
                _2q2 = 2.0 * q2
                _2q3 = 2.0 * q3
                _2q0q2 = 2.0 * q0 * q2
                _2q2q3 = 2.0 * q2 * q3
                q0q0 = q0 * q0
                q0q1 = q0 * q1
                q0q2 = q0 * q2
                q0q3 = q0 * q3
                q1q1 = q1 * q1
                q1q2 = q1 * q2
                q1q3 = q1 * q3
                q2q2 = q2 * q2
                q2q3 = q2 * q3
                q3q3 = q3 * q3

                hx = (
                    mx * q0q0
                    - _2q0my * q3
                    + _2q0mz * q2
                    + mx * q1q1
                    + _2q1 * my * q2
                    + _2q1 * mz * q3
                    - mx * q2q2
                    - mx * q3q3
                )
                hy = (
                    _2q0mx * q3
                    + my * q0q0
                    - _2q0mz * q1
                    + _2q1mx * q2
                    - my * q1q1
                    + my * q2q2
                    + _2q2 * mz * q3
                    - my * q3q3
                )
                _2bx = math.sqrt(hx * hx + hy * hy)
                _2bz = (
                    -_2q0mx * q2
                    + _2q0my * q1
                    + mz * q0q0
                    + _2q1mx * q3
                    - mz * q1q1
                    + _2q2 * my * q3
                    - mz * q2q2
                    + mz * q3q3
                )
                _4bx = 2.0 * _2bx
                _4bz = 2.0 * _2bz

                s0 = (
                    -_2q2 * (2.0 * q1q3 - _2q0q2 - ax)
                    + _2q1 * (2.0 * q0q1 + _2q2q3 - ay)
                    - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                    + (-_2bx * q3 + _2bz * q1) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                    + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
                )
                s1 = (
                    _2q3 * (2.0 * q1q3 - _2q0q2 - ax)
                    + _2q0 * (2.0 * q0q1 + _2q2q3 - ay)
                    - 4.0 * q1 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az)
                    + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                    + (_2bx * q2 + _2bz * q0) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                    + (_2bx * q3 - _4bz * q1)
                    * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
                )
                s2 = (
                    -_2q0 * (2.0 * q1q3 - _2q0q2 - ax)
                    + _2q3 * (2.0 * q0q1 + _2q2q3 - ay)
                    - 4.0 * q2 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az)
                    + (-_4bx * q2 - _2bz * q0)
                    * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                    + (_2bx * q1 + _2bz * q3) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                    + (_2bx * q0 - _4bz * q2)
                    * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
                )
                s3 = (
                    _2q1 * (2.0 * q1q3 - _2q0q2 - ax)
                    + _2q2 * (2.0 * q0q1 + _2q2q3 - ay)
                    + (-_4bx * q3 + _2bz * q1)
                    * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
                    + (-_2bx * q0 + _2bz * q2) * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
                    + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
                )
                s_norm = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
                if s_norm > 0:
                    s0 /= s_norm
                    s1 /= s_norm
                    s2 /= s_norm
                    s3 /= s_norm
                    qDot1 -= self.beta * s0
                    qDot2 -= self.beta * s1
                    qDot3 -= self.beta * s2
                    qDot4 -= self.beta * s3

        q0 += qDot1 * self.sample_period
        q1 += qDot2 * self.sample_period
        q2 += qDot3 * self.sample_period
        q3 += qDot4 * self.sample_period
        n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if n > 0:
            self.q = np.array([q0 / n, q1 / n, q2 / n, q3 / n])


@dataclass
class MahonyAHRS:
    """Mahony complementary-style AHRS filter ported from the x-io C reference."""

    sample_period: float
    two_kp: float = 2.0 * 0.5
    two_ki: float = 2.0 * 0.0
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    _ifb: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def update_imu(self, gx: float, gy: float, gz: float, ax: float, ay: float, az: float) -> None:
        q0, q1, q2, q3 = self.q
        if not (ax == 0.0 and ay == 0.0 and az == 0.0):
            na = math.sqrt(ax * ax + ay * ay + az * az)
            if na > 0:
                ax /= na
                ay /= na
                az /= na
                halfvx = q1 * q3 - q0 * q2
                halfvy = q0 * q1 + q2 * q3
                halfvz = q0 * q0 - 0.5 + q3 * q3
                halfex = ay * halfvz - az * halfvy
                halfey = az * halfvx - ax * halfvz
                halfez = ax * halfvy - ay * halfvx

                if self.two_ki > 0.0:
                    self._ifb[0] += self.two_ki * halfex * self.sample_period
                    self._ifb[1] += self.two_ki * halfey * self.sample_period
                    self._ifb[2] += self.two_ki * halfez * self.sample_period
                    gx += self._ifb[0]
                    gy += self._ifb[1]
                    gz += self._ifb[2]
                else:
                    self._ifb[:] = 0.0

                gx += self.two_kp * halfex
                gy += self.two_kp * halfey
                gz += self.two_kp * halfez

        gx *= 0.5 * self.sample_period
        gy *= 0.5 * self.sample_period
        gz *= 0.5 * self.sample_period
        qa, qb, qc = q0, q1, q2
        q0 += -qb * gx - qc * gy - q3 * gz
        q1 += qa * gx + qc * gz - q3 * gy
        q2 += qa * gy - qb * gz + q3 * gx
        q3 += qa * gz + qb * gy - qc * gx
        n = math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        if n > 0:
            self.q = np.array([q0 / n, q1 / n, q2 / n, q3 / n])


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _parse_locale_float(value) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a float")
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("Empty float")
    return float(text)


def _load_context_from_toml(base_dir: Path | None = None) -> dict:
    """Load barbell weight / sample rate / mass etc. from optional TOML beside the CSV."""
    search_paths: list[Path] = []
    if base_dir:
        search_paths.append(base_dir / "vaila_deadlift_config.toml")
    search_paths.append(Path(__file__).parent / "vaila_deadlift_config.toml")

    for p in search_paths:
        if p.exists():
            try:
                reader = _toml_reader
                if reader is None:
                    import toml

                    data = toml.load(str(p))
                else:
                    with open(p, "rb") as f:
                        data = reader.load(f)  # ty: ignore[unresolved-attribute]
                cfg = data.get("deadlift_context", {})
                return {
                    "mass_kg": _parse_locale_float(cfg.get("mass_kg", 75.0)),
                    "fps": _parse_locale_float(cfg.get("fps", 30.0)),
                    "weight_kg": _parse_locale_float(cfg.get("weight_kg", 20.0)),
                    "use_total_mass_for_power": bool(cfg.get("use_total_mass_for_power", True)),
                    "imu_fps": _parse_locale_float(cfg.get("imu_fps", 25.0)),
                }
            except Exception:
                pass
    return {}


def _parse_parameters_file(path: Path) -> dict:
    """Parse a ``deadlift_parameters.txt`` file.

    Two formats are accepted:

    * **New CSV-style header format** (preferred, single row of values)::

          subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz
          75.0,1.75,20.0,25.0

      Known column aliases are normalised to the canonical keys
      ``subject_mass_kg`` / ``subject_height_m`` / ``deadlift_mass_kg`` /
      ``fs_hz``. Any extra column is preserved with its original header name.

    * **Legacy key,value format** (one parameter per line)::

          weight,20
          real_repetition_count,10

      Each line is stored as ``key -> value`` (string).

    The returned dict always carries one extra key ``__source__`` pointing to
    the resolved file path (string) so callers can report it back to the user.
    """
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return {}

    # Aliases tolerated in the new format (case-insensitive). The canonical
    # column names are the keys.
    alias_map = {
        "subject_mass_kg": {
            "subject_mass_kg",
            "subject_mass",
            "body_mass_kg",
            "body_mass",
            "mass_kg",
            "mass",
            "athlete_mass_kg",
            "athlete_mass",
        },
        "subject_height_m": {
            "subject_height_m",
            "subject_height",
            "height_m",
            "height",
            "athlete_height_m",
            "athlete_height",
            "stature_m",
            "stature",
        },
        "deadlift_mass_kg": {
            "deadlift_mass_kg",
            "deadlift_mass",
            "barbell_kg",
            "barbell_mass_kg",
            "weight_kg",
            "weight",
        },
        "fs_hz": {
            "fs_hz",
            "fs",
            "fps",
            "imu_fps",
            "sample_rate",
            "sample_rate_hz",
            "sampling_rate",
            "sampling_rate_hz",
        },
        "real_repetition_count": {
            "real_repetition_count",
            "repetition_count",
            "rep_count",
            "reps",
        },
    }

    first = lines[0]
    has_header = (
        "," in first
        and not first.split(",", 1)[1].strip().replace(".", "", 1).replace("-", "", 1).isdigit()
    )

    params: dict = {"__source__": str(path)}

    if has_header and len(lines) >= 2:
        headers = [h.strip() for h in first.split(",")]
        values = [v.strip() for v in lines[1].split(",")]
        if len(values) == len(headers):
            for h, v in zip(headers, values, strict=False):
                canonical = h.lower()
                for canon, aliases in alias_map.items():
                    if canonical in aliases:
                        canonical = canon
                        break
                params[canonical] = v
            # Back-compat shortcut: expose ``weight`` for callers that still
            # read the legacy key.
            if "deadlift_mass_kg" in params and "weight" not in params:
                params["weight"] = params["deadlift_mass_kg"]
            return params

    # Legacy ``key,value`` per-line format.
    for line in lines:
        if "," not in line:
            continue
        key, val = line.split(",", 1)
        params[key.strip().lower()] = val.strip()
    return params


def _load_parameters_file(base_dir: Path, explicit_path: str | os.PathLike | None = None) -> dict:
    """Load ``deadlift_parameters.txt`` from an explicit path or by searching.

    If ``explicit_path`` is provided and exists, it is used directly.
    Otherwise the function looks for a file named ``deadlift_parameters.txt``
    in ``base_dir`` and up to three parent directories (legacy behaviour).
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.exists():
            return _parse_parameters_file(p)

    search_dirs = [base_dir, *base_dir.parents[:3]]
    for directory in search_dirs:
        p = directory / "deadlift_parameters.txt"
        if p.exists():
            return _parse_parameters_file(p)
    return {}


# ---------------------------------------------------------------------------
# AHRS processing pipeline
# ---------------------------------------------------------------------------


@dataclass
class IMUFusionResult:
    """Container for the per-sample fusion output."""

    time_s: np.ndarray
    quaternion: np.ndarray  # (N, 4)
    euler_deg: np.ndarray  # (N, 3) - roll, pitch, yaw
    acc_earth_ms2: np.ndarray  # (N, 3) - full earth-frame acc (gravity included)
    acc_linear_ms2: np.ndarray  # (N, 3) - earth-frame acc with gravity removed
    acc_vert_ms2: np.ndarray  # (N,)   - vertical linear acceleration (Earth +Z)
    velocity_ms: np.ndarray  # (N,)   - vertical velocity (per-rep drift corrected)
    displacement_m: np.ndarray  # (N,)   - vertical displacement
    g_earth: np.ndarray  # (3,)   - estimated gravity in Earth frame (~[0,0,+g])
    g_mag: float  # |g|


def _detect_static_window(acc: np.ndarray, gyr: np.ndarray, fps: float) -> tuple[int, int]:
    """Return ``(start, end)`` indices of the longest low-motion window at the file start.

    Used both for gravity estimation and gyro bias estimation. We look at the
    first ``~2 s`` (or 30 samples, whichever is larger) and trim the window
    where the gyroscope rate or accelerometer vector exceeds a small threshold.
    """
    n = len(acc)
    win = max(int(round(2.0 * fps)), 30)
    win = min(win, n // 2 if n > 60 else n)

    gyr_norm = np.linalg.norm(gyr[:win], axis=1)
    gyr_q = (
        np.quantile(gyr_norm, 0.5) + 1.0
    )  # deg/s threshold relative to median (rad/s if user supplied)
    acc_norm = np.linalg.norm(acc[:win], axis=1)
    acc_q = max(0.5, np.quantile(acc_norm, 0.5) * 0.05)

    mask = (gyr_norm < gyr_q) & (np.abs(acc_norm - np.median(acc_norm)) < acc_q)
    # find longest contiguous True run within [0, win)
    best_start, best_end = 0, 0
    cur_start = -1
    cur_len = 0
    best_len = 0
    for i, m in enumerate(mask):
        if m:
            if cur_start == -1:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = i + 1
        else:
            cur_start = -1
            cur_len = 0
    if best_len < max(5, win // 4):
        # fallback: just take the first quarter of the inspection window
        return 0, max(win // 4, 5)
    return best_start, best_end


def run_ahrs_fusion(
    df: pd.DataFrame,
    fps: float,
    filter_name: str = "madgwick",
    beta: float = 0.1,
    gyro_units: str = "deg",
) -> IMUFusionResult:
    """Run Madgwick or Mahony AHRS over the full IMU CSV.

    Parameters
    ----------
    df:
        DataFrame containing ``acc_x/y/z`` and ``gyr_x/y/z`` columns. Optional
        ``mag_x/y/z`` columns enable the full AHRS branch.
    fps:
        Sample rate in Hz.
    filter_name:
        ``"madgwick"`` (default) or ``"mahony"``.
    beta:
        Madgwick gain (proportional gain) - 0.033 is the original Madgwick
        recommendation; 0.1 is a more responsive default used in the x-io C
        reference, helpful for hand-recorded barbell data where the static
        accelerometer reading is reliable.
    gyro_units:
        ``"deg"`` (default) for deg/s input or ``"rad"`` for rad/s.
    """
    dt = 1.0 / fps
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    gyr = df[["gyr_x", "gyr_y", "gyr_z"]].to_numpy(dtype=float)
    has_mag = {"mag_x", "mag_y", "mag_z"}.issubset(df.columns)
    mag = df[["mag_x", "mag_y", "mag_z"]].to_numpy(dtype=float) if has_mag else np.zeros_like(acc)

    if gyro_units == "deg":
        gyr = np.deg2rad(gyr)

    # Static window + gyro bias correction
    s, e = _detect_static_window(acc, gyr, fps)
    gyr_bias = gyr[s:e].mean(axis=0) if e > s else np.zeros(3)
    gyr_corr = gyr - gyr_bias

    # Initial orientation from gravity in static window
    acc_static = acc[s:e].mean(axis=0)
    g_mag = float(np.linalg.norm(acc_static))
    if g_mag < 1.0:
        g_mag = 9.81
    q0 = estimate_initial_quaternion(acc_static)

    if filter_name.lower() == "mahony":
        filt = MahonyAHRS(sample_period=dt)
        filt.q = q0
    else:
        filt = MadgwickAHRS(sample_period=dt, beta=beta)
        filt.q = q0

    n = len(df)
    quats = np.zeros((n, 4))
    eulers = np.zeros((n, 3))
    acc_earth = np.zeros((n, 3))

    for i in range(n):
        gx, gy, gz = gyr_corr[i]
        ax, ay, az = acc[i]
        if has_mag and isinstance(filt, MadgwickAHRS):
            mx, my, mz = mag[i]
            filt.update(gx, gy, gz, ax, ay, az, mx, my, mz)
        else:
            filt.update_imu(gx, gy, gz, ax, ay, az)

        quats[i] = filt.q
        roll, pitch, yaw = _quat_to_euler(filt.q)
        eulers[i] = np.rad2deg([roll, pitch, yaw])
        R = _quat_to_rotmat(filt.q)
        acc_earth[i] = R @ np.array([ax, ay, az])

    # Gravity in Earth frame is estimated from the static window of the rotated signal
    g_earth = acc_earth[s:e].mean(axis=0) if e > s else np.array([0.0, 0.0, g_mag])
    acc_linear = acc_earth - g_earth  # subtract DC gravity vector
    acc_vert = acc_linear[:, 2]

    return IMUFusionResult(
        time_s=np.arange(n) * dt,
        quaternion=quats,
        euler_deg=eulers,
        acc_earth_ms2=acc_earth,
        acc_linear_ms2=acc_linear,
        acc_vert_ms2=acc_vert,
        velocity_ms=np.zeros(n),  # filled later, per-rep
        displacement_m=np.zeros(n),
        g_earth=g_earth,
        g_mag=g_mag,
    )


# ---------------------------------------------------------------------------
# Rep detection and per-rep metrics (gravity-removed, AHRS-stabilised)
# ---------------------------------------------------------------------------


def _reject_static_window_reps(
    reps: list[dict],
    duration_outlier_factor: float = 2.5,
) -> list[dict]:
    """Clean spurious repetitions from the raw peak detection.

    Two artifacts are handled:

    1. **Duplicate brackets.** When two concentric peaks fall inside the same
       pair of bracketing valleys they yield identical ``(start_frame,
       end_frame)`` reps. Only the first is kept.
    2. **Static lulls.** When the recording starts or ends with the barbell at
       rest the detector lumps that long, low-motion window into a single fake
       repetition. A rep whose duration is an extreme outlier
       (``> duration_outlier_factor * median``) is dropped.

    Rep numbers are renumbered after filtering.
    """
    # 1) Deduplicate identical brackets, keeping the higher concentric peak.
    by_bracket: dict[tuple[int, int], dict] = {}
    for rep in reps:
        key = (rep["start_frame"], rep["end_frame"])
        by_bracket.setdefault(key, rep)
    deduped = sorted(by_bracket.values(), key=lambda r: r["start_frame"])

    if len(deduped) < 4:
        for i, rep in enumerate(deduped, start=1):
            rep["rep_number"] = i
        return deduped

    # 2) Drop duration outliers (leading/trailing static windows).
    durations = np.array([r["duration_s"] for r in deduped], dtype=float)
    median_dur = float(np.median(durations))
    kept = [r for r in deduped if r["duration_s"] <= duration_outlier_factor * median_dur]

    for i, rep in enumerate(kept, start=1):
        rep["rep_number"] = i
    return kept


def _valid_lowpass_cutoff(cutoff_hz: float, fps: float) -> float:
    """Clamp a low-pass cutoff below Nyquist for Butterworth filtering."""
    nyq = fps / 2.0
    if nyq <= 0:
        raise ValueError(f"Invalid sampling frequency: {fps}")
    return max(0.01, min(float(cutoff_hz), nyq * 0.9))


def detect_reps_from_vertical_acc(
    a_vert: np.ndarray,
    fps: float,
    min_rep_s: float = 1.5,
    prominence_frac: float = 0.10,
    cutoff_hz: float = DEFAULT_IMU_CUTOFF_HZ,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Segment lifts from gravity-free vertical acceleration.

    Returns ``(reps, velocity, displacement, a_filtered)`` matching the
    interface used by :mod:`vaila.vaila_deadlift`. Each rep dict carries
    ``rep_number, start_frame, peak_frame, end_frame, duration_s``.

    The detector uses two passes:
    1. Heavy low-pass (~1.2 Hz) to find the concentric peak (positive vertical
       acceleration) and the eccentric valleys that bracket it.
    2. Light low-pass (~5 Hz or Nyquist/2) for the per-rep integration so we
       preserve the dynamics.

    Spurious reps caused by the leading/trailing static windows (barbell at
    rest before/after the set) are removed via :func:`_reject_static_window_reps`
    before per-rep integration.
    """
    nyq = fps / 2.0
    cutoff_metric = _valid_lowpass_cutoff(cutoff_hz, fps)
    cutoff_detect = min(1.2, nyq * 0.9)
    pad = min(50, len(a_vert) - 1)

    b_hi, a_hi = butter(2, cutoff_metric / nyq)
    a_filt = filtfilt(b_hi, a_hi, a_vert, padlen=pad)

    b_det, a_det = butter(2, cutoff_detect / nyq)
    a_detect = filtfilt(b_det, a_det, a_vert, padlen=pad)

    min_dist = max(int(min_rep_s * fps), 3)
    prom = np.ptp(a_detect) * prominence_frac

    peaks, _ = find_peaks(a_detect, distance=min_dist, prominence=max(prom, 0.1))

    # One rep per concentric peak. The boundary between two adjacent reps is the
    # lowest point (eccentric bottom) of the detection signal between them. This
    # guarantees exactly ``len(peaks)`` reps with no duplicated brackets even
    # when an intermediate valley is too shallow for ``find_peaks`` to flag.
    reps: list[dict] = []
    n_det = len(a_detect)
    for i, pk in enumerate(peaks):
        pk = int(pk)
        if i == 0:
            start = int(np.argmin(a_detect[: pk + 1])) if pk > 0 else 0
        else:
            prev_pk = int(peaks[i - 1])
            start = prev_pk + int(np.argmin(a_detect[prev_pk : pk + 1]))
        if i == len(peaks) - 1:
            end = pk + int(np.argmin(a_detect[pk:])) if pk < n_det - 1 else n_det - 1
        else:
            next_pk = int(peaks[i + 1])
            end = pk + int(np.argmin(a_detect[pk : next_pk + 1]))
        reps.append(
            {
                "rep_number": i + 1,
                "start_frame": start,
                "peak_frame": pk,
                "end_frame": end,
                "duration_s": (end - start) / fps,
            }
        )

    reps = _reject_static_window_reps(reps)

    n = len(a_filt)
    dt = 1.0 / fps
    velocity = np.zeros(n)
    disp = np.zeros(n)
    for rep in reps:
        s, e = rep["start_frame"], min(rep["end_frame"], n - 1)
        a_seg = a_filt[s : e + 1]
        v_seg = np.cumsum(a_seg) * dt
        v_seg -= np.linspace(v_seg[0], v_seg[-1], len(v_seg))
        velocity[s : e + 1] = v_seg
        d_seg = np.cumsum(v_seg) * dt
        d_seg -= np.linspace(d_seg[0], d_seg[-1], len(d_seg))
        disp[s : e + 1] = d_seg

    return reps, velocity, disp, a_filt


def _trim_reps_to_expected(reps: list[dict], expected: int | None) -> list[dict]:
    if expected is None or expected <= 0 or len(reps) <= expected:
        return reps
    excess = len(reps) - expected
    drop_start = excess // 2
    drop_end = excess - drop_start
    trimmed = reps[drop_start : len(reps) - drop_end]
    for i, rep in enumerate(trimmed, start=1):
        rep["rep_number"] = i
    return trimmed


def compute_rep_metrics(
    a_filt: np.ndarray,
    velocity: np.ndarray,
    disp: np.ndarray,
    fps: float,
    barbell_kg: float,
    body_mass_kg: float | None,
    use_total_mass_for_power: bool,
    reps: list[dict],
) -> list[dict]:
    """Per-rep biomechanical metrics from AHRS-stabilised vertical signals."""
    dt = 1.0 / fps
    g = 9.81
    if use_total_mass_for_power and body_mass_kg is not None:
        analysis_mass = float(body_mass_kg + barbell_kg)
    else:
        analysis_mass = float(barbell_kg)

    metrics: list[dict] = []
    for rep in reps:
        s, pk, e = rep["start_frame"], rep["peak_frame"], rep["end_frame"]
        v_conc = velocity[s : pk + 1]
        a_conc = a_filt[s : pk + 1]
        d_rep = disp[s : e + 1]

        force_conc = analysis_mass * (g + a_conc)
        power_conc = force_conc * v_conc

        pos_v = v_conc[v_conc > 0]
        pos_p = power_conc[power_conc > 0]

        peak_vel = float(np.max(np.abs(v_conc))) if len(v_conc) else 0.0
        mean_vel = float(np.mean(pos_v)) if len(pos_v) else 0.0
        peak_power = float(np.max(power_conc)) if len(power_conc) else 0.0
        mean_power = float(np.mean(pos_p)) if len(pos_p) else 0.0
        work = float(np.trapezoid(pos_p, dx=dt)) if len(pos_p) else 0.0
        rom = float(np.ptp(d_rep)) if len(d_rep) else 0.0
        peak_force = float(np.max(force_conc)) if len(force_conc) else 0.0
        mean_force = float(np.mean(force_conc)) if len(force_conc) else 0.0
        impulse = float(np.trapezoid(force_conc, dx=dt)) if len(force_conc) else 0.0

        metrics.append(
            {
                "rep_number": rep["rep_number"],
                "start_frame": s,
                "peak_frame": pk,
                "end_frame": e,
                "duration_s": rep["duration_s"],
                "concentric_s": (pk - s) / fps,
                "eccentric_s": (e - pk) / fps,
                "peak_velocity_ms": peak_vel,
                "mean_velocity_ms": mean_vel,
                "peak_power_W": peak_power,
                "mean_power_W": mean_power,
                "work_J": work,
                "rom_m": rom,
                "peak_force_N": peak_force,
                "mean_force_N": mean_force,
                "impulse_Ns": impulse,
                "analysis_mass_kg": analysis_mass,
                "barbell_mass_kg": float(barbell_kg),
                "body_mass_kg": float(body_mass_kg) if body_mass_kg is not None else None,
            }
        )
    return metrics


def compute_cadence(rep_metrics: list[dict], fps: float) -> dict:
    n = len(rep_metrics)
    if n == 0:
        return {"n_reps": 0, "mean_rep_time_s": 0.0, "std_rep_time_s": 0.0}
    durations = [r["duration_s"] for r in rep_metrics]
    peak_times = [r["peak_frame"] / fps for r in rep_metrics]
    result: dict = {
        "n_reps": n,
        "mean_rep_time_s": float(np.mean(durations)),
        "std_rep_time_s": float(np.std(durations)),
    }
    if n >= 2:
        total = peak_times[-1] - peak_times[0]
        intervals = np.diff(peak_times)
        result["cadence_rpm"] = float((n - 1) / (total / 60.0)) if total > 0 else 0.0
        result["mean_interval_s"] = float(np.mean(intervals))
        result["std_interval_s"] = float(np.std(intervals))
        result["mean_concentric_s"] = float(np.mean([r["concentric_s"] for r in rep_metrics]))
        result["mean_eccentric_s"] = float(np.mean([r["eccentric_s"] for r in rep_metrics]))
    else:
        result["cadence_rpm"] = 0.0
    return result


def compute_rep_comparison(rep_metrics: list[dict]) -> dict[str, dict]:
    if len(rep_metrics) < 2:
        return {}
    df_reps = pd.DataFrame(rep_metrics)
    cols = [
        "peak_velocity_ms",
        "mean_velocity_ms",
        "peak_power_W",
        "mean_power_W",
        "work_J",
        "rom_m",
        "duration_s",
        "concentric_s",
        "eccentric_s",
        "peak_force_N",
        "impulse_Ns",
    ]
    out: dict[str, dict] = {}
    for col in cols:
        if col not in df_reps.columns or df_reps[col].isna().all():
            continue
        vals = df_reps[col].dropna()
        if len(vals) < 2:
            continue
        m = float(vals.mean())
        out[col] = {
            "max": float(vals.max()),
            "min": float(vals.min()),
            "range": float(vals.max() - vals.min()),
            "mean": m,
            "std": float(vals.std()),
            "cv_pct": float(vals.std() / m * 100) if m > 0 else 0.0,
            "best_rep": int(vals.idxmax()) + 1,
            "worst_rep": int(vals.idxmin()) + 1,
        }
    return out


# ---------------------------------------------------------------------------
# Plots and HTML report
# ---------------------------------------------------------------------------


def _quaternion_axis_angle_spherical(quat: np.ndarray) -> dict[str, np.ndarray]:
    """Return a stable axis-angle + spherical representation of unit quaternions.

    ``q`` and ``-q`` encode the same rigid-body orientation. The series is first
    normalized and made hemisphere-continuous, then mapped to the canonical
    shortest rotation ``q = [cos(theta/2), n * sin(theta/2)]`` with ``theta`` in
    degrees and ``n`` on the unit sphere.
    """
    q = np.asarray(quat, dtype=float).copy()
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError("Quaternion array must have shape (N, 4)")

    q_norm = np.linalg.norm(q, axis=1)
    safe_norm = np.where(q_norm > 1e-12, q_norm, 1.0)
    q = q / safe_norm[:, None]

    # Avoid antipodal sign jumps in plots/animation: q and -q are identical rotations.
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0

    q_short = q.copy()
    q_short[q_short[:, 0] < 0.0] *= -1.0
    w = np.clip(q_short[:, 0], -1.0, 1.0)
    v = q_short[:, 1:]
    v_norm = np.linalg.norm(v, axis=1)

    axis = np.zeros_like(v)
    valid = v_norm > 1e-8
    axis[valid] = v[valid] / v_norm[valid, None]

    last_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    for i in range(len(axis)):
        if valid[i] and np.all(np.isfinite(axis[i])):
            last_axis = axis[i]
        else:
            axis[i] = last_axis

    rotation_deg = np.degrees(2.0 * np.arctan2(v_norm, np.maximum(w, 1e-12)))
    rotation_deg = np.clip(rotation_deg, 0.0, 180.0)
    latitude_deg = np.degrees(np.arcsin(np.clip(axis[:, 2], -1.0, 1.0)))
    longitude_wrapped_deg = np.degrees(np.arctan2(axis[:, 1], axis[:, 0]))
    longitude_unwrapped_deg = np.degrees(np.unwrap(np.radians(longitude_wrapped_deg)))

    return {
        "quaternion": q,
        "q_short": q_short,
        "q_norm": q_norm,
        "axis": axis,
        "rotation_deg": rotation_deg,
        "latitude_deg": latitude_deg,
        "longitude_deg": longitude_wrapped_deg,
        "longitude_unwrapped_deg": longitude_unwrapped_deg,
    }


def _select_rigidbody_frames(n_samples: int, reps: list[dict], max_frames: int = 4) -> list[int]:
    """Choose representative frames for the static rigid-body quaternion figure."""
    if n_samples <= 0:
        return []
    candidates: list[int] = [0]
    peak_frames = [int(r.get("peak_frame", 0)) for r in reps]
    peak_frames = [min(max(f, 0), n_samples - 1) for f in peak_frames]
    if peak_frames:
        candidates.extend([peak_frames[0], peak_frames[len(peak_frames) // 2], peak_frames[-1]])
    candidates.append(n_samples - 1)
    candidates.extend(int(round(v)) for v in np.linspace(0, n_samples - 1, max_frames))

    selected: list[int] = []
    for frame in candidates:
        if frame not in selected:
            selected.append(frame)
        if len(selected) >= max_frames:
            break
    return selected


def _cube_vertices(size: float = 0.46) -> np.ndarray:
    s = size / 2.0
    return np.array(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [-s, s, -s],
            [s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [-s, s, s],
            [s, s, s],
        ],
        dtype=float,
    )


def _draw_rigidbody_cube_frame(
    ax,
    q: np.ndarray,
    axis_vec: np.ndarray,
    rotation_deg: float,
    latitude_deg: float,
    longitude_deg: float,
    frame: int,
    time_s: float,
) -> None:
    """Draw one quaternion-oriented cube with fixed Earth axes and body axes."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    R = _quat_to_rotmat(q)
    verts = (R @ _cube_vertices().T).T
    faces = [
        [verts[i] for i in [0, 1, 3, 2]],
        [verts[i] for i in [4, 5, 7, 6]],
        [verts[i] for i in [0, 1, 5, 4]],
        [verts[i] for i in [2, 3, 7, 6]],
        [verts[i] for i in [0, 2, 6, 4]],
        [verts[i] for i in [1, 3, 7, 5]],
    ]
    poly = Poly3DCollection(faces, alpha=0.22, facecolor="#60a5fa", edgecolor="#1e3a8a")
    ax.add_collection3d(poly)

    earth_axes = [
        (np.array([1.0, 0.0, 0.0]), "#dc2626", "X"),
        (np.array([0.0, 1.0, 0.0]), "#16a34a", "Y"),
        (np.array([0.0, 0.0, 1.0]), "#2563eb", "Z"),
    ]
    for vec, color, label in earth_axes:
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, linewidth=1.6, arrow_length_ratio=0.12)
        ax.text(*(vec * 1.08), f"{label}e", color=color, fontsize=8)

    body_labels = ["Xb", "Yb", "Zb"]
    for k, (color, label) in enumerate(zip(["#f87171", "#4ade80", "#60a5fa"], body_labels, strict=True)):
        vec = R[:, k] * 0.72
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color, linewidth=2.6, arrow_length_ratio=0.16)
        ax.text(*(vec * 1.10), label, color=color, fontsize=8)

    axis_vec = np.asarray(axis_vec, dtype=float)
    ax.quiver(
        0,
        0,
        0,
        axis_vec[0],
        axis_vec[1],
        axis_vec[2],
        color="#111827",
        linewidth=2.2,
        arrow_length_ratio=0.12,
    )
    ax.plot(
        [-axis_vec[0], axis_vec[0]],
        [-axis_vec[1], axis_vec[1]],
        [-axis_vec[2], axis_vec[2]],
        color="#111827",
        linestyle=":",
        linewidth=1.0,
        alpha=0.55,
    )
    ax.text(*(axis_vec * 1.12), "n(q)", color="#111827", fontsize=8)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_zlim(-1.15, 1.15)
    ax.set_xlabel("Earth X")
    ax.set_ylabel("Earth Y")
    ax.set_zlabel("Earth Z")
    ax.set_title(
        f"frame {frame} | t={time_s:.2f}s\n"
        f"rot={rotation_deg:.1f} deg, lat={latitude_deg:.1f} deg, lon={longitude_deg:.1f} deg",
        fontsize=9,
    )
    ax.view_init(elev=22, azim=-38)
    with contextlib.suppress(Exception):
        ax.set_box_aspect([1, 1, 1])
    ax.grid(True, alpha=0.25)


def _generate_quaternion_rigidbody_animation_html(
    fusion: IMUFusionResult,
    geom: dict[str, np.ndarray],
    output_dir: str,
    base_name: str,
    ts: str,
    max_frames: int = 360,
) -> str:
    """Write a standalone canvas-based 3D rigid-body quaternion animation."""
    n = len(fusion.time_s)
    if n == 0:
        raise ValueError("Cannot animate an empty quaternion series")
    step = max(1, int(math.ceil(n / max_frames)))
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)

    translation_m = np.column_stack(
        [
            np.zeros_like(fusion.displacement_m),
            np.zeros_like(fusion.displacement_m),
            fusion.displacement_m,
        ]
    )
    payload = {
        "title": base_name,
        "time": np.round(fusion.time_s[idx], 4).tolist(),
        "q": np.round(geom["quaternion"][idx], 6).tolist(),
        "axis": np.round(geom["axis"][idx], 6).tolist(),
        "translation_m": np.round(translation_m[idx], 6).tolist(),
        "rotation_deg": np.round(geom["rotation_deg"][idx], 3).tolist(),
        "latitude_deg": np.round(geom["latitude_deg"][idx], 3).tolist(),
        "longitude_deg": np.round(geom["longitude_deg"][idx], 3).tolist(),
        "source_samples": int(n),
        "display_samples": int(len(idx)),
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    html_path = os.path.join(output_dir, f"{base_name}_imu_quaternion_rigidbody_animation_{ts}.html")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Quaternion rigid-body animation - {base_name}</title>
<style>
body{{font-family:Arial,sans-serif;margin:18px;background:#f8fafc;color:#0f172a}}
.wrap{{max-width:960px;margin:0 auto;background:#fff;border:1px solid #dbe3ef;border-radius:8px;padding:14px}}
canvas{{width:100%;height:auto;background:#ffffff;border:1px solid #d1d5db;border-radius:6px}}
.controls{{display:flex;gap:10px;align-items:center;margin-top:10px;flex-wrap:wrap}}
input[type=range]{{flex:1;min-width:260px}}
button{{padding:7px 12px;border:1px solid #94a3b8;background:#e2e8f0;border-radius:5px;cursor:pointer}}
.note{{font-size:13px;color:#475569;line-height:1.45}}
</style>
</head>
<body>
<div class="wrap">
<h2>Quaternion rigid-body animation: real-world frame + translated/rotated cube</h2>
<canvas id="scene" width="920" height="620"></canvas>
<div class="controls">
<button id="play">Pause</button>
<input id="slider" type="range" min="0" max="0" value="0" step="1">
<span id="readout"></span>
</div>
<p class="note">Real-world Cartesian axes are fixed (Xe, Ye, Ze). The cube center translates by the AHRS/ZUPT vertical displacement and the cube/body axes (Xb, Yb, Zb) rotate by q(t). The black vector is the unit quaternion axis n(q); rotation, latitude, longitude, and translation are shown in physical units.</p>
</div>
<script>
const data = {data_json};
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const playBtn = document.getElementById('play');
const readout = document.getElementById('readout');
slider.max = String(data.time.length - 1);
let frame = 0;
let playing = true;
const cubeVerts = [[-.24,-.24,-.24],[.24,-.24,-.24],[-.24,.24,-.24],[.24,.24,-.24],[-.24,-.24,.24],[.24,-.24,.24],[-.24,.24,.24],[.24,.24,.24]];
const faces = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]];
function rotmat(q){{
  const [w,x,y,z] = q;
  return [
    [w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)],
    [2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)],
    [2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z]
  ];
}}
function mv(R, v){{return [R[0][0]*v[0]+R[0][1]*v[1]+R[0][2]*v[2], R[1][0]*v[0]+R[1][1]*v[1]+R[1][2]*v[2], R[2][0]*v[0]+R[2][1]*v[1]+R[2][2]*v[2]];}}
function add(a,b){{return [a[0]+b[0],a[1]+b[1],a[2]+b[2]];}}
function view(p){{
  const yaw=-38*Math.PI/180, pitch=23*Math.PI/180;
  let x=Math.cos(yaw)*p[0]-Math.sin(yaw)*p[1];
  let y=Math.sin(yaw)*p[0]+Math.cos(yaw)*p[1];
  let z=p[2];
  let y2=Math.cos(pitch)*y-Math.sin(pitch)*z;
  let z2=Math.sin(pitch)*y+Math.cos(pitch)*z;
  return [x,y2,z2];
}}
function proj(p){{
  const v=view(p); const scale=220/(3.1-v[2]);
  return [canvas.width/2 + v[0]*scale, canvas.height/2 + 35 - v[1]*scale, v[2]];
}}
function line3(a,b,color,w,label,alpha=1){{
  const A=proj(a), B=proj(b); ctx.save(); ctx.globalAlpha=alpha; ctx.strokeStyle=color; ctx.lineWidth=w; ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
  if(label){{ctx.fillStyle=color; ctx.font='14px Arial'; ctx.fillText(label,B[0]+5,B[1]-5);}}
  ctx.restore();
}}
function drawGrid(){{
  for(let g=-1; g<=1.001; g+=0.5){{
    line3([-1,g,0],[1,g,0],'#cbd5e1',0.8,null,0.65);
    line3([g,-1,0],[g,1,0],'#cbd5e1',0.8,null,0.65);
  }}
}}
function drawTrace(i){{
  for(let j=1;j<=i;j++){{line3(data.translation_m[j-1],data.translation_m[j],'#f59e0b',2,null,0.85);}}
}}
function draw(i){{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const q=data.q[i], R=rotmat(q), axis=data.axis[i], T=data.translation_m[i];
  drawGrid();
  line3([0,0,0],[1.12,0,0],'#dc2626',2,'Xe'); line3([0,0,0],[0,1.12,0],'#16a34a',2,'Ye'); line3([0,0,0],[0,0,1.12],'#2563eb',2,'Ze');
  drawTrace(i);
  line3([T[0],T[1],0],T,'#f59e0b',1.5,'translation',0.7);
  const verts=cubeVerts.map(v=>add(T,mv(R,v)));
  const faceData=faces.map(f=>({{idx:f, z:f.reduce((s,j)=>s+proj(verts[j])[2],0)/f.length}})).sort((a,b)=>a.z-b.z);
  for(const fd of faceData){{
    const pts=fd.idx.map(j=>proj(verts[j]));
    ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]); for(let k=1;k<pts.length;k++)ctx.lineTo(pts[k][0],pts[k][1]); ctx.closePath();
    ctx.fillStyle='rgba(96,165,250,0.22)'; ctx.fill(); ctx.strokeStyle='#1e3a8a'; ctx.lineWidth=1.2; ctx.stroke();
  }}
  line3(T,add(T,mv(R,[.78,0,0])),'#f87171',4,'Xb'); line3(T,add(T,mv(R,[0,.78,0])),'#4ade80',4,'Yb'); line3(T,add(T,mv(R,[0,0,.78])),'#60a5fa',4,'Zb');
  line3(T,add(T,[axis[0]*1.18,axis[1]*1.18,axis[2]*1.18]),'#111827',3,'n(q)');
  ctx.fillStyle='#0f172a'; ctx.font='18px Arial'; ctx.fillText(data.title,18,30);
  ctx.font='14px Arial';
  const txt=`frame ${{i+1}}/${{data.time.length}} | t=${{data.time[i].toFixed(2)}} s | rot=${{data.rotation_deg[i].toFixed(1)}} deg | lat=${{data.latitude_deg[i].toFixed(1)}} deg | lon=${{data.longitude_deg[i].toFixed(1)}} deg | Z=${{T[2].toFixed(3)}} m`;
  ctx.fillText(txt,18,54);
  ctx.fillStyle='#64748b'; ctx.fillText(`world translation [X,Y,Z] = [${{T[0].toFixed(3)}}, ${{T[1].toFixed(3)}}, ${{T[2].toFixed(3)}}] m | downsampled ${{data.display_samples}} / ${{data.source_samples}} samples`,18,76);
  readout.textContent = txt;
  slider.value = String(i);
}}
function tick(){{ if(playing){{frame=(frame+1)%data.time.length; draw(frame);}} requestAnimationFrame(tick); }}
slider.addEventListener('input', e=>{{frame=Number(e.target.value); draw(frame);}});
playBtn.addEventListener('click', ()=>{{playing=!playing; playBtn.textContent=playing?'Pause':'Play';}});
draw(0); tick();
</script>
</body>
</html>"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path


def _generate_plots(
    fusion: IMUFusionResult,
    a_filt: np.ndarray,
    velocity: np.ndarray,
    disp: np.ndarray,
    reps: list[dict],
    rep_metrics: list[dict],
    output_dir: str,
    base_name: str,
    fps: float,
    filter_name: str,
    raw_acc: np.ndarray,
) -> list[str]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out: list[str] = []
    time_axis = fusion.time_s

    # 0. Raw accelerometer axes (acc_x, acc_y, acc_z)
    # The axis with the largest peak-to-peak swing is, in practice, the one
    # aligned with the lift direction (vertical) - it also shows the clearest
    # peaks used for the repetition count.
    ptp_axes = np.ptp(raw_acc, axis=0)
    vertical_idx = int(np.argmax(ptp_axes))
    axis_labels = ["acc_x", "acc_y", "acc_z"]
    axis_colors = ["steelblue", "darkorange", "seagreen"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for k in range(3):
        suffix = " (likely vertical)" if k == vertical_idx else ""
        ax.plot(
            time_axis,
            raw_acc[:, k],
            color=axis_colors[k],
            linewidth=1.0 if k == vertical_idx else 0.7,
            alpha=0.95 if k == vertical_idx else 0.65,
            label=f"{axis_labels[k]}{suffix}  (ptp={ptp_axes[k]:.1f} m/s²)",
        )
    for rep in reps:
        ax.axvline(rep["peak_frame"] / fps, color="red", linestyle="--", alpha=0.45)
        ax.annotate(
            f"R{rep['rep_number']}",
            (rep["peak_frame"] / fps, raw_acc[rep["peak_frame"], vertical_idx]),
            fontsize=7,
            ha="center",
            va="bottom",
            color="red",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Raw acceleration (m/s²)")
    ax.set_title(
        f"Raw Accelerometer Axes (gravity included) - {base_name}\n"
        f"Dominant axis: {axis_labels[vertical_idx]} - {len(reps)} repetitions detected"
    )
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p0 = os.path.join(output_dir, f"{base_name}_imu_raw_acc_axes_{ts}.png")
    fig.savefig(p0, dpi=150)
    plt.close(fig)
    out.append(p0)

    # 1. Orientation (Euler) and Earth-frame acc
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(time_axis, fusion.euler_deg[:, 0], label="Roll", color="steelblue")
    axes[0].plot(time_axis, fusion.euler_deg[:, 1], label="Pitch", color="darkorange")
    axes[0].plot(time_axis, fusion.euler_deg[:, 2], label="Yaw", color="seagreen")
    axes[0].set_ylabel("Euler (deg)")
    axes[0].set_title(
        f"AHRS Orientation ({filter_name.title()}) - {base_name}",
    )
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, fusion.acc_earth_ms2[:, 0], label="ax_earth", color="steelblue")
    axes[1].plot(time_axis, fusion.acc_earth_ms2[:, 1], label="ay_earth", color="darkorange")
    axes[1].plot(
        time_axis, fusion.acc_earth_ms2[:, 2], label="az_earth (gravity in)", color="seagreen"
    )
    axes[1].axhline(fusion.g_mag, color="red", linestyle="--", alpha=0.4, label="|g|")
    axes[1].set_ylabel("Earth acc (m/s²)")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, a_filt, color="purple", linewidth=0.8, label="Vertical linear acc")
    axes[2].set_ylabel("a_v linear (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    for rep in reps:
        axes[2].axvline(rep["peak_frame"] / fps, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    p1 = os.path.join(output_dir, f"{base_name}_imu_ahrs_orientation_{ts}.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    out.append(p1)

    # 1b. Quaternion spherical coordinates: rotation + latitude + longitude.
    quat = fusion.quaternion
    quat_geom = _quaternion_axis_angle_spherical(quat)
    q_norm = quat_geom["q_norm"]
    rotation_deg = quat_geom["rotation_deg"]
    latitude_deg = quat_geom["latitude_deg"]
    longitude_unwrapped_deg = quat_geom["longitude_unwrapped_deg"]
    norm_dev = float(np.max(np.abs(q_norm - 1.0)))

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(time_axis, q_norm, color="#1d4ed8", linewidth=1.0)
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.45)
    axes[0].set_ylabel("|q|")
    axes[0].set_title(f"Unit quaternion norm check (max deviation {norm_dev:.2e})")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, rotation_deg, color="#b45309", linewidth=1.0)
    axes[1].set_ylabel("Rotation (deg)")
    axes[1].set_title("Axis-angle rotation: theta = 2 atan2(||q_xyz||, q_w)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, latitude_deg, color="#047857", linewidth=1.0)
    axes[2].set_ylabel("Latitude (deg)")
    axes[2].set_ylim(-95, 95)
    axes[2].set_title("Quaternion unit axis latitude: asin(n_z)")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_axis, longitude_unwrapped_deg, color="#7e22ce", linewidth=1.0)
    axes[3].set_ylabel("Longitude (deg)")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title("Quaternion unit axis longitude: atan2(n_y, n_x), unwrapped for line continuity")
    axes[3].grid(True, alpha=0.3)

    for rep in reps:
        for ax in axes:
            ax.axvline(rep["peak_frame"] / fps, color="red", linestyle=":", alpha=0.35)

    fig.suptitle(
        f"Quaternion spherical coordinates - {base_name} ({filter_name.title()} AHRS)",
        fontsize=13,
        y=1.005,
    )
    fig.tight_layout()
    p1b = os.path.join(output_dir, f"{base_name}_imu_quaternion_spherical_{ts}.png")
    fig.savefig(p1b, dpi=150)
    plt.close(fig)
    out.append(p1b)

    # 1c. Standalone interactive 3D animation (HTML canvas, no external deps).
    p1d = _generate_quaternion_rigidbody_animation_html(
        fusion,
        quat_geom,
        output_dir,
        base_name,
        ts,
    )
    out.append(p1d)

    # 2. Vertical velocity / displacement
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(time_axis, velocity, color="darkorange", linewidth=0.9)
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title(f"Vertical velocity / displacement - {base_name}")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(time_axis, disp, color="seagreen", linewidth=0.9)
    axes[1].set_ylabel("Displacement (m)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    for rep in reps:
        for ax in axes:
            ax.axvline(rep["peak_frame"] / fps, color="red", linestyle="--", alpha=0.4)
        axes[1].annotate(
            f"R{rep['rep_number']}",
            (rep["peak_frame"] / fps, disp[rep["peak_frame"]]),
            fontsize=8,
            ha="center",
            va="bottom",
            color="red",
        )
    fig.tight_layout()
    p2 = os.path.join(output_dir, f"{base_name}_imu_vel_disp_{ts}.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    out.append(p2)

    # 3. Per-rep bar charts
    if len(rep_metrics) >= 2:
        rep_nums = [r["rep_number"] for r in rep_metrics]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].bar(rep_nums, [r["peak_velocity_ms"] for r in rep_metrics], color="steelblue")
        axes[0, 0].set_ylabel("Peak Velocity (m/s)")
        axes[0, 0].set_title("Peak Velocity per Rep")
        axes[0, 1].bar(rep_nums, [r["peak_power_W"] for r in rep_metrics], color="darkorange")
        axes[0, 1].set_ylabel("Peak Power (W)")
        axes[0, 1].set_title("Peak Power per Rep")
        axes[1, 0].bar(rep_nums, [r["work_J"] for r in rep_metrics], color="seagreen")
        axes[1, 0].set_ylabel("Work (J)")
        axes[1, 0].set_title("Work per Rep")
        axes[1, 0].set_xlabel("Rep #")
        axes[1, 1].bar(rep_nums, [r["duration_s"] for r in rep_metrics], color="mediumpurple")
        axes[1, 1].set_ylabel("Duration (s)")
        axes[1, 1].set_title("Duration per Rep")
        axes[1, 1].set_xlabel("Rep #")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle(f"Rep-by-Rep Metrics - {base_name}", fontsize=13, y=1.01)
        fig.tight_layout()
        p3 = os.path.join(output_dir, f"{base_name}_imu_rep_bars_{ts}.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        out.append(p3)

    # 4. Velocity overlay per rep
    if len(rep_metrics) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap("viridis")
        for i, rep in enumerate(reps):
            s, e = rep["start_frame"], rep["end_frame"]
            v_rep = velocity[s : e + 1]
            t_rep = np.arange(len(v_rep)) / fps
            color = cmap(i / max(len(reps) - 1, 1))
            ax.plot(t_rep, v_rep, color=color, label=f"Rep {rep['rep_number']}", alpha=0.85)
        ax.set_xlabel("Time within rep (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title(f"Velocity Profile Overlay - {base_name}")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p4 = os.path.join(output_dir, f"{base_name}_imu_velocity_overlay_{ts}.png")
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        out.append(p4)

    return out


def _generate_html_report(
    rep_metrics: list[dict],
    cadence: dict,
    comparison: dict[str, dict],
    plot_files: list[str],
    output_dir: str,
    base_name: str,
    barbell_kg: float,
    body_mass_kg: float | None,
    fps: float,
    filter_name: str,
    g_mag: float,
    body_height_m: float | None = None,
) -> str:
    report_path = os.path.join(output_dir, f"{base_name}_imu_ahrs_report.html")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rep_rows = ""
    for r in rep_metrics:
        rep_rows += f"""
        <tr>
            <td>{r["rep_number"]}</td>
            <td>{r["duration_s"]:.2f}</td>
            <td>{r["concentric_s"]:.2f}</td>
            <td>{r["eccentric_s"]:.2f}</td>
            <td>{r["peak_velocity_ms"]:.3f}</td>
            <td>{r["mean_velocity_ms"]:.3f}</td>
            <td>{r["peak_power_W"]:.1f}</td>
            <td>{r["mean_power_W"]:.1f}</td>
            <td>{r["work_J"]:.2f}</td>
            <td>{r["rom_m"]:.3f}</td>
            <td>{r["peak_force_N"]:.1f}</td>
            <td>{r["impulse_Ns"]:.2f}</td>
        </tr>"""

    label_map = {
        "peak_velocity_ms": "Peak Velocity (m/s)",
        "mean_velocity_ms": "Mean Velocity (m/s)",
        "peak_power_W": "Peak Power (W)",
        "mean_power_W": "Mean Power (W)",
        "work_J": "Work (J)",
        "rom_m": "ROM (m)",
        "duration_s": "Duration (s)",
        "concentric_s": "Concentric (s)",
        "eccentric_s": "Eccentric (s)",
        "peak_force_N": "Peak Force (N)",
        "impulse_Ns": "Impulse (N·s)",
    }
    comp_rows = ""
    for key, stats in comparison.items():
        name = label_map.get(key, key)
        comp_rows += f"""
        <tr>
            <td>{name}</td>
            <td>{stats["max"]:.3f}</td>
            <td>{stats["min"]:.3f}</td>
            <td>{stats["range"]:.3f}</td>
            <td>{stats["mean"]:.3f}</td>
            <td>{stats["std"]:.3f}</td>
            <td>{stats["cv_pct"]:.1f}%</td>
            <td>#{stats["best_rep"]}</td>
            <td>#{stats["worst_rep"]}</td>
        </tr>"""

    media_blocks: list[str] = []
    for p in plot_files:
        name = os.path.basename(p)
        if name.lower().endswith(".html"):
            media_blocks.append(
                f'<div class="img-container"><iframe class="anim-frame" src="{name}" '
                f'title="{name}"></iframe><p><a href="{name}">Open 3D animation in a new tab</a></p></div>\n'
            )
        else:
            media_blocks.append(f'<div class="img-container"><img src="{name}" alt="{name}"></div>\n')
    images_html = "".join(media_blocks)

    body_mass_str = (
        f"{body_mass_kg:.1f} kg" if body_mass_kg is not None else "n/a (barbell-only power)"
    )
    body_height_str = f"{body_height_m:.2f} m" if body_height_m is not None else "n/a"
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>vailá - IMU Deadlift AHRS Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background: #fafafa; color: #333; }}
        h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: center; }}
        th {{ background: #ebf8ff; color: #2b6cb0; }}
        .summary-box {{ background: #e2e8f0; padding: 15px; border-left: 6px solid #4a5568;
                        font-size: 1.1em; margin: 20px 0; }}
        .img-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 90%; height: auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .anim-frame {{ width: 96%; height: 760px; border: 1px solid #cbd5e1; border-radius: 8px; background: white; }}
        .refs {{ background: #f7fafc; border-left: 6px solid #2b6cb0; padding: 15px 20px;
                 margin: 25px 0; font-size: 0.95em; }}
        .refs h2 {{ margin-top: 0; }}
        .refs a {{ color: #2b6cb0; word-break: break-all; }}
        .tribute {{ background: #fffaf0; border-left: 6px solid #b7791f; padding: 15px 20px;
                    margin: 25px 0; font-size: 0.95em; }}
        .tribute h2 {{ margin-top: 0; color: #975a16; }}
        .tribute a {{ color: #975a16; word-break: break-all; }}
        footer {{ margin-top: 35px; padding-top: 15px; border-top: 1px solid #e2e8f0;
                  color: #718096; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>IMU Deadlift Biomechanical Report (AHRS)</h1>
    <p><strong>File:</strong> {base_name}</p>
    <p><strong>Date:</strong> {now}</p>
    <p><strong>Filter:</strong> {filter_name.title()} AHRS
        &nbsp;|&nbsp; <strong>Sample rate:</strong> {fps:.1f} Hz
        &nbsp;|&nbsp; <strong>|g| (sensor):</strong> {g_mag:.3f} m/s²</p>
    <p><strong>Barbell weight:</strong> {barbell_kg:.1f} kg
        &nbsp;|&nbsp; <strong>Body mass:</strong> {body_mass_str}
        &nbsp;|&nbsp; <strong>Body height:</strong> {body_height_str}</p>

    <div class="summary-box">
        <strong>Repetitions detected:</strong> {cadence["n_reps"]}
        &nbsp;|&nbsp; <strong>Cadence:</strong> {cadence.get("cadence_rpm", 0):.1f} reps/min
        &nbsp;|&nbsp; <strong>Mean rep time:</strong>
        {cadence.get("mean_rep_time_s", 0):.2f} ± {cadence.get("std_rep_time_s", 0):.2f} s
    </div>

    <h2>Per-Repetition Metrics</h2>
    <table>
        <tr>
            <th>Rep</th><th>Duration (s)</th><th>Conc. (s)</th><th>Ecc. (s)</th>
            <th>Peak Vel (m/s)</th><th>Mean Vel (m/s)</th>
            <th>Peak Power (W)</th><th>Mean Power (W)</th>
            <th>Work (J)</th><th>ROM (m)</th><th>Peak Force (N)</th><th>Impulse (N·s)</th>
        </tr>
        {rep_rows}
    </table>

    <h2>Rep-to-Rep Comparison</h2>
    <table>
        <tr>
            <th>Metric</th><th>Max</th><th>Min</th><th>Range</th>
            <th>Mean</th><th>Std</th><th>CV%</th>
            <th>Best Rep</th><th>Worst Rep</th>
        </tr>
        {comp_rows}
    </table>

    <h2>Time-series and rep visualizations</h2>
    {images_html}

    <div class="quat-didactic" style="background:#f0f9ff; border-left:6px solid #0e7490;
         padding:15px 20px; margin:25px 0; font-size:0.95em;">
        <h2 style="margin-top:0; color:#155e75;">Quaternion axis-angle spherical view</h2>
        <p>The report now visualizes the unit quaternion as a rigid-body rotation:</p>
        <p><code>q = [w, x, y, z] = [cos(theta/2), n<sub>x</sub> sin(theta/2),
        n<sub>y</sub> sin(theta/2), n<sub>z</sub> sin(theta/2)]</code>, where
        <code>n</code> is the unit rotation axis. Because <code>|q| = 1</code>, the
        axis can be represented on the unit sphere with:</p>
        <ul>
            <li><strong>rotation_deg:</strong> <code>theta</code>, the rigid-body rotation angle in degrees;</li>
            <li><strong>axis_latitude_deg:</strong> <code>asin(n<sub>z</sub>)</code> in degrees;</li>
            <li><strong>axis_longitude_deg:</strong> <code>atan2(n<sub>y</sub>, n<sub>x</sub>)</code> in degrees.</li>
        </ul>
        <p>The spherical line graph shows these three values through time. The embedded
        3D animation is the main rigid-body view: the real-world Cartesian axes stay fixed
        (Xe, Ye, Ze), the cube center translates by the AHRS/ZUPT vertical displacement,
        the sensor cube and body axes (Xb, Yb, Zb) rotate by <code>q(t)</code>, and the black
        vector draws the unit quaternion axis <code>n(q)</code>.</p>
        <p>Unlike Euler angles, quaternions are singularity-free: no gimbal lock, no axis-order
        ambiguity, and no discontinuous roll/pitch/yaw interpretation. The code still makes
        <code>q</code> sign-continuous for display because <code>q</code> and <code>-q</code>
        are the same physical orientation.</p>
    </div>

    <div class="refs">
        <h2>References &amp; Credits</h2>
        <p>This report was produced by the <strong>vailá</strong> IMU Deadlift
        AHRS pipeline (<code>vaila_deadlift_imu.py</code>). The orientation
        tracking is a direct Python port of the open-source x-io Technologies
        AHRS C reference. Method credits:</p>
        <ul>
            <li>Madgwick, S. O. H. (2010).
                <em>An efficient orientation filter for inertial and
                inertial/magnetic sensor arrays.</em> Technical report,
                University of Bristol.</li>
            <li>Mahony, R., Hamel, T., &amp; Pflimlin, J.-M. (2008).
                <em>Nonlinear complementary filters on the special orthogonal
                group.</em> IEEE Transactions on Automatic Control, 53(5),
                1203&ndash;1218.</li>
            <li>x-io Technologies &mdash; Open-source IMU and AHRS algorithms:
                <a href="https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/">
                x-io.co.uk/open-source-imu-and-ahrs-algorithms</a></li>
            <li>xioTechnologies/Fusion (modern C/C++ reference):
                <a href="https://github.com/xioTechnologies/Fusion/tree/main">
                github.com/xioTechnologies/Fusion</a></li>
            <li>Madgwick filter walk-through (cross-check):
                <a href="https://medium.com/@k66115704/imu-madgwick-filter-explanation-556fbe7f02e3">
                medium.com/@k66115704/imu-madgwick-filter-explanation</a></li>
        </ul>
        <p>The vendored C reference used to validate the port lives under
        <code>tests/Deadlift/madgwick_algorithm_c/</code>
        (<code>MadgwickAHRS/</code> and <code>MahonyAHRS/</code>).</p>
    </div>

    <div class="tribute">
        <h2>Publication History &amp; Tribute to Prof. René Jean Brenzikofer</h2>
        <p>Modern biomechanics is grounded in solid physical&ndash;mathematical
        foundations for modeling human movement. I leave here my profound tribute
        and final farewell to one of the greatest exponents of this scientific
        rigor in Brazil, Professor <strong>René Jean Brenzikofer</strong>
        (UNICAMP). It is an honor to have known him and to have shared creative
        ideas that demonstrated a biomechanics that truly makes a difference. I
        express my most sincere posthumous gratitude to his memory, celebrating
        his intellectual generosity and the privilege of his academic
        companionship. This close collaboration enabled the <strong>first
        publication of three-dimensional modeling data based on
        Quaternions</strong> in the book <em>"Modelos Matemáticos nas Ciências
        Não-Exatas"</em> (Editora Blucher, 2007).</p>

        <p>Today, Quaternions are an indispensable technological reality. This
        four-dimensional algebra solves the mathematical singularity known as
        <em>gimbal lock</em>, and is the exact basis for the data-fusion
        algorithms in Inertial Measurement Units (IMUs) embedded in wearable
        devices (such as smartwatches and GPS units with IMUs attached to soccer
        players) for real-time motion analysis. They are also the absolute
        computational standard for kinematic computation in 3D animation and game
        physics engines &mdash; the very mathematics that powers the AHRS fusion
        used in this report.</p>

        <p>The impact and importance of Prof. René on my scientific trajectory
        are historically recorded in the very structure of this research. In the
        acknowledgments section of my doctoral thesis his name was expressly
        written in recognition of the support and inspiration that were
        fundamental to consolidating this method in the country. May Professor
        René's legacy of dedication to science inspire the academic trajectory of
        all of you.</p>

        <p><strong>Primary references (BibTeX-ready):</strong></p>
        <ul>
            <li>Nogueira, E. A., Martins, L. E. B., &amp; Brenzikofer, R.
                (2007). <em>Modelos Matemáticos nas Ciências Não-Exatas - vol. 1.</em>
                São Paulo: Editora Blucher. ISBN 978-85-212-0419-0.
                <a href="https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190">
                blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1</a></li>
            <li>Santiago, P. R. P. (2009). <em>Rotações tridimensionais em
                biomecânica via quatérnions: aplicações na análise dos
                movimentos esportivos.</em> Tese (Doutorado) &mdash;
                Universidade Estadual Paulista (Unesp), Instituto de
                Biociências de Rio Claro.
                <a href="http://hdl.handle.net/11449/100404">hdl.handle.net/11449/100404</a>
                &middot;
                <a href="https://repositorio.unesp.br/server/api/core/bitstreams/41603fa7-545b-4e74-a045-57ce94885e0c/content">PDF</a></li>
        </ul>
        <p><strong>Tribute &amp; social-media coverage:</strong></p>
        <ul>
            <li>Instagram post (tribute &amp; publication history):
                <a href="https://www.instagram.com/p/DZLogIJoFbF/">instagram.com/p/DZLogIJoFbF</a></li>
            <li>LinkedIn post (publication history &amp; tribute to
                Prof. Brenzikofer):
                <a href="https://www.linkedin.com/posts/paulo-roberto-pereira-santiago-132619112_hist%C3%B3rico-de-publica%C3%A7%C3%A3o-e-homenagem-ao-prof-ugcPost-7468670091845472257-3tzD/">
                linkedin.com/posts/paulo-roberto-pereira-santiago - homenagem ao Prof. Brenzikofer</a></li>
        </ul>
        <pre style="background:#fff5e6; border:1px solid #f6ad55; padding:10px;
                    border-radius:6px; font-size:0.85em; overflow-x:auto;">
@book{{nogueira2007modelos,
  title     = {{Modelos matem\\'aticos nas ci\\^encias n\\~ao-exatas - vol. 1}},
  author    = {{Nogueira, Eduardo Arantes and Martins, Luiz Eduardo Barreto
               and Brenzikofer, Ren\\'e}},
  year      = {{2007}},
  publisher = {{Editora Blucher}},
  isbn      = {{9788521204190}},
  url       = {{https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190}}
}}

@phdthesis{{santiago2009rotaccoes,
  title    = {{Rota\\c{{c}}\\~oes tridimensionais em biomec\\^anica via quat\\'ernions:
              aplica\\c{{c}}\\~oes na an\\'alise dos movimentos esportivos}},
  author   = {{Santiago, Paulo Roberto Pereira}},
  school   = {{Universidade Estadual Paulista (Unesp),
              Instituto de Bioci\\^encias de Rio Claro}},
  year     = {{2009}},
  url      = {{http://hdl.handle.net/11449/100404}}
}}</pre>
        <p style="text-align:right; font-style:italic; color:#975a16;">
        &mdash; Prof. Paulo R. P. Santiago</p>
    </div>

    <footer>
        <p>Generated by vailá &mdash; Versatile Anarcho Integrated Liberation
        Ánalysis &middot; <code>vaila_deadlift_imu.py</code> v{VAILA_VERSION}
        &middot; Author: Prof. Paulo R. P. Santiago &middot;
        <a href="https://github.com/vaila-multimodaltoolbox/vaila">
        github.com/vaila-multimodaltoolbox/vaila</a> &middot; AGPL-3.0</p>
    </footer>

</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def _generate_md_report(
    rep_metrics: list[dict],
    cadence: dict,
    comparison: dict[str, dict],
    plot_files: list[str],
    output_dir: str,
    base_name: str,
    barbell_kg: float,
    body_mass_kg: float | None,
    fps: float,
    filter_name: str,
    g_mag: float,
    body_height_m: float | None = None,
) -> str:
    """Generate a Markdown sibling of the HTML report (same data, plain text)."""
    report_path = os.path.join(output_dir, f"{base_name}_imu_ahrs_report.md")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body_mass_str = (
        f"{body_mass_kg:.1f} kg" if body_mass_kg is not None else "n/a (barbell-only power)"
    )
    body_height_str = f"{body_height_m:.2f} m" if body_height_m is not None else "n/a"

    label_map = {
        "peak_velocity_ms": "Peak Velocity (m/s)",
        "mean_velocity_ms": "Mean Velocity (m/s)",
        "peak_power_W": "Peak Power (W)",
        "mean_power_W": "Mean Power (W)",
        "work_J": "Work (J)",
        "rom_m": "ROM (m)",
        "duration_s": "Duration (s)",
        "concentric_s": "Concentric (s)",
        "eccentric_s": "Eccentric (s)",
        "peak_force_N": "Peak Force (N)",
        "impulse_Ns": "Impulse (N·s)",
    }

    lines: list[str] = []
    lines.append(f"# IMU Deadlift Biomechanical Report (AHRS) - {base_name}")
    lines.append("")
    lines.append(f"- **File:** `{base_name}`")
    lines.append(f"- **Date:** {now}")
    lines.append(
        f"- **Filter:** {filter_name.title()} AHRS  |  **Sample rate:** {fps:.1f} Hz  "
        f"|  **|g| (sensor):** {g_mag:.3f} m/s²"
    )
    lines.append(
        f"- **Barbell weight:** {barbell_kg:.1f} kg  |  **Body mass:** {body_mass_str}  "
        f"|  **Body height:** {body_height_str}"
    )
    lines.append("")
    lines.append(
        f"> **Repetitions detected:** {cadence['n_reps']}  |  "
        f"**Cadence:** {cadence.get('cadence_rpm', 0):.1f} reps/min  |  "
        f"**Mean rep time:** {cadence.get('mean_rep_time_s', 0):.2f} "
        f"± {cadence.get('std_rep_time_s', 0):.2f} s"
    )
    lines.append("")

    lines.append("## Per-Repetition Metrics")
    lines.append("")
    lines.append(
        "| Rep | Duration (s) | Conc. (s) | Ecc. (s) | Peak Vel (m/s) | Mean Vel (m/s) "
        "| Peak Power (W) | Mean Power (W) | Work (J) | ROM (m) | Peak Force (N) | "
        "Impulse (N·s) |"
    )
    lines.append(
        "|-----|--------------|-----------|----------|---------------|---------------"
        "|----------------|----------------|----------|---------|----------------|"
        "----------------|"
    )
    for r in rep_metrics:
        lines.append(
            f"| {r['rep_number']} | {r['duration_s']:.2f} | {r['concentric_s']:.2f} | "
            f"{r['eccentric_s']:.2f} | {r['peak_velocity_ms']:.3f} | "
            f"{r['mean_velocity_ms']:.3f} | {r['peak_power_W']:.1f} | "
            f"{r['mean_power_W']:.1f} | {r['work_J']:.2f} | {r['rom_m']:.3f} | "
            f"{r['peak_force_N']:.1f} | {r['impulse_Ns']:.2f} |"
        )
    lines.append("")

    if comparison:
        lines.append("## Rep-to-Rep Comparison")
        lines.append("")
        lines.append("| Metric | Max | Min | Range | Mean | Std | CV% | Best Rep | Worst Rep |")
        lines.append("|--------|-----|-----|-------|------|-----|-----|----------|-----------|")
        for key, stats in comparison.items():
            name = label_map.get(key, key)
            lines.append(
                f"| {name} | {stats['max']:.3f} | {stats['min']:.3f} | "
                f"{stats['range']:.3f} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                f"{stats['cv_pct']:.1f}% | #{stats['best_rep']} | #{stats['worst_rep']} |"
            )
        lines.append("")

    lines.append("## Time-series and rep visualizations")
    lines.append("")
    for p in plot_files:
        rel = os.path.basename(p)
        if rel.lower().endswith(".html"):
            lines.append(f"[Open 3D quaternion rigid-body animation]({rel})")
        else:
            lines.append(f"![{rel}]({rel})")
        lines.append("")

    lines.append("## Quaternion axis-angle spherical view")
    lines.append("")
    lines.append(
        "The report visualizes the unit quaternion as a rigid-body rotation: "
        "`q = [w, x, y, z] = [cos(theta/2), nx sin(theta/2), ny sin(theta/2), "
        "nz sin(theta/2)]`, where `n` is the unit rotation axis. Because `|q| = 1`, "
        "the axis can be represented on the unit sphere."
    )
    lines.append("")
    lines.append("- **rotation_deg**: `theta`, the rigid-body rotation angle in degrees.")
    lines.append("- **axis_latitude_deg**: `asin(nz)` in degrees.")
    lines.append("- **axis_longitude_deg**: `atan2(ny, nx)` in degrees.")
    lines.append("")
    lines.append(
        "The spherical line graph shows these values through time. The embedded HTML "
        "animation is the main rigid-body view: real-world Earth axes stay fixed "
        "(Xe, Ye, Ze), the cube center translates by AHRS/ZUPT vertical displacement, "
        "the sensor cube/body axes (Xb, Yb, Zb) rotate by `q(t)`, and the black "
        "vector draws the unit quaternion axis `n(q)`."
    )
    lines.append("")
    lines.append(
        "Unlike Euler angles, quaternions are singularity-free: no gimbal lock, no "
        "axis-order ambiguity, and no discontinuous roll/pitch/yaw interpretation. "
        "The code makes `q` sign-continuous for display because `q` and `-q` encode "
        "the same physical orientation."
    )
    lines.append("")

    lines.append("## References & Credits")
    lines.append("")
    lines.append(
        "This report was produced by the **vailá** IMU Deadlift AHRS pipeline "
        "(`vaila_deadlift_imu.py`). The orientation tracking is a direct Python "
        "port of the open-source x-io Technologies AHRS C reference."
    )
    lines.append("")
    lines.append("**Method credits:**")
    lines.append("")
    lines.append(
        "- Madgwick, S. O. H. (2010). *An efficient orientation filter for inertial "
        "and inertial/magnetic sensor arrays.* Technical report, University of "
        "Bristol."
    )
    lines.append(
        "- Mahony, R., Hamel, T., & Pflimlin, J.-M. (2008). *Nonlinear complementary "
        "filters on the special orthogonal group.* IEEE Transactions on Automatic "
        "Control, 53(5), 1203–1218."
    )
    lines.append(
        "- x-io Technologies — Open-source IMU and AHRS algorithms: "
        "<https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/>"
    )
    lines.append(
        "- xioTechnologies/Fusion (modern C/C++ reference): "
        "<https://github.com/xioTechnologies/Fusion/tree/main>"
    )
    lines.append(
        "- Madgwick filter walk-through (cross-check): "
        "<https://medium.com/@k66115704/imu-madgwick-filter-explanation-556fbe7f02e3>"
    )
    lines.append("")

    lines.append("## Publication History & Tribute to Prof. René Jean Brenzikofer")
    lines.append("")
    lines.append(
        "Modern biomechanics is grounded in solid physical–mathematical foundations "
        "for modeling human movement. I leave here my profound tribute and final "
        "farewell to one of the greatest exponents of this scientific rigor in "
        "Brazil, Professor **René Jean Brenzikofer** (UNICAMP). It is an honor to "
        "have known him and to have shared creative ideas that demonstrated a "
        "biomechanics that truly makes a difference. This close collaboration "
        "enabled the **first publication of three-dimensional modeling data based "
        'on Quaternions** in the book *"Modelos Matemáticos nas Ciências '
        'Não-Exatas, vol. 1"* (Editora Blucher, 2007).'
    )
    lines.append("")
    lines.append("**Primary references (BibTeX-ready):**")
    lines.append("")
    lines.append(
        "- Nogueira, E. A., Martins, L. E. B., & Brenzikofer, R. (2007). "
        "*Modelos Matemáticos nas Ciências Não-Exatas — vol. 1.* São Paulo: "
        "Editora Blucher. ISBN 978-85-212-0419-0. "
        "<https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190>"
    )
    lines.append(
        "- Santiago, P. R. P. (2009). *Rotações tridimensionais em biomecânica via "
        "quatérnions: aplicações na análise dos movimentos esportivos.* Tese "
        "(Doutorado) — Universidade Estadual Paulista (Unesp), Instituto de "
        "Biociências de Rio Claro. <http://hdl.handle.net/11449/100404> · "
        "[PDF](https://repositorio.unesp.br/server/api/core/bitstreams/41603fa7-545b-4e74-a045-57ce94885e0c/content)"
    )
    lines.append("")
    lines.append("**Tribute & social-media coverage:**")
    lines.append("")
    lines.append(
        "- Instagram post (tribute & publication history): "
        "<https://www.instagram.com/p/DZLogIJoFbF/>"
    )
    lines.append(
        "- LinkedIn post (publication history & tribute to Prof. Brenzikofer): "
        "<https://www.linkedin.com/posts/paulo-roberto-pereira-santiago-132619112_hist%C3%B3rico-de-publica%C3%A7%C3%A3o-e-homenagem-ao-prof-ugcPost-7468670091845472257-3tzD/>"
    )
    lines.append("")
    lines.append("```bibtex")
    lines.append("@book{nogueira2007modelos,")
    lines.append("  title     = {Modelos matem\\'aticos nas ci\\^encias n\\~ao-exatas - vol. 1},")
    lines.append("  author    = {Nogueira, Eduardo Arantes and Martins, Luiz Eduardo Barreto")
    lines.append("               and Brenzikofer, Ren\\'e},")
    lines.append("  year      = {2007},")
    lines.append("  publisher = {Editora Blucher},")
    lines.append("  isbn      = {9788521204190},")
    lines.append(
        "  url       = {https://www.blucher.com.br/modelos-matematicos-nas-ciencias-nao-exatas-vol-1_9788521204190}"
    )
    lines.append("}")
    lines.append("")
    lines.append("@phdthesis{santiago2009rotaccoes,")
    lines.append(
        "  title    = {Rota\\c{c}\\~oes tridimensionais em biomec\\^anica via quat\\'ernions:"
    )
    lines.append("              aplica\\c{c}\\~oes na an\\'alise dos movimentos esportivos},")
    lines.append("  author   = {Santiago, Paulo Roberto Pereira},")
    lines.append("  school   = {Universidade Estadual Paulista (Unesp),")
    lines.append("              Instituto de Bioci\\^encias de Rio Claro},")
    lines.append("  year     = {2009},")
    lines.append("  url      = {http://hdl.handle.net/11449/100404}")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("— Prof. Paulo R. P. Santiago")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"Generated by **vailá** — Versatile Anarcho Integrated Liberation Ánalysis · "
        f"`vaila_deadlift_imu.py` v{VAILA_VERSION} · "
        "Author: Prof. Paulo R. P. Santiago · "
        "<https://github.com/vaila-multimodaltoolbox/vaila> · AGPL-3.0"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def process_imu_file(
    input_file: str,
    output_dir: str,
    filter_name: str = "madgwick",
    beta: float = 0.1,
    gyro_units: str = "deg",
    sample_rate: float | None = None,
    barbell_kg_override: float | None = None,
    body_mass_kg_override: float | None = None,
    body_height_m_override: float | None = None,
    reps_override: int | None = None,
    params_file: str | os.PathLike | None = None,
    cutoff_hz: float = DEFAULT_IMU_CUTOFF_HZ,
) -> bool:
    """Run the full AHRS deadlift IMU pipeline on a single CSV file.

    Parameter precedence (highest first):
    1. Explicit ``*_override`` arguments (CLI flags / GUI form).
    2. ``deadlift_parameters.txt`` (``params_file`` if given, otherwise the
       first match in the CSV's directory or up to three parents).
    3. ``vaila_deadlift_config.toml`` (legacy fallback).
    4. Hard-coded defaults (75 kg body mass, 20 kg barbell, 25 Hz).
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    required = {"acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(
            f"Input CSV {input_file} is missing required IMU columns: {sorted(missing)}"
        )
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    ctx = _load_context_from_toml(Path(input_file).parent)
    params = _load_parameters_file(Path(input_file).parent, explicit_path=params_file)

    fps = float(ctx.get("imu_fps", 25.0))
    barbell_kg = float(ctx.get("weight_kg", 20.0))
    body_mass_kg: float | None = float(ctx.get("mass_kg", 75.0)) if ctx else None
    body_height_m: float | None = None
    use_total_mass = bool(ctx.get("use_total_mass_for_power", True))

    # The shared ``deadlift_parameters.txt`` rep count is only used to read the
    # barbell weight. We deliberately do NOT trim the AHRS rep detection to the
    # annotated count, because that file frequently describes a different
    # capture than the IMU CSV. An explicit ``--reps`` (reps_override) is the
    # only way to force a fixed repetition count.
    expected_reps: int | None = reps_override

    if params:
        with contextlib.suppress(ValueError, TypeError):
            if "deadlift_mass_kg" in params:
                barbell_kg = float(params["deadlift_mass_kg"])
            elif "weight" in params:
                barbell_kg = float(params["weight"])
        with contextlib.suppress(ValueError, TypeError):
            if "subject_mass_kg" in params:
                body_mass_kg = float(params["subject_mass_kg"])
        with contextlib.suppress(ValueError, TypeError):
            if "subject_height_m" in params:
                body_height_m = float(params["subject_height_m"])
        with contextlib.suppress(ValueError, TypeError):
            if "fs_hz" in params:
                fps = float(params["fs_hz"])
        with contextlib.suppress(ValueError, TypeError):
            if "real_repetition_count" in params and expected_reps is None:
                # do NOT auto-trim by default (kept opt-in via --reps); just log
                pass

    # Explicit overrides from CLI/GUI win over everything.
    if sample_rate is not None:
        fps = float(sample_rate)
    if barbell_kg_override is not None:
        barbell_kg = float(barbell_kg_override)
    if body_mass_kg_override is not None:
        body_mass_kg = float(body_mass_kg_override)
    if body_height_m_override is not None:
        body_height_m = float(body_height_m_override)
    cutoff_hz = _valid_lowpass_cutoff(cutoff_hz, fps)

    src_label = params.get("__source__", "<defaults>") if params else "<defaults>"
    print(
        f"[IMU-AHRS] {base_name}: filter={filter_name}, fps={fps:.2f} Hz, "
        f"barbell={barbell_kg:.1f} kg, body_mass={body_mass_kg}, "
        f"height={body_height_m} m, use_total_mass={use_total_mass}, "
        f"butterworth_cutoff={cutoff_hz:.2f} Hz (params: {src_label})"
    )

    fusion = run_ahrs_fusion(df, fps=fps, filter_name=filter_name, beta=beta, gyro_units=gyro_units)
    print(
        f"[IMU-AHRS] |g| sensor={fusion.g_mag:.3f} m/s²  g_earth={fusion.g_earth.round(3).tolist()}"
    )

    reps, velocity, disp, a_filt = detect_reps_from_vertical_acc(
        fusion.acc_vert_ms2, fps=fps, cutoff_hz=cutoff_hz
    )
    reps = _trim_reps_to_expected(reps, expected_reps)
    print(f"[IMU-AHRS] Detected {len(reps)} repetitions (expected: {expected_reps})")
    fusion.velocity_ms[:] = velocity
    fusion.displacement_m[:] = disp

    rep_metrics = compute_rep_metrics(
        a_filt=a_filt,
        velocity=velocity,
        disp=disp,
        fps=fps,
        barbell_kg=barbell_kg,
        body_mass_kg=body_mass_kg,
        use_total_mass_for_power=use_total_mass,
        reps=reps,
    )
    cadence = compute_cadence(rep_metrics, fps)
    comparison = compute_rep_comparison(rep_metrics)

    plot_files = _generate_plots(
        fusion,
        a_filt,
        velocity,
        disp,
        reps,
        rep_metrics,
        output_dir,
        base_name,
        fps,
        filter_name,
        raw_acc=df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float),
    )
    report = _generate_html_report(
        rep_metrics,
        cadence,
        comparison,
        plot_files,
        output_dir,
        base_name,
        barbell_kg=barbell_kg,
        body_mass_kg=body_mass_kg,
        fps=fps,
        filter_name=filter_name,
        g_mag=fusion.g_mag,
        body_height_m=body_height_m,
    )
    report_md = _generate_md_report(
        rep_metrics,
        cadence,
        comparison,
        plot_files,
        output_dir,
        base_name,
        barbell_kg=barbell_kg,
        body_mass_kg=body_mass_kg,
        fps=fps,
        filter_name=filter_name,
        g_mag=fusion.g_mag,
        body_height_m=body_height_m,
    )

    ts_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    quat_geom = _quaternion_axis_angle_spherical(fusion.quaternion)

    proc = pd.DataFrame(
        {
            "time_s": fusion.time_s,
            "roll_deg": fusion.euler_deg[:, 0],
            "pitch_deg": fusion.euler_deg[:, 1],
            "yaw_deg": fusion.euler_deg[:, 2],
            "q0": fusion.quaternion[:, 0],
            "q1": fusion.quaternion[:, 1],
            "q2": fusion.quaternion[:, 2],
            "q3": fusion.quaternion[:, 3],
            "quat_norm": quat_geom["q_norm"],
            "quat_rotation_deg": quat_geom["rotation_deg"],
            "quat_axis_x": quat_geom["axis"][:, 0],
            "quat_axis_y": quat_geom["axis"][:, 1],
            "quat_axis_z": quat_geom["axis"][:, 2],
            "quat_axis_latitude_deg": quat_geom["latitude_deg"],
            "quat_axis_longitude_deg": quat_geom["longitude_deg"],
            "quat_axis_longitude_unwrapped_deg": quat_geom["longitude_unwrapped_deg"],
            "acc_earth_x": fusion.acc_earth_ms2[:, 0],
            "acc_earth_y": fusion.acc_earth_ms2[:, 1],
            "acc_earth_z": fusion.acc_earth_ms2[:, 2],
            "acc_linear_x": fusion.acc_linear_ms2[:, 0],
            "acc_linear_y": fusion.acc_linear_ms2[:, 1],
            "acc_linear_z": fusion.acc_linear_ms2[:, 2],
            "a_vert_linear_ms2": a_filt,
            "butterworth_cutoff_hz": np.full_like(fusion.time_s, cutoff_hz, dtype=float),
            "velocity_ms": velocity,
            "displacement_m": disp,
            "translation_world_x_m": np.zeros_like(fusion.time_s),
            "translation_world_y_m": np.zeros_like(fusion.time_s),
            "translation_world_z_m": disp,
        }
    )
    proc_path = os.path.join(output_dir, f"{base_name}_imu_ahrs_processed_{ts_stamp}.csv")
    proc.to_csv(proc_path, index=False, float_format="%.6f")

    if rep_metrics:
        pd.DataFrame(rep_metrics).to_csv(
            os.path.join(output_dir, f"{base_name}_imu_ahrs_rep_metrics_{ts_stamp}.csv"),
            index=False,
            float_format="%.6f",
        )

    print("\n[SUCCESS] IMU Deadlift AHRS Analysis Complete")
    print(f"  Reps: {cadence['n_reps']}")
    print(f"  Cadence: {cadence.get('cadence_rpm', 0):.1f} reps/min")
    if rep_metrics:
        best_v = max(r["peak_velocity_ms"] for r in rep_metrics)
        best_p = max(r["peak_power_W"] for r in rep_metrics)
        print(f"  Best peak velocity: {best_v:.3f} m/s")
        print(f"  Best peak power: {best_p:.1f} W")
    print(f"  Report (HTML): {report}")
    print(f"  Report (Markdown): {report_md}")
    print(f"  Processed CSV: {proc_path}")
    return True


def _resolve_filter_names(filter_name: str) -> list[str]:
    """Resolve the requested AHRS filter selection to one or both filters."""
    requested = (filter_name or "both").strip().lower()
    if requested in {"both", "all", "madgwick+mahony", "mahony+madgwick"}:
        return ["madgwick", "mahony"]
    if requested in {"madgwick", "mahony"}:
        return [requested]
    raise ValueError("filter_name must be 'both', 'madgwick' or 'mahony'")


def _filter_selection_label(filter_name: str) -> str:
    filters = _resolve_filter_names(filter_name)
    if len(filters) == 2:
        return "madgwick + mahony"
    return filters[0]


def _write_filter_choice_index(
    output_dir: str,
    base_name: str,
    reports: dict[str, Path],
    processed_csvs: dict[str, Path],
) -> str:
    """Write a small chooser page linking the Madgwick and Mahony outputs."""
    out = Path(output_dir)
    html_path = out / f"{base_name}_imu_ahrs_filter_index.html"
    rows = []
    md_lines = [f"# {base_name} - AHRS filter outputs", ""]
    for name in ["madgwick", "mahony"]:
        report = reports.get(name)
        proc = processed_csvs.get(name)
        if report is None:
            rows.append(f"<tr><td>{name.title()}</td><td colspan='2'>failed or not generated</td></tr>")
            continue
        report_rel = report.relative_to(out).as_posix()
        proc_rel = proc.relative_to(out).as_posix() if proc else ""
        rows.append(
            "<tr>"
            f"<td>{name.title()}</td>"
            f"<td><a href='{report_rel}'>Open HTML report</a></td>"
            f"<td><a href='{proc_rel}'>Processed CSV</a></td>"
            "</tr>"
        )
        md_lines.append(f"- **{name.title()}**: [HTML report]({report_rel})")
        if proc_rel:
            md_lines.append(f"  - [Processed CSV]({proc_rel})")
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vailá - Deadlift IMU AHRS filter chooser</title>
<style>
body{font-family:Arial,sans-serif;margin:32px;background:#f8fafc;color:#0f172a}
.wrap{max-width:860px;margin:0 auto;background:#fff;border:1px solid #cbd5e1;border-radius:8px;padding:20px}
table{width:100%;border-collapse:collapse;margin-top:16px}th,td{border:1px solid #e2e8f0;padding:10px;text-align:left}th{background:#e0f2fe}
a{color:#0369a1}.note{color:#475569;line-height:1.45}
</style>
</head>
<body><div class="wrap">
<h1>Deadlift IMU AHRS filter outputs</h1>
<p class="note">Both Madgwick and Mahony were processed from the same IMU file. Open either report to compare orientation, quaternion animation, velocity, displacement and repetition metrics.</p>
<table><tr><th>Filter</th><th>Report</th><th>Processed data</th></tr>
""" + "\n".join(rows) + """
</table>
</div></body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    (out / f"{base_name}_imu_ahrs_filter_index.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )
    print(f"[IMU-AHRS] Filter chooser: {html_path}")
    return str(html_path)


def process_imu_file_filters(
    input_file: str,
    output_dir: str,
    filter_name: str = "both",
    beta: float = 0.1,
    gyro_units: str = "deg",
    sample_rate: float | None = None,
    barbell_kg_override: float | None = None,
    body_mass_kg_override: float | None = None,
    body_height_m_override: float | None = None,
    reps_override: int | None = None,
    params_file: str | os.PathLike | None = None,
    cutoff_hz: float = DEFAULT_IMU_CUTOFF_HZ,
) -> dict[str, bool]:
    """Run one or both AHRS filters for a single IMU CSV file."""
    filters = _resolve_filter_names(filter_name)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    results: dict[str, bool] = {}
    reports: dict[str, Path] = {}
    processed_csvs: dict[str, Path] = {}

    for name in filters:
        filter_out = os.path.join(output_dir, name) if len(filters) > 1 else output_dir
        os.makedirs(filter_out, exist_ok=True)
        print(f"\n[IMU-AHRS] Running {name.title()} filter for {base_name}")
        try:
            process_imu_file(
                input_file,
                filter_out,
                filter_name=name,
                beta=beta,
                gyro_units=gyro_units,
                sample_rate=sample_rate,
                barbell_kg_override=barbell_kg_override,
                body_mass_kg_override=body_mass_kg_override,
                body_height_m_override=body_height_m_override,
                reps_override=reps_override,
                params_file=params_file,
                cutoff_hz=cutoff_hz,
            )
            results[name] = True
            report_path = Path(filter_out) / f"{base_name}_imu_ahrs_report.html"
            if report_path.exists():
                reports[name] = report_path
            csv_candidates = sorted(Path(filter_out).glob(f"{base_name}_imu_ahrs_processed_*.csv"))
            if csv_candidates:
                processed_csvs[name] = csv_candidates[-1]
        except Exception as exc:
            results[name] = False
            print(f"[IMU-AHRS] ERROR running {name} for {base_name}: {exc}")
            traceback.print_exc()

    if len(filters) > 1 and reports:
        _write_filter_choice_index(output_dir, base_name, reports, processed_csvs)
    return results


# ---------------------------------------------------------------------------
# CLI / GUI feedback helpers
# ---------------------------------------------------------------------------


DEFAULT_SETTINGS = {
    "subject_mass_kg": 75.0,
    "subject_height_m": 1.75,
    "deadlift_mass_kg": 20.0,
    "fs_hz": 25.0,
    "filter_name": "both",
    "beta": 0.1,
    "gyro_units": "deg",
    "cutoff_hz": DEFAULT_IMU_CUTOFF_HZ,
}


def _format_cli_command(
    *,
    input_dir: str = "",
    output_dir: str = "",
    filter_name: str = "both",
    beta: float = 0.1,
    gyro_units: str = "deg",
    subject_mass_kg: float = 75.0,
    subject_height_m: float = 1.75,
    deadlift_mass_kg: float = 20.0,
    fs_hz: float = 25.0,
    cutoff_hz: float = DEFAULT_IMU_CUTOFF_HZ,
    reps: int | None = None,
    params_file: str = "",
    batch: bool = True,
) -> str:
    """Build a copy-pasteable CLI command equivalent to the current settings.

    When ``batch`` is True the command uses ``-d`` (directory mode); otherwise
    it uses ``-i`` (single file). Unset paths are rendered as ``<input_dir>``
    / ``<output_dir>`` placeholders so the snippet is still informative when
    the user has not picked directories yet.
    """
    in_str = input_dir or ("<input_dir>" if batch else "<input.csv>")
    out_str = output_dir or "<output_dir>"
    flag = "-d" if batch else "-i"
    lines = [
        "uv run python vaila/vaila_deadlift_imu.py \\",
        f'  {flag} "{in_str}" \\',
        f'  -o "{out_str}" \\',
        f"  --filter {filter_name} --beta {beta} --gyro-units {gyro_units} \\",
        (
            f"  --weight {deadlift_mass_kg} --mass {subject_mass_kg} "
            f"--height {subject_height_m} --fps {fs_hz} --cutoff {cutoff_hz}"
        ),
    ]
    extras: list[str] = []
    if reps is not None:
        extras.append(f"--reps {reps}")
    if params_file:
        extras.append(f'--params-file "{params_file}"')
    if extras:
        lines[-1] += " \\"
        lines.append("  " + " ".join(extras))
    return "\n".join(lines)


def _print_cli_help_banner() -> None:
    """Print a quick-reference CLI banner to the terminal.

    Called at the top of :func:`main_gui` so the user sees the CLI alternatives
    and current defaults *before* the Tkinter dialog steals focus.
    """
    bar = "=" * 70
    print(bar)
    print("  vailá - Deadlift IMU (AHRS)  -  GUI mode")
    print(bar)
    print("Required CSV columns: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z")
    print("                     (optional: mag_x, mag_y, mag_z)")
    print()
    print("Defaults (override in the form or via CLI flags):")
    rows: list[tuple[str, str, str]] = [
        ("subject_mass", f"{DEFAULT_SETTINGS['subject_mass_kg']} kg", "[--mass]"),
        ("subject_height", f"{DEFAULT_SETTINGS['subject_height_m']} m", "[--height]"),
        ("barbell mass", f"{DEFAULT_SETTINGS['deadlift_mass_kg']} kg", "[--weight]"),
        ("sample rate", f"{DEFAULT_SETTINGS['fs_hz']} Hz", "[--fps]"),
        ("Butterworth cutoff", f"{DEFAULT_SETTINGS['cutoff_hz']} Hz", "[--cutoff]"),
        ("AHRS filters", "madgwick + mahony", "[--filter both|madgwick|mahony]"),
        ("beta", f"{DEFAULT_SETTINGS['beta']}", "[--beta]"),
        ("gyro units", f"{DEFAULT_SETTINGS['gyro_units']}", "[--gyro-units deg|rad]"),
        ("reps", "auto-detect", "[--reps N]"),
    ]
    for name, value, flag in rows:
        print(f"  {name:<15} = {value:<14}  {flag}")
    print()
    print("Equivalent CLI invocations:")
    print("  # single file:")
    print("  uv run python vaila/vaila_deadlift_imu.py -i deadlift_imu.csv  # runs both filters")
    print("  # single file with explicit params file:")
    print("  uv run python vaila/vaila_deadlift_imu.py -i deadlift_imu.csv -p deadlift_parameters.txt")
    print("  # whole directory (batch, same as the GUI):")
    print("  uv run python vaila/vaila_deadlift_imu.py -d /path/to/csv_dir -o /path/to/out -p /path/to/deadlift_parameters.txt")
    print("  # full help:")
    print("  uv run python vaila/vaila_deadlift_imu.py --help")
    print()
    print("Params file (-p/--params-file; auto-loaded if found next to the CSV):")
    print("  deadlift_parameters.txt  (CSV header format)")
    print("    subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz")
    print("    75.0,1.75,20.0,25.0")
    print(bar)
    print("Opening configuration dialog...", flush=True)


# ---------------------------------------------------------------------------
# GUI entry point
# ---------------------------------------------------------------------------


@dataclass
class _GuiResult:
    """Container for the values returned by the unified Deadlift IMU dialog."""

    input_dir: str = ""
    output_dir: str = ""
    params_file: str = ""
    subject_mass_kg: float = 75.0
    subject_height_m: float = 1.75
    deadlift_mass_kg: float = 20.0
    fs_hz: float = 25.0
    cutoff_hz: float = DEFAULT_IMU_CUTOFF_HZ
    filter_name: str = "both"
    beta: float = 0.1
    gyro_units: str = "deg"
    reps: int | None = None
    cancelled: bool = True


def _show_deadlift_imu_dialog() -> _GuiResult:
    """Single-window Tkinter form for Deadlift IMU analysis.

    The dialog exposes:

    * pre-filled defaults for every numeric field;
    * a live "Equivalent CLI command" preview that updates as the user types;
    * a "Status / Log" text widget that mirrors directory-pick and parameter
      events to the GUI (the same events are also printed to stdout, so the
      user gets the exact same feedback whether they are looking at the GUI
      or at the terminal);
    * a "CLI help" button that opens a popup with the full CLI usage.

    Returns a :class:`_GuiResult` with ``cancelled=True`` if the user closed the
    dialog or clicked "Cancel". All inputs are validated; invalid numeric
    fields are rejected with a ``messagebox.showerror``.
    """
    result = _GuiResult()

    parent = tk._default_root
    owns_parent = False
    if parent is None:
        parent = Tk()
        parent.withdraw()
        owns_parent = True

    root = tk.Toplevel(parent)
    root.title("vailá - Deadlift IMU (AHRS) - Single-form input")
    root.attributes("-topmost", True)
    with contextlib.suppress(Exception):  # non-graphical envs
        root.geometry("980x900")
    with contextlib.suppress(Exception):
        if parent.winfo_viewable():
            root.transient(parent)

    PX, PY = 8, 4

    header = tk.Label(
        root,
        text="Deadlift IMU (AHRS) - fill the form once, then click Run.",
        font=("TkDefaultFont", 11, "bold"),
        justify="left",
    )
    header.grid(row=0, column=0, columnspan=3, sticky="w", padx=PX, pady=PY)

    # ---- Status / Log widget (declared early so callbacks can write to it) ---
    log_text: tk.Text | None = None
    status_var = tk.StringVar(value="Ready.")

    def _log(msg: str) -> None:
        """Append ``msg`` to the in-GUI log widget AND to stdout."""
        print(msg, flush=True)
        with contextlib.suppress(tk.TclError):
            status_var.set(msg)
        if log_text is not None:
            with contextlib.suppress(tk.TclError):
                log_text.config(state="normal")
                log_text.insert("end", msg + "\n")
                log_text.see("end")
                log_text.config(state="disabled")
                root.update_idletasks()

    def _show_cli_help_popup() -> None:
        popup = tk.Toplevel(root)
        popup.title("Deadlift IMU - CLI usage")
        popup.geometry("780x520")
        txt = tk.Text(popup, wrap="word", font=("TkFixedFont", 10))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        help_body = (
            "vailá - Deadlift IMU (AHRS) - CLI reference\n"
            "=====================================================================\n"
            "\n"
            "Required CSV columns:\n"
            "  acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z\n"
            "  optional: mag_x, mag_y, mag_z   (enables magnetometer-aided AHRS)\n"
            "\n"
            "Default values (used when no field is set and no params file is loaded):\n"
            f"  subject_mass = {DEFAULT_SETTINGS['subject_mass_kg']} kg     [--mass]\n"
            f"  subject_height = {DEFAULT_SETTINGS['subject_height_m']} m   [--height]\n"
            f"  barbell mass = {DEFAULT_SETTINGS['deadlift_mass_kg']} kg    [--weight]\n"
            f"  sample rate  = {DEFAULT_SETTINGS['fs_hz']} Hz               [--fps]\n"
            f"  cutoff       = {DEFAULT_SETTINGS['cutoff_hz']} Hz                [--cutoff]\n"
            "  filters      = madgwick + mahony        [--filter both|madgwick|mahony]\n"
            f"  beta         = {DEFAULT_SETTINGS['beta']}                   [--beta]\n"
            f"  gyro_units   = {DEFAULT_SETTINGS['gyro_units']}             [--gyro-units deg|rad]\n"
            "  reps         = auto-detect                       [--reps N]\n"
            "\n"
            "CLI invocations\n"
            "---------------\n"
            "1) Open this same Tkinter form from a terminal:\n"
            "   uv run python vaila/vaila_deadlift_imu.py\n"
            "\n"
            '2) Single CSV file:\n'
            '   uv run python vaila/vaila_deadlift_imu.py \\\n'
            '     -i /path/to/deadlift_imu.csv \\\n'
            '     -o /path/to/output_dir \\\n'
            '     -p /path/to/deadlift_parameters.txt \\\n'
            '     --filter both --beta 0.1 --gyro-units deg \\\n'
            '     --weight 20.0 --mass 75.0 --height 1.75 --fps 25.0\n'
            '\n'
            '3) Whole directory (batch, mirrors the GUI - processes every *.csv):\n'
            '   uv run python vaila/vaila_deadlift_imu.py \\\n'
            '     -d /path/to/csv_dir -o /path/to/output_dir \\\n'
            '     -p /path/to/deadlift_parameters.txt \\\n'
            '     --filter both --beta 0.1 --gyro-units deg\n'
            '\n'
            '4) Reuse the bundled example params file explicitly:\n'
            '   uv run python vaila/vaila_deadlift_imu.py \\\n'
            '     -d tests/Deadlift/imu \\\n'
            '     -p tests/Deadlift/imu/deadlift_parameters.txt \\\n'
            '     -o /tmp/dlift_out\n'
            '\n'
            "Params file format (CSV header preferred)\n"
            "-----------------------------------------\n"
            "  subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz\n"
            "  75.0,1.75,20.0,25.0\n"
            "\n"
            "Legacy 'key,value' lines are still accepted (e.g. weight,20).\n"
            "\n"
            "Show full argparse help:\n"
            "  uv run python vaila/vaila_deadlift_imu.py --help\n"
        )
        txt.insert("1.0", help_body)
        txt.config(state="disabled")
        tk.Button(popup, text="Close", command=popup.destroy).pack(pady=6)

    # Brief tagline + Help button row
    tag_frame = tk.Frame(root)
    tag_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=PX, pady=2)
    tag_frame.grid_columnconfigure(0, weight=1)
    tk.Label(
        tag_frame,
        text=(
            "Required CSV cols: acc_x/y/z, gyr_x/y/z  |  defaults pre-filled below  |  "
            "click 'CLI help' for the equivalent CLI invocations."
        ),
        fg="#444",
        justify="left",
        wraplength=720,
    ).grid(row=0, column=0, sticky="w")
    tk.Button(tag_frame, text="CLI help", command=_show_cli_help_popup, width=10).grid(
        row=0, column=1, sticky="e", padx=(8, 0)
    )
    tk.Label(
        tag_frame,
        textvariable=status_var,
        fg="#1d4ed8",
        justify="left",
        wraplength=820,
    ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))

    # ---- Directory pickers ---------------------------------------------------
    in_var = tk.StringVar()
    out_var = tk.StringVar()
    params_var = tk.StringVar()
    csv_preview_var = tk.StringVar(value="CSV files: select an input directory to preview files.")

    # Entry widget references (filled in below). Used by `_show_full_path`
    # so we can scroll the Entry to the end after a path is set - this is the
    # *only* reliable way to make a long Windows path like
    # ``C:\Users\<name>\Documents\<...>\imu_data`` visually show the trailing
    # (meaningful) portion in a Tk Entry. Without xview_moveto(1.0), Tk parks
    # the view at character 0 and the user sees only ``C:\Users\<n`` which
    # *looks* like the dir was not picked at all.
    in_entry: tk.Entry | None = None
    out_entry: tk.Entry | None = None
    params_entry: tk.Entry | None = None

    def _show_full_path(entry: tk.Entry | None) -> None:
        if entry is None:
            return
        with contextlib.suppress(tk.TclError):
            entry.update_idletasks()
            entry.icursor("end")
            entry.xview_moveto(1.0)

    def _initial_dir(*candidates: str) -> str:
        for candidate in candidates:
            if not candidate:
                continue
            p = Path(candidate).expanduser()
            if p.is_file():
                return str(p.parent)
            if p.is_dir():
                return str(p)
        return str(Path.cwd())

    def _update_csv_preview(directory: str) -> int:
        try:
            files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".csv"))
        except OSError as exc:
            csv_preview_var.set(f"CSV files: cannot read directory ({exc})")
            return 0
        if not files:
            csv_preview_var.set("CSV files found: 0")
            return 0
        shown = ", ".join(files[:6])
        if len(files) > 6:
            shown += f", ... (+{len(files) - 6} more)"
        csv_preview_var.set(f"CSV files found ({len(files)}): {shown}")
        return len(files)

    def _pick_input_dir() -> None:
        start_dir = _initial_dir(in_var.get(), out_var.get())
        _log(f"[GUI] Opening input directory browser at: {start_dir}")
        d = filedialog.askdirectory(
            title="Select directory containing Deadlift IMU CSV files (acc + gyr)",
            parent=root,
            initialdir=start_dir,
        )
        if not d:
            _log("[GUI] Input directory selection cancelled.")
            return
        # On Windows the askdirectory return value uses forward slashes; keep
        # it as-is (Path handles either) so the visible string matches what
        # users typed in Explorer.
        in_var.set(d)
        _show_full_path(in_entry)
        csv_count = _update_csv_preview(d)
        _log(f"[GUI] Input dir selected: {d}  ({csv_count} CSV file(s) found)")
        if not out_var.get():
            out_var.set(d)
            _show_full_path(out_entry)
            _log(f"[GUI] Output dir auto-set to input dir: {d}")
        # Auto-detect a sibling params file
        candidate = Path(d) / "deadlift_parameters.txt"
        if candidate.exists() and not params_var.get():
            params_var.set(str(candidate))
            _show_full_path(params_entry)
            _log(f"[GUI] Auto-detected params file: {candidate}")
            _load_params_into_form(str(candidate))

    def _pick_output_dir() -> None:
        start_dir = _initial_dir(out_var.get(), in_var.get())
        _log(f"[GUI] Opening output directory browser at: {start_dir}")
        d = filedialog.askdirectory(
            title="Select output directory",
            parent=root,
            initialdir=start_dir,
        )
        if not d:
            _log("[GUI] Output directory selection cancelled.")
            return
        out_var.set(d)
        _show_full_path(out_entry)
        _log(f"[GUI] Output dir selected: {d}")

    def _pick_params_file() -> None:
        start_dir = _initial_dir(params_var.get(), in_var.get(), out_var.get())
        _log(f"[GUI] Opening params-file browser at: {start_dir}")
        f = filedialog.askopenfilename(
            title="Select deadlift_parameters.txt",
            parent=root,
            initialdir=start_dir,
            filetypes=[("Parameters", "*.txt"), ("All files", "*.*")],
        )
        if not f:
            _log("[GUI] Params file selection cancelled.")
            return
        params_var.set(f)
        _show_full_path(params_entry)
        _log(f"[GUI] Params file selected: {f}")
        _load_params_into_form(f)

    def _load_params_into_form(path: str) -> None:
        params = _parse_parameters_file(Path(path))
        if not params:
            _log(f"[GUI] Params file is empty or unparseable: {path}")
            return
        loaded: list[str] = []
        with contextlib.suppress(ValueError, TypeError):
            if "subject_mass_kg" in params:
                mass_var.set(f"{float(params['subject_mass_kg']):.2f}")
                loaded.append(f"subject_mass={params['subject_mass_kg']}")
        with contextlib.suppress(ValueError, TypeError):
            if "subject_height_m" in params:
                height_var.set(f"{float(params['subject_height_m']):.2f}")
                loaded.append(f"subject_height={params['subject_height_m']}")
        with contextlib.suppress(ValueError, TypeError):
            if "deadlift_mass_kg" in params:
                bar_var.set(f"{float(params['deadlift_mass_kg']):.2f}")
                loaded.append(f"deadlift_mass={params['deadlift_mass_kg']}")
            elif "weight" in params:
                bar_var.set(f"{float(params['weight']):.2f}")
                loaded.append(f"weight={params['weight']}")
        with contextlib.suppress(ValueError, TypeError):
            if "fs_hz" in params:
                fs_var.set(f"{float(params['fs_hz']):.2f}")
                loaded.append(f"fs_hz={params['fs_hz']}")
        with contextlib.suppress(ValueError, TypeError):
            if "real_repetition_count" in params:
                reps_var.set(str(int(float(params["real_repetition_count"]))))
                loaded.append(f"reps={params['real_repetition_count']}")
        _log(f"[GUI] Loaded params: {', '.join(loaded) if loaded else '(none)'}")

    tk.Label(root, text="Input dir (CSVs):").grid(row=2, column=0, sticky="e", padx=PX, pady=PY)
    in_entry = tk.Entry(root, textvariable=in_var, width=70)
    in_entry.grid(row=2, column=1, padx=PX, pady=PY, sticky="ew")
    tk.Button(root, text="Browse...", command=_pick_input_dir).grid(
        row=2, column=2, padx=PX, pady=PY
    )

    tk.Label(root, text="Output dir:").grid(row=3, column=0, sticky="e", padx=PX, pady=PY)
    out_entry = tk.Entry(root, textvariable=out_var, width=70)
    out_entry.grid(row=3, column=1, padx=PX, pady=PY, sticky="ew")
    tk.Button(root, text="Browse...", command=_pick_output_dir).grid(
        row=3, column=2, padx=PX, pady=PY
    )

    tk.Label(root, text="Params file (optional):").grid(
        row=4, column=0, sticky="e", padx=PX, pady=PY
    )
    params_entry = tk.Entry(root, textvariable=params_var, width=70)
    params_entry.grid(row=4, column=1, padx=PX, pady=PY, sticky="ew")
    tk.Button(root, text="Load .txt...", command=_pick_params_file).grid(
        row=4, column=2, padx=PX, pady=PY
    )

    tk.Label(root, text="CSV preview:").grid(row=5, column=0, sticky="e", padx=PX, pady=PY)
    tk.Label(
        root,
        textvariable=csv_preview_var,
        anchor="w",
        justify="left",
        wraplength=720,
        fg="#444",
    ).grid(row=5, column=1, columnspan=2, sticky="ew", padx=PX, pady=PY)

    # Let the middle column grow so long paths use the full window width
    # (helps on Windows where default Tk DPI makes ``width=60`` only ~480 px).
    root.grid_columnconfigure(1, weight=1)

    ttk.Separator(root, orient="horizontal").grid(
        row=6, column=0, columnspan=3, sticky="ew", padx=8, pady=8
    )

    # ---- Numeric subject / bar / fps parameters -----------------------------
    mass_var = tk.StringVar(value=str(DEFAULT_SETTINGS["subject_mass_kg"]))
    height_var = tk.StringVar(value=str(DEFAULT_SETTINGS["subject_height_m"]))
    bar_var = tk.StringVar(value=str(DEFAULT_SETTINGS["deadlift_mass_kg"]))
    fs_var = tk.StringVar(value=str(DEFAULT_SETTINGS["fs_hz"]))
    cutoff_var = tk.StringVar(value=str(DEFAULT_SETTINGS["cutoff_hz"]))
    reps_var = tk.StringVar(value="")

    def _row(label: str, var: tk.StringVar, row: int, hint: str = "") -> None:
        tk.Label(root, text=label).grid(row=row, column=0, sticky="e", padx=PX, pady=PY)
        tk.Entry(root, textvariable=var, width=14).grid(
            row=row, column=1, sticky="w", padx=PX, pady=PY
        )
        if hint:
            tk.Label(root, text=hint, fg="#666").grid(
                row=row, column=2, sticky="w", padx=PX, pady=PY
            )

    _row("Subject mass (kg):", mass_var, 7, hint="default 75.0")
    _row("Subject height (m):", height_var, 8, hint="default 1.75 (informational)")
    _row("Deadlift bar mass (kg):", bar_var, 9, hint="default 20.0")
    _row("Sample rate (Hz):", fs_var, 10, hint="default 25.0 (IMU sampling frequency)")
    _row("Butterworth cutoff (Hz):", cutoff_var, 11, hint="default 4.0; auto-clamped below Nyquist")
    _row("Expected reps (optional):", reps_var, 12, hint="leave blank for auto-detect")

    ttk.Separator(root, orient="horizontal").grid(
        row=13, column=0, columnspan=3, sticky="ew", padx=8, pady=8
    )

    # ---- AHRS filter settings ----------------------------------------------
    filter_var = tk.StringVar(value=str(DEFAULT_SETTINGS["filter_name"]))
    beta_var = tk.StringVar(value=str(DEFAULT_SETTINGS["beta"]))
    gyro_units_var = tk.StringVar(value=str(DEFAULT_SETTINGS["gyro_units"]))

    tk.Label(root, text="AHRS filters:").grid(row=14, column=0, sticky="e", padx=PX, pady=PY)
    tk.Label(
        root,
        text="Madgwick + Mahony (both processed)",
        fg="#0f172a",
    ).grid(row=14, column=1, sticky="w", padx=PX, pady=PY)
    tk.Label(root, text="CLI can restrict with --filter madgwick|mahony", fg="#666").grid(
        row=14, column=2, sticky="w", padx=PX, pady=PY
    )

    tk.Label(root, text="Beta (Madgwick):").grid(row=15, column=0, sticky="e", padx=PX, pady=PY)
    tk.Entry(root, textvariable=beta_var, width=14).grid(
        row=15, column=1, sticky="w", padx=PX, pady=PY
    )
    tk.Label(root, text="default 0.1", fg="#666").grid(
        row=15, column=2, sticky="w", padx=PX, pady=PY
    )

    tk.Label(root, text="Gyro units:").grid(row=16, column=0, sticky="e", padx=PX, pady=PY)
    ttk.Combobox(
        root,
        textvariable=gyro_units_var,
        values=["deg", "rad"],
        width=12,
        state="readonly",
    ).grid(row=16, column=1, sticky="w", padx=PX, pady=PY)
    tk.Label(root, text="default deg (deg/s)", fg="#666").grid(
        row=16, column=2, sticky="w", padx=PX, pady=PY
    )

    ttk.Separator(root, orient="horizontal").grid(
        row=17, column=0, columnspan=3, sticky="ew", padx=8, pady=8
    )

    # ---- Equivalent CLI command preview (live) -----------------------------
    tk.Label(
        root,
        text="Equivalent CLI command (auto-updates - copy/paste-ready):",
        font=("TkDefaultFont", 9, "bold"),
    ).grid(row=18, column=0, columnspan=3, sticky="w", padx=PX)

    cli_text = tk.Text(
        root, height=6, wrap="none", bg="#0f172a", fg="#e2e8f0", font=("TkFixedFont", 9)
    )
    cli_text.grid(row=19, column=0, columnspan=3, sticky="ew", padx=PX, pady=2)

    def _update_cli_preview(*_args) -> None:
        try:
            reps_val: int | None = int(reps_var.get()) if reps_var.get().strip() else None
        except ValueError:
            reps_val = None
        try:
            cmd = _format_cli_command(
                input_dir=in_var.get(),
                output_dir=out_var.get() or in_var.get(),
                filter_name=filter_var.get(),
                beta=float(beta_var.get() or DEFAULT_SETTINGS["beta"]),
                gyro_units=gyro_units_var.get(),
                subject_mass_kg=float(mass_var.get() or DEFAULT_SETTINGS["subject_mass_kg"]),
                subject_height_m=float(height_var.get() or DEFAULT_SETTINGS["subject_height_m"]),
                deadlift_mass_kg=float(bar_var.get() or DEFAULT_SETTINGS["deadlift_mass_kg"]),
                fs_hz=float(fs_var.get() or DEFAULT_SETTINGS["fs_hz"]),
                cutoff_hz=float(cutoff_var.get() or DEFAULT_SETTINGS["cutoff_hz"]),
                reps=reps_val,
                params_file=params_var.get(),
                batch=True,
            )
        except ValueError:
            cmd = "(invalid numeric field - fix to refresh preview)"
        cli_text.config(state="normal")
        cli_text.delete("1.0", "end")
        cli_text.insert("1.0", cmd)
        cli_text.config(state="disabled")

    for v in (
        in_var,
        out_var,
        params_var,
        mass_var,
        height_var,
        bar_var,
        fs_var,
        reps_var,
        filter_var,
        beta_var,
        gyro_units_var,
        cutoff_var,
    ):
        v.trace_add("write", _update_cli_preview)
    _update_cli_preview()  # initial render with defaults

    # ---- Status / log widget -----------------------------------------------
    tk.Label(
        root,
        text="Status / Log (mirrored to terminal):",
        font=("TkDefaultFont", 9, "bold"),
    ).grid(row=20, column=0, columnspan=3, sticky="w", padx=PX, pady=(6, 0))

    log_frame = tk.Frame(root)
    log_frame.grid(row=21, column=0, columnspan=3, sticky="ew", padx=PX, pady=2)
    log_scroll = tk.Scrollbar(log_frame, orient="vertical")
    log_scroll.pack(side="right", fill="y")
    log_text = tk.Text(
        log_frame,
        height=8,
        wrap="word",
        bg="#f8fafc",
        font=("TkFixedFont", 9),
        yscrollcommand=log_scroll.set,
        state="disabled",
    )
    log_text.pack(side="left", fill="both", expand=True)
    log_scroll.config(command=log_text.yview)

    _log("[GUI] Deadlift IMU (AHRS) dialog ready. Pick input dir to begin.")
    _log(
        "[GUI] Defaults: mass={mass} kg, height={h} m, bar={bar} kg, fs={fs} Hz, "
        "filters=madgwick+mahony, beta={b}, gyro={g}, cutoff={c} Hz.".format(
            mass=DEFAULT_SETTINGS["subject_mass_kg"],
            h=DEFAULT_SETTINGS["subject_height_m"],
            bar=DEFAULT_SETTINGS["deadlift_mass_kg"],
            fs=DEFAULT_SETTINGS["fs_hz"],
            b=DEFAULT_SETTINGS["beta"],
            g=DEFAULT_SETTINGS["gyro_units"],
            c=DEFAULT_SETTINGS["cutoff_hz"],
        )
    )

    # ---- Action buttons -----------------------------------------------------
    btn_frame = tk.Frame(root)
    btn_frame.grid(row=22, column=0, columnspan=3, pady=12)

    def _on_run() -> None:
        if not in_var.get():
            _log("[GUI] ERROR: input directory is required.")
            messagebox.showerror("Error", "Input directory is required.", parent=root)
            return
        try:
            result.subject_mass_kg = float(mass_var.get())
            result.subject_height_m = float(height_var.get())
            result.deadlift_mass_kg = float(bar_var.get())
            result.fs_hz = float(fs_var.get())
            result.cutoff_hz = float(cutoff_var.get())
            result.beta = float(beta_var.get())
        except ValueError as e:
            _log(f"[GUI] ERROR: invalid numeric field: {e}")
            messagebox.showerror("Error", f"Invalid numeric field: {e}", parent=root)
            return
        if reps_var.get().strip():
            try:
                result.reps = int(reps_var.get())
            except ValueError:
                _log("[GUI] ERROR: expected reps must be an integer.")
                messagebox.showerror("Error", "Expected reps must be an integer.", parent=root)
                return
        result.input_dir = in_var.get()
        result.output_dir = out_var.get() or in_var.get()
        result.params_file = params_var.get()
        result.filter_name = filter_var.get()
        result.gyro_units = gyro_units_var.get()
        result.cancelled = False
        _log("[GUI] Inputs validated. Closing dialog and starting batch analysis...")
        with contextlib.suppress(tk.TclError):
            root.update_idletasks()
            root.grab_release()
        root.destroy()

    def _on_cancel() -> None:
        result.cancelled = True
        _log("[GUI] Cancelled by user.")
        with contextlib.suppress(tk.TclError):
            root.grab_release()
        root.destroy()

    tk.Button(
        btn_frame, text="Run analysis", command=_on_run, width=16, bg="#2563eb", fg="white"
    ).pack(side="left", padx=6)
    tk.Button(btn_frame, text="CLI help", command=_show_cli_help_popup, width=12).pack(
        side="left", padx=6
    )
    tk.Button(btn_frame, text="Cancel", command=_on_cancel, width=10).pack(side="left", padx=6)

    root.protocol("WM_DELETE_WINDOW", _on_cancel)

    # Defensive: re-assert the visible value of every default StringVar AFTER
    # the layout is settled. On Windows + Tk 8.6 with HiDPI the initial render
    # occasionally drops the Entry contents until the widget is focused; the
    # explicit ``.set(.get())`` round-trip below forces a redraw with the
    # already-stored default (no value actually changes).
    def _reassert_defaults() -> None:
        for v in (
            mass_var,
            height_var,
            bar_var,
            fs_var,
            cutoff_var,
            beta_var,
            filter_var,
            gyro_units_var,
        ):
            v.set(v.get())
        for entry in (in_entry, out_entry, params_entry):
            _show_full_path(entry)

    root.after(50, _reassert_defaults)
    with contextlib.suppress(Exception):
        root.update_idletasks()
        root.lift()
        root.focus_force()
        root.grab_set()
    try:
        parent.wait_window(root)
    finally:
        with contextlib.suppress(Exception):
            root.grab_release()
        if owns_parent:
            with contextlib.suppress(Exception):
                parent.destroy()
    print(
        "[GUI] Dialog returned: "
        f"cancelled={result.cancelled}, input_dir={result.input_dir or '<none>'}",
        flush=True,
    )
    return result


def _print_run_settings(
    *,
    input_dir: str,
    output_dir: str,
    csv_files: list[str],
    params_file: str,
    subject_mass_kg: float,
    subject_height_m: float,
    deadlift_mass_kg: float,
    fs_hz: float,
    filter_name: str,
    beta: float,
    gyro_units: str,
    reps: int | None,
    cutoff_hz: float,
    source: str,
) -> None:
    """Print the resolved run configuration + equivalent CLI command."""
    bar = "=" * 70
    print(bar)
    print(f"  Run configuration  ({source})")
    print(bar)
    print(f"Input dir:    {input_dir}")
    print(f"Output dir:   {output_dir}")
    print(f"Params file:  {params_file or '<none - defaults will apply>'}")
    print(f"Files found:  {len(csv_files)} CSV file(s)")
    print(f"Subject:      mass={subject_mass_kg} kg, height={subject_height_m} m")
    print(f"Barbell:      {deadlift_mass_kg} kg")
    print(f"Sample rate:  {fs_hz} Hz")
    print(f"Cutoff:       {cutoff_hz} Hz")
    print(f"Filters:      {_filter_selection_label(filter_name)} (beta={beta})")
    print(f"Gyro units:   {gyro_units}")
    print(f"Expected reps: {reps if reps is not None else 'auto-detect'}")
    print()
    print("Equivalent CLI command:")
    cmd = _format_cli_command(
        input_dir=input_dir,
        output_dir=output_dir,
        filter_name=filter_name,
        beta=beta,
        gyro_units=gyro_units,
        subject_mass_kg=subject_mass_kg,
        subject_height_m=subject_height_m,
        deadlift_mass_kg=deadlift_mass_kg,
        fs_hz=fs_hz,
        cutoff_hz=cutoff_hz,
        reps=reps,
        params_file=params_file,
        batch=True,
    )
    for line in cmd.splitlines():
        print(f"  {line}")
    print(bar, flush=True)


def _batch_process_directory(
    *,
    input_dir: str,
    output_dir: str,
    filter_name: str,
    beta: float,
    gyro_units: str,
    subject_mass_kg: float,
    subject_height_m: float,
    deadlift_mass_kg: float,
    fs_hz: float,
    reps: int | None,
    params_file: str,
    cutoff_hz: float,
    source: str,
) -> tuple[int, int, str]:
    """Process every ``*.csv`` in ``input_dir`` with per-file progress logs.

    Returns ``(processed, total, output_parent_dir)``. Used by both the GUI
    (:func:`main_gui`) and the CLI ``-d``/``--input-dir`` batch mode so the
    two paths print identical, debuggable feedback.
    """
    csv_files = sorted(
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        print(f"[ERROR] No CSV files found in: {input_dir}")
        return 0, 0, ""

    _print_run_settings(
        input_dir=input_dir,
        output_dir=output_dir,
        csv_files=csv_files,
        params_file=params_file,
        subject_mass_kg=subject_mass_kg,
        subject_height_m=subject_height_m,
        deadlift_mass_kg=deadlift_mass_kg,
        fs_hz=fs_hz,
        filter_name=filter_name,
        beta=beta,
        gyro_units=gyro_units,
        reps=reps,
        cutoff_hz=cutoff_hz,
        source=source,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_parent = os.path.join(output_dir, f"vaila_deadlift_imu_ahrs_{timestamp}")
    os.makedirs(output_parent, exist_ok=True)
    print(f"[BATCH] Results directory: {output_parent}")

    processed = 0
    total = len(csv_files)
    for idx, f in enumerate(csv_files, start=1):
        base = os.path.splitext(os.path.basename(f))[0]
        per_file = os.path.join(output_parent, base)
        os.makedirs(per_file, exist_ok=True)
        print(f"\n[BATCH] [{idx}/{total}] Processing: {f}")
        try:
            results = process_imu_file_filters(
                f,
                per_file,
                filter_name=filter_name,
                beta=beta,
                gyro_units=gyro_units,
                sample_rate=fs_hz,
                barbell_kg_override=deadlift_mass_kg,
                body_mass_kg_override=subject_mass_kg,
                body_height_m_override=subject_height_m,
                reps_override=reps,
                params_file=params_file or None,
                cutoff_hz=cutoff_hz,
            )
            if results and all(results.values()):
                processed += 1
        except Exception as e:  # keep going on per-file errors
            print(f"[BATCH] [{idx}/{total}] ERROR processing {f}: {e}")
            traceback.print_exc()

    print(f"\n[BATCH] Done: {processed}/{total} file(s) succeeded.")
    print(f"[BATCH] Results directory: {output_parent}", flush=True)
    return processed, total, output_parent


def main_gui() -> None:
    """Single unified Tkinter form for Deadlift IMU batch analysis."""
    _print_cli_help_banner()

    cfg = _show_deadlift_imu_dialog()
    if cfg.cancelled or not cfg.input_dir:
        print("[GUI] Aborted by user.")
        return

    if not os.path.isdir(cfg.input_dir):
        print(f"[GUI] ERROR: input dir is not a directory: {cfg.input_dir}")
        messagebox.showerror("Error", f"Input dir is not a directory:\n{cfg.input_dir}")
        return

    print(
        f"[GUI] Starting batch analysis: {cfg.input_dir} -> {cfg.output_dir or cfg.input_dir}",
        flush=True,
    )
    processed, total, output_parent = _batch_process_directory(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir or cfg.input_dir,
        filter_name=cfg.filter_name,
        beta=cfg.beta,
        gyro_units=cfg.gyro_units,
        subject_mass_kg=cfg.subject_mass_kg,
        subject_height_m=cfg.subject_height_m,
        deadlift_mass_kg=cfg.deadlift_mass_kg,
        fs_hz=cfg.fs_hz,
        reps=cfg.reps,
        cutoff_hz=cfg.cutoff_hz,
        params_file=cfg.params_file,
        source="GUI mode",
    )
    if total == 0:
        messagebox.showerror("Error", "No CSV files found in selected directory.")
        return

    messagebox.showinfo(
        "Analysis Complete",
        f"Processed {processed}/{total} file(s).\nResults directory:\n{output_parent}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "IMU-only Deadlift / RDL analysis with Madgwick or Mahony AHRS sensor fusion. "
            "Run with no arguments to open the Tkinter form; use -i for a single file or "
            "-d for batch processing of an entire directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  Single file (default runs Madgwick + Mahony):\n'
            '    uv run python vaila/vaila_deadlift_imu.py -i tests/Deadlift/imu/deadlift_imu.csv\n'
            '\n'
            '  Single file with explicit params .txt:\n'
            '    uv run python vaila/vaila_deadlift_imu.py \\\n'
            '      -i tests/Deadlift/imu/deadlift_imu.csv \\\n'
            '      -p tests/Deadlift/imu/deadlift_parameters.txt\n'
            '\n'
            '  Whole dir with explicit params .txt (default runs both filters):\n'
            '    uv run python vaila/vaila_deadlift_imu.py \\\n'
            '      -d tests/Deadlift/imu \\\n'
            '      -p tests/Deadlift/imu/deadlift_parameters.txt \\\n'
            '      -o /tmp/dlift_out\n'
            '\n'
            '  Force GUI:\n'
            '    uv run python vaila/vaila_deadlift_imu.py --gui\n'
            '\n'
            'deadlift_parameters.txt CSV-header format:\n'
            '  subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz\n'
            '  75.0,1.77,20.0,21.0\n'
        ),
    )
    p.add_argument("-i", "--input", type=str, help="Path to a single input IMU CSV file")
    p.add_argument(
        "-d",
        "--input-dir",
        type=str,
        help="Process every *.csv in this directory (batch mode, mirrors the GUI)",
    )
    p.add_argument(
        "-o", "--output", type=str, help="Destination directory for plots, CSVs and HTML report"
    )
    p.add_argument(
        "--filter",
        choices=["both", "madgwick", "mahony"],
        default="both",
        help="AHRS filter(s) to run. Default: both; use madgwick or mahony to restrict.",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Madgwick proportional gain (used by the Madgwick branch when --filter both or madgwick)",
    )
    p.add_argument(
        "--gyro-units",
        choices=["deg", "rad"],
        default="deg",
        help="Units of the gyroscope columns in the CSV",
    )
    p.add_argument(
        "-p",
        "--params-file",
        type=str,
        help=(
            "Path to parameters .txt/.csv file, for example "
            "'-p tests/Deadlift/imu/deadlift_parameters.txt'. Supports one of:\n"
            "  - CSV header format: "
            "'subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz' then a values row, "
            "or\n  - legacy 'key,value' lines (e.g. 'weight,20'). "
            "Auto-detected beside the CSV when omitted."
        ),
    )
    p.add_argument("--fps", type=float, help="Override IMU sample rate in Hz (overrides params)")
    p.add_argument(
        "--cutoff",
        type=float,
        default=DEFAULT_IMU_CUTOFF_HZ,
        help="Butterworth low-pass cutoff in Hz for vertical acceleration metrics (default: 4.0)",
    )
    p.add_argument("--weight", type=float, help="Override barbell weight in kg (overrides params)")
    p.add_argument("--mass", type=float, help="Override athlete body mass in kg (overrides params)")
    p.add_argument(
        "--height",
        type=float,
        help="Override athlete body height in meters (overrides params; informational)",
    )
    p.add_argument(
        "--reps",
        type=int,
        help="Force the expected repetition count (trims lead-in/trailing cycles)",
    )
    p.add_argument("--gui", action="store_true", help="Force the Tkinter single-form dialog")
    return p


def main() -> None:
    args = _build_argparser().parse_args()

    # No data path and not explicitly headless -> open the GUI.
    if args.gui or (not args.input and not args.input_dir):
        main_gui()
        return

    # Batch mode (-d / --input-dir): mirrors the GUI on the terminal.
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[ERROR] Input dir is not a directory: {args.input_dir}")
            return
        out_dir = args.output or args.input_dir
        # Cast defaults to float explicitly to satisfy the type checker - the
        # DEFAULT_SETTINGS dict mixes floats and strings, so we have to be
        # explicit about which keys yield numerics.
        default_mass = float(DEFAULT_SETTINGS["subject_mass_kg"])  # type: ignore[arg-type]
        default_height = float(DEFAULT_SETTINGS["subject_height_m"])  # type: ignore[arg-type]
        default_bar = float(DEFAULT_SETTINGS["deadlift_mass_kg"])  # type: ignore[arg-type]
        default_fs = float(DEFAULT_SETTINGS["fs_hz"])  # type: ignore[arg-type]
        _batch_process_directory(
            input_dir=args.input_dir,
            output_dir=out_dir,
            filter_name=args.filter,
            beta=args.beta,
            gyro_units=args.gyro_units,
            subject_mass_kg=args.mass if args.mass is not None else default_mass,
            subject_height_m=args.height if args.height is not None else default_height,
            deadlift_mass_kg=args.weight if args.weight is not None else default_bar,
            fs_hz=args.fps if args.fps is not None else default_fs,
            reps=args.reps,
            cutoff_hz=args.cutoff,
            params_file=args.params_file or "",
            source="CLI batch mode (-d)",
        )
        return

    # Single-file mode (-i / --input).
    out = args.output or os.path.dirname(os.path.abspath(args.input))
    process_imu_file_filters(
        args.input,
        out,
        filter_name=args.filter,
        beta=args.beta,
        gyro_units=args.gyro_units,
        sample_rate=args.fps,
        barbell_kg_override=args.weight,
        body_mass_kg_override=args.mass,
        body_height_m_override=args.height,
        reps_override=args.reps,
        params_file=args.params_file,
        cutoff_hz=args.cutoff,
    )


if __name__ == "__main__":
    main()
