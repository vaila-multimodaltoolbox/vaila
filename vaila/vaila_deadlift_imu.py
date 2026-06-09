"""
===============================================================================
vaila_deadlift_imu.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 09 June 2026
Update Date: 09 June 2026
Version: 0.3.48
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
* GUI: ``uv run python vaila/vaila_deadlift_imu.py``
* CLI: ``uv run python vaila/vaila_deadlift_imu.py -i deadlift_imu.csv``
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import math
import os
import tkinter as tk
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
VAILA_VERSION = "0.3.48"


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


def detect_reps_from_vertical_acc(
    a_vert: np.ndarray,
    fps: float,
    min_rep_s: float = 1.5,
    prominence_frac: float = 0.10,
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
    cutoff_metric = min(5.0, nyq * 0.9)
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

    # 1b. Didactic quaternion figure (q components, |q|, angle, axis)
    # The unit quaternion q = [q0, q1, q2, q3] = [cos(theta/2), sin(theta/2) * n]
    # encodes a rotation by angle theta around unit axis n. Plotting all four
    # components plus the derived (theta, n) makes the abstract algebra concrete
    # for users who only know Euler angles.
    quat = fusion.quaternion  # (N, 4)
    q_norm = np.linalg.norm(quat, axis=1)
    q0_clip = np.clip(np.abs(quat[:, 0]), 0.0, 1.0)
    theta_deg = np.degrees(2.0 * np.arccos(q0_clip))  # rotation angle [0, 360]
    sin_half = np.sqrt(np.clip(1.0 - q0_clip**2, 1e-12, 1.0))
    axis = np.zeros_like(quat[:, 1:])  # (N, 3) unit rotation axis
    axis[:, 0] = quat[:, 1] / sin_half
    axis[:, 1] = quat[:, 2] / sin_half
    axis[:, 2] = quat[:, 3] / sin_half

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(
        time_axis, quat[:, 0], color="#1a365d", linewidth=1.0, label=r"$q_0 = w = \cos(\theta/2)$"
    )
    axes[0].plot(
        time_axis,
        quat[:, 1],
        color="#c0392b",
        linewidth=1.0,
        label=r"$q_1 = x = n_x \sin(\theta/2)$",
    )
    axes[0].plot(
        time_axis,
        quat[:, 2],
        color="#27ae60",
        linewidth=1.0,
        label=r"$q_2 = y = n_y \sin(\theta/2)$",
    )
    axes[0].plot(
        time_axis,
        quat[:, 3],
        color="#8e44ad",
        linewidth=1.0,
        label=r"$q_3 = z = n_z \sin(\theta/2)$",
    )
    axes[0].axhline(0.0, color="gray", linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("Quaternion components")
    axes[0].set_title(
        rf"Quaternion orientation $q = [w, x, y, z]$ - {base_name} "
        rf"(filter: {filter_name.title()} AHRS)"
    )
    axes[0].legend(fontsize=8, ncol=4, loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        time_axis,
        q_norm,
        color="#2c5282",
        linewidth=1.0,
        label=r"$|q| = \sqrt{q_0^2 + q_1^2 + q_2^2 + q_3^2}$",
    )
    axes[1].axhline(
        1.0, color="red", linestyle="--", alpha=0.5, label=r"Unit norm $|q| = 1$ (rigid rotation)"
    )
    norm_dev = float(np.max(np.abs(q_norm - 1.0)))
    axes[1].set_ylabel("Quaternion norm")
    axes[1].set_title(
        rf"Norm check: $\max\,|\,|q| - 1\,| = {norm_dev:.2e}$ "
        rf"(should stay numerically at 1)"
    )
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time_axis, theta_deg, color="#b7791f", linewidth=1.0, label=r"$\theta = 2\,\arccos(|q_0|)$"
    )
    axes[2].set_ylabel("Rotation angle θ (deg)")
    axes[2].set_title(r"Total rotation angle from sensor frame to Earth frame")
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_axis, axis[:, 0], color="#c0392b", linewidth=0.9, label=r"$n_x$ (axis X)")
    axes[3].plot(time_axis, axis[:, 1], color="#27ae60", linewidth=0.9, label=r"$n_y$ (axis Y)")
    axes[3].plot(time_axis, axis[:, 2], color="#8e44ad", linewidth=0.9, label=r"$n_z$ (axis Z)")
    axes[3].axhline(0.0, color="gray", linewidth=0.5, alpha=0.5)
    axes[3].set_ylabel("Rotation axis n")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_title(
        r"Instantaneous rotation axis $\mathbf{n} = (n_x, n_y, n_z)$ "
        r"with $q_{1..3} = \mathbf{n}\sin(\theta/2)$"
    )
    axes[3].legend(fontsize=8, ncol=3, loc="upper right")
    axes[3].grid(True, alpha=0.3)

    for rep in reps:
        for ax in axes:
            ax.axvline(rep["peak_frame"] / fps, color="red", linestyle=":", alpha=0.35)

    fig.suptitle(
        "Quaternion-based 3D orientation (didactic view) - "
        r"$q = \cos(\theta/2) + \sin(\theta/2)\,(n_x i + n_y j + n_z k)$",
        fontsize=12,
        y=1.005,
    )
    fig.tight_layout()
    p1b = os.path.join(output_dir, f"{base_name}_imu_quaternion_didactic_{ts}.png")
    fig.savefig(p1b, dpi=150)
    plt.close(fig)
    out.append(p1b)

    # 1c. Quaternion polar view: rotation axis on the unit sphere (top-down)
    # Project the unit axis n(t) onto the XY plane and colour by time. This is a
    # compact, didactic "where is the rotation axis pointing right now?" plot.
    fig, ax = plt.subplots(figsize=(7, 7))
    nx = axis[:, 0]
    ny = axis[:, 1]
    nz = axis[:, 2]
    colors = np.arange(len(nx))
    sc = ax.scatter(nx, ny, c=colors, cmap="viridis", s=6, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Sample index (time progression)")
    circle = plt.Circle((0.0, 0.0), 1.0, color="gray", fill=False, linestyle="--", linewidth=1.0)
    ax.add_patch(circle)
    ax.axhline(0.0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$n_x$  (X component of rotation axis)")
    ax.set_ylabel(r"$n_y$  (Y component of rotation axis)")
    ax.set_title(
        f"Rotation axis n(t) on the unit sphere (XY view) - {base_name}\n"
        rf"Z (out-of-plane) component shown as size: mean $n_z$ = {float(np.mean(nz)):.2f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p1c = os.path.join(output_dir, f"{base_name}_imu_quaternion_axis_xy_{ts}.png")
    fig.savefig(p1c, dpi=150)
    plt.close(fig)
    out.append(p1c)

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

    images_html = "".join(
        f'<div class="img-container"><img src="{os.path.basename(p)}"></div>\n' for p in plot_files
    )

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
        <h2 style="margin-top:0; color:#155e75;">Why quaternions? (didactic note)</h2>
        <p>A unit quaternion
        <code>q = [q<sub>0</sub>, q<sub>1</sub>, q<sub>2</sub>, q<sub>3</sub>] =
        [cos(θ/2), n<sub>x</sub> sin(θ/2), n<sub>y</sub> sin(θ/2),
        n<sub>z</sub> sin(θ/2)]</code> represents a rotation by angle
        <code>θ</code> around the unit axis
        <code>n = (n<sub>x</sub>, n<sub>y</sub>, n<sub>z</sub>)</code>. The
        figures above visualise:</p>
        <ul>
            <li><strong>Quaternion components (q<sub>0..3</sub>):</strong> raw
                output of the AHRS filter, sample by sample.</li>
            <li><strong>Norm |q| ≈ 1:</strong> a numerical sanity check that the
                tracker is producing a rigid rotation (no scaling, no shear).</li>
            <li><strong>Rotation angle θ(t):</strong> the total angular distance
                between the sensor frame and the Earth frame at each instant.</li>
            <li><strong>Rotation axis n(t):</strong> the instantaneous direction
                around which the barbell is rotating; the unit-sphere XY view
                makes its trajectory visible at a glance.</li>
        </ul>
        <p>Unlike Euler angles, quaternions are <em>singularity-free</em>: no
        gimbal lock, no discontinuous unwrapping, and no ambiguous axis order.
        This is precisely why every modern IMU/AHRS chip, every game engine and
        every 3D animation system uses quaternions internally - and why the
        AHRS pipeline of this report stores orientation as
        <code>q = [w, x, y, z]</code> rather than (roll, pitch, yaw).</p>
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
        lines.append(f"![{rel}]({rel})")
        lines.append("")

    lines.append("## Why quaternions? (didactic note)")
    lines.append("")
    lines.append(
        "A unit quaternion `q = [q0, q1, q2, q3] = [cos(θ/2), nx sin(θ/2), "
        "ny sin(θ/2), nz sin(θ/2)]` represents a rotation by angle `θ` around the "
        "unit axis `n = (nx, ny, nz)`. The figures above visualise:"
    )
    lines.append("")
    lines.append(
        "- **Quaternion components q0..q3** — raw output of the AHRS filter, sample by sample."
    )
    lines.append(
        "- **Norm |q| ≈ 1** — numerical sanity check that the tracker is producing "
        "a rigid rotation (no scaling, no shear)."
    )
    lines.append(
        "- **Rotation angle θ(t)** — total angular distance between sensor frame "
        "and Earth frame at each instant."
    )
    lines.append(
        "- **Rotation axis n(t)** — instantaneous direction around which the "
        "barbell is rotating; the unit-sphere XY view makes its trajectory visible "
        "at a glance."
    )
    lines.append("")
    lines.append(
        "Unlike Euler angles, quaternions are *singularity-free*: no gimbal lock, "
        "no discontinuous unwrapping, no ambiguous axis order. That is why every "
        "modern IMU/AHRS chip, every game engine and every 3D animation system "
        "uses quaternions internally — and why the AHRS pipeline of this report "
        "stores orientation as `q = [w, x, y, z]` rather than (roll, pitch, yaw)."
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

    src_label = params.get("__source__", "<defaults>") if params else "<defaults>"
    print(
        f"[IMU-AHRS] {base_name}: filter={filter_name}, fps={fps:.2f} Hz, "
        f"barbell={barbell_kg:.1f} kg, body_mass={body_mass_kg}, "
        f"height={body_height_m} m, use_total_mass={use_total_mass} "
        f"(params: {src_label})"
    )

    fusion = run_ahrs_fusion(df, fps=fps, filter_name=filter_name, beta=beta, gyro_units=gyro_units)
    print(
        f"[IMU-AHRS] |g| sensor={fusion.g_mag:.3f} m/s²  g_earth={fusion.g_earth.round(3).tolist()}"
    )

    reps, velocity, disp, a_filt = detect_reps_from_vertical_acc(fusion.acc_vert_ms2, fps=fps)
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
            "acc_earth_x": fusion.acc_earth_ms2[:, 0],
            "acc_earth_y": fusion.acc_earth_ms2[:, 1],
            "acc_earth_z": fusion.acc_earth_ms2[:, 2],
            "acc_linear_x": fusion.acc_linear_ms2[:, 0],
            "acc_linear_y": fusion.acc_linear_ms2[:, 1],
            "acc_linear_z": fusion.acc_linear_ms2[:, 2],
            "a_vert_linear_ms2": a_filt,
            "velocity_ms": velocity,
            "displacement_m": disp,
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
    filter_name: str = "madgwick"
    beta: float = 0.1
    gyro_units: str = "deg"
    reps: int | None = None
    cancelled: bool = True


def _show_deadlift_imu_dialog() -> _GuiResult:
    """Single-window Tkinter form for Deadlift IMU analysis.

    Returns a :class:`_GuiResult` with ``cancelled=True`` if the user closed the
    dialog or clicked "Cancel". All inputs are validated; invalid numeric
    fields are rejected with a ``messagebox.showerror``.
    """
    result = _GuiResult()

    root = Tk()
    root.title("vailá - Deadlift IMU (AHRS) - Single-form input")
    root.attributes("-topmost", True)
    with contextlib.suppress(Exception):  # non-graphical envs
        root.geometry("640x520")

    PX, PY = 8, 4

    header = tk.Label(
        root,
        text="Deadlift IMU (AHRS) - fill the form once, then click Run.",
        font=("TkDefaultFont", 11, "bold"),
        justify="left",
    )
    header.grid(row=0, column=0, columnspan=3, sticky="w", padx=PX, pady=PY)

    # ---- Directory pickers ---------------------------------------------------
    in_var = tk.StringVar()
    out_var = tk.StringVar()
    params_var = tk.StringVar()

    def _pick_input_dir() -> None:
        d = filedialog.askdirectory(
            title="Select directory containing Deadlift IMU CSV files (acc + gyr)",
            parent=root,
        )
        if d:
            in_var.set(d)
            if not out_var.get():
                out_var.set(d)
            # Auto-detect a sibling params file
            candidate = Path(d) / "deadlift_parameters.txt"
            if candidate.exists() and not params_var.get():
                params_var.set(str(candidate))
                _load_params_into_form(str(candidate))

    def _pick_output_dir() -> None:
        d = filedialog.askdirectory(title="Select output directory", parent=root)
        if d:
            out_var.set(d)

    def _pick_params_file() -> None:
        f = filedialog.askopenfilename(
            title="Select deadlift_parameters.txt",
            parent=root,
            filetypes=[("Parameters", "*.txt"), ("All files", "*.*")],
        )
        if f:
            params_var.set(f)
            _load_params_into_form(f)

    def _load_params_into_form(path: str) -> None:
        params = _parse_parameters_file(Path(path))
        if not params:
            return
        with contextlib.suppress(ValueError, TypeError):
            if "subject_mass_kg" in params:
                mass_var.set(f"{float(params['subject_mass_kg']):.2f}")
        with contextlib.suppress(ValueError, TypeError):
            if "subject_height_m" in params:
                height_var.set(f"{float(params['subject_height_m']):.2f}")
        with contextlib.suppress(ValueError, TypeError):
            if "deadlift_mass_kg" in params:
                bar_var.set(f"{float(params['deadlift_mass_kg']):.2f}")
            elif "weight" in params:
                bar_var.set(f"{float(params['weight']):.2f}")
        with contextlib.suppress(ValueError, TypeError):
            if "fs_hz" in params:
                fs_var.set(f"{float(params['fs_hz']):.2f}")
        with contextlib.suppress(ValueError, TypeError):
            if "real_repetition_count" in params:
                reps_var.set(str(int(float(params["real_repetition_count"]))))

    tk.Label(root, text="Input dir (CSVs):").grid(row=1, column=0, sticky="e", padx=PX, pady=PY)
    tk.Entry(root, textvariable=in_var, width=46).grid(row=1, column=1, padx=PX, pady=PY)
    tk.Button(root, text="Browse...", command=_pick_input_dir).grid(
        row=1, column=2, padx=PX, pady=PY
    )

    tk.Label(root, text="Output dir:").grid(row=2, column=0, sticky="e", padx=PX, pady=PY)
    tk.Entry(root, textvariable=out_var, width=46).grid(row=2, column=1, padx=PX, pady=PY)
    tk.Button(root, text="Browse...", command=_pick_output_dir).grid(
        row=2, column=2, padx=PX, pady=PY
    )

    tk.Label(root, text="Params file (optional):").grid(
        row=3, column=0, sticky="e", padx=PX, pady=PY
    )
    tk.Entry(root, textvariable=params_var, width=46).grid(row=3, column=1, padx=PX, pady=PY)
    tk.Button(root, text="Load .txt...", command=_pick_params_file).grid(
        row=3, column=2, padx=PX, pady=PY
    )

    ttk.Separator(root, orient="horizontal").grid(
        row=4, column=0, columnspan=3, sticky="ew", padx=8, pady=8
    )

    # ---- Numeric subject / bar / fps parameters -----------------------------
    mass_var = tk.StringVar(value="75.0")
    height_var = tk.StringVar(value="1.75")
    bar_var = tk.StringVar(value="20.0")
    fs_var = tk.StringVar(value="25.0")
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

    _row("Subject mass (kg):", mass_var, 5, hint="e.g. 75.0")
    _row("Subject height (m):", height_var, 6, hint="e.g. 1.75 (informational)")
    _row("Deadlift bar mass (kg):", bar_var, 7, hint="barbell weight, e.g. 20.0")
    _row("Sample rate (Hz):", fs_var, 8, hint="IMU sampling frequency, e.g. 25.0")
    _row("Expected reps (optional):", reps_var, 9, hint="leave blank for auto-detect")

    ttk.Separator(root, orient="horizontal").grid(
        row=10, column=0, columnspan=3, sticky="ew", padx=8, pady=8
    )

    # ---- AHRS filter settings ----------------------------------------------
    filter_var = tk.StringVar(value="madgwick")
    beta_var = tk.StringVar(value="0.1")
    gyro_units_var = tk.StringVar(value="deg")

    tk.Label(root, text="AHRS filter:").grid(row=11, column=0, sticky="e", padx=PX, pady=PY)
    ttk.Combobox(
        root,
        textvariable=filter_var,
        values=["madgwick", "mahony"],
        width=12,
        state="readonly",
    ).grid(row=11, column=1, sticky="w", padx=PX, pady=PY)
    tk.Label(root, text="Madgwick gradient gain", fg="#666").grid(
        row=11, column=2, sticky="w", padx=PX, pady=PY
    )

    tk.Label(root, text="Beta (Madgwick):").grid(row=12, column=0, sticky="e", padx=PX, pady=PY)
    tk.Entry(root, textvariable=beta_var, width=14).grid(
        row=12, column=1, sticky="w", padx=PX, pady=PY
    )
    tk.Label(root, text="default 0.1", fg="#666").grid(
        row=12, column=2, sticky="w", padx=PX, pady=PY
    )

    tk.Label(root, text="Gyro units:").grid(row=13, column=0, sticky="e", padx=PX, pady=PY)
    ttk.Combobox(
        root,
        textvariable=gyro_units_var,
        values=["deg", "rad"],
        width=12,
        state="readonly",
    ).grid(row=13, column=1, sticky="w", padx=PX, pady=PY)
    tk.Label(root, text="deg/s (default) or rad/s", fg="#666").grid(
        row=13, column=2, sticky="w", padx=PX, pady=PY
    )

    # ---- Action buttons -----------------------------------------------------
    btn_frame = tk.Frame(root)
    btn_frame.grid(row=14, column=0, columnspan=3, pady=14)

    def _on_run() -> None:
        if not in_var.get():
            messagebox.showerror("Error", "Input directory is required.", parent=root)
            return
        try:
            result.subject_mass_kg = float(mass_var.get())
            result.subject_height_m = float(height_var.get())
            result.deadlift_mass_kg = float(bar_var.get())
            result.fs_hz = float(fs_var.get())
            result.beta = float(beta_var.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid numeric field: {e}", parent=root)
            return
        if reps_var.get().strip():
            try:
                result.reps = int(reps_var.get())
            except ValueError:
                messagebox.showerror("Error", "Expected reps must be an integer.", parent=root)
                return
        result.input_dir = in_var.get()
        result.output_dir = out_var.get() or in_var.get()
        result.params_file = params_var.get()
        result.filter_name = filter_var.get()
        result.gyro_units = gyro_units_var.get()
        result.cancelled = False
        root.destroy()

    def _on_cancel() -> None:
        result.cancelled = True
        root.destroy()

    tk.Button(
        btn_frame, text="Run analysis", command=_on_run, width=16, bg="#2563eb", fg="white"
    ).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", command=_on_cancel, width=10).pack(side="left", padx=6)

    root.protocol("WM_DELETE_WINDOW", _on_cancel)
    root.mainloop()
    return result


def main_gui() -> None:
    """Single unified Tkinter form for Deadlift IMU batch analysis."""
    cfg = _show_deadlift_imu_dialog()
    if cfg.cancelled or not cfg.input_dir:
        return

    target_dir = cfg.input_dir
    csv_files = [
        os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith(".csv")
    ]
    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in selected directory.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_parent = os.path.join(cfg.output_dir, f"vaila_deadlift_imu_ahrs_{timestamp}")
    os.makedirs(output_parent, exist_ok=True)

    processed = 0
    for f in csv_files:
        base = os.path.splitext(os.path.basename(f))[0]
        per_file = os.path.join(output_parent, base)
        os.makedirs(per_file, exist_ok=True)
        try:
            process_imu_file(
                f,
                per_file,
                filter_name=cfg.filter_name,
                beta=cfg.beta,
                gyro_units=cfg.gyro_units,
                sample_rate=cfg.fs_hz,
                barbell_kg_override=cfg.deadlift_mass_kg,
                body_mass_kg_override=cfg.subject_mass_kg,
                body_height_m_override=cfg.subject_height_m,
                reps_override=cfg.reps,
                params_file=cfg.params_file or None,
            )
            processed += 1
        except Exception as e:  # keep going on per-file errors
            print(f"[ERROR] Failed to process {f}: {e}")

    messagebox.showinfo(
        "Analysis Complete",
        f"Processed {processed}/{len(csv_files)} file(s).\nResults directory:\n{output_parent}",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("IMU-only Deadlift / RDL analysis with Madgwick or Mahony AHRS sensor fusion.")
    )
    p.add_argument("-i", "--input", type=str, help="Path to input IMU CSV file")
    p.add_argument(
        "-o", "--output", type=str, help="Destination directory for plots, CSVs and HTML report"
    )
    p.add_argument(
        "--filter", choices=["madgwick", "mahony"], default="madgwick", help="AHRS filter to use"
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Madgwick proportional gain (only used with --filter madgwick)",
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
            "Path to a deadlift_parameters.txt with one of:\n"
            "  - CSV header format: "
            "'subject_mass_kg,subject_height_m,deadlift_mass_kg,fs_hz' then a values row, "
            "or\n  - legacy 'key,value' lines (e.g. 'weight,20'). "
            "Auto-detected beside the CSV when omitted."
        ),
    )
    p.add_argument("--fps", type=float, help="Override IMU sample rate in Hz (overrides params)")
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
    if args.gui or not args.input:
        main_gui()
        return
    out = args.output or os.path.dirname(os.path.abspath(args.input))
    process_imu_file(
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
    )


if __name__ == "__main__":
    main()
