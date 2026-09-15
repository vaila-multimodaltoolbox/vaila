#!/usr/bin/env python3
"""Compatibility entry for the fixed-scene video stabilizer.

Version: 0.4.1
Update Date: 15 September 2026
License: AGPL-3.0-or-later

The production implementation and GUI live in video_stabilizer.py.
Legacy --estimator global and --reference-frame N are accepted.
Use --output-dir for a complete run (video, CSVs, report and diagnostics).
"""

import sys

if __package__:
    from .video_stabilizer import main, run_video_stabilizer, run_video_stabilizer_gui
else:
    from video_stabilizer import main, run_video_stabilizer, run_video_stabilizer_gui

__all__ = ["main", "run_video_stabilizer", "run_video_stabilizer_gui"]


def legacy_main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(args[:-1]):
        if arg == "--estimator" and args[i + 1] == "global":
            args[i + 1] = "robust-lsq"
        if arg == "--reference-frame":
            args[i : i + 2] = ["--reference", "frame:" + args[i + 1]]
    return main(args)


if __name__ == "__main__":
    raise SystemExit(legacy_main())
