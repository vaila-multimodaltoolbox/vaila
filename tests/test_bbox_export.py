import os
import json
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from vaila.getpixelvideo import parse_contours_json, do_export_bbox_coords

def test_parse_contours_json():
    # Create mock contours json payload
    payload = {
        "schema": "vaila_sam_contours_v1",
        "video": "test_video.mp4",
        "width": 1280,
        "height": 720,
        "fps": 30.0,
        "n_frames": 10,
        "object_ids": [1, 2],
        "frames": [
            {
                "frame": 0,
                "objects": [
                    {
                        "obj_id": 1,
                        "bbox_xywh_px": [10, 20, 100, 200],
                        "score": 0.95
                    },
                    {
                        "obj_id": 2,
                        "polygons": [[[30, 40], [30, 90], [80, 90], [80, 40]]],
                        "score": 0.88
                    }
                ]
            },
            {
                "frame": 1,
                "objects": [
                    {
                        "obj_id": 1,
                        "bbox_xywh_px": [12, 22, 98, 198],
                        "score": 0.96
                    }
                ]
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "sam_contours.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        bboxes = parse_contours_json(json_path)
        assert len(bboxes) == 3

        # First box (obj_id=1, frame=0)
        b0 = bboxes[0]
        assert b0["frame"] == 0
        assert b0["obj_id"] == 1
        assert b0["x1"] == 10
        assert b0["y1"] == 20
        assert b0["x2"] == 110
        assert b0["y2"] == 220
        assert b0["score"] == 0.95

        # Second box (obj_id=2, frame=0, calculated from polygons)
        b1 = bboxes[1]
        assert b1["frame"] == 0
        assert b1["obj_id"] == 2
        assert b1["x1"] == 30
        assert b1["y1"] == 40
        assert b1["x2"] == 80
        assert b1["y2"] == 90
        assert b1["score"] == 0.88

        # Third box (obj_id=1, frame=1)
        b2 = bboxes[2]
        assert b2["frame"] == 1
        assert b2["obj_id"] == 1
        assert b2["x1"] == 12
        assert b2["y1"] == 22
        assert b2["x2"] == 110
        assert b2["y2"] == 220


def test_do_export_bbox_coords():
    payload = {
        "schema": "vaila_sam_contours_v1",
        "video": "test_video.mp4",
        "width": 1280,
        "height": 720,
        "fps": 30.0,
        "n_frames": 3,
        "object_ids": [1, 2],
        "frames": [
            {
                "frame": 0,
                "objects": [
                    {
                        "obj_id": 1,
                        "bbox_xywh_px": [10, 20, 100, 200],
                        "score": 0.95
                    },
                    {
                        "obj_id": 2,
                        "bbox_xywh_px": [30, 40, 50, 50],
                        "score": 0.88
                    }
                ]
            },
            {
                "frame": 2,
                "objects": [
                    {
                        "obj_id": 1,
                        "bbox_xywh_px": [12, 22, 98, 198],
                        "score": 0.96
                    }
                ]
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "sam_contours.json"
        json_path.write_text(json.dumps(payload), encoding="utf-8")

        res = do_export_bbox_coords(json_path)
        assert len(res) == 5

        for path_str in res:
            assert os.path.exists(path_str)

        # Check content of bottom coordinates file
        bottom_csv_path = Path(tmpdir) / "sam_contours_bottom.csv"
        df = pd.read_csv(bottom_csv_path)

        # Columns should be frame, p0_x, p0_y, p1_x, p1_y
        assert list(df.columns) == ["frame", "p0_x", "p0_y", "p1_x", "p1_y"]
        assert len(df) == 3

        # Frame 0:
        # obj_id 1 (p0): x_center = (10 + 110) / 2 = 60, y_max = 220
        # obj_id 2 (p1): x_center = (30 + 80) / 2 = 55, y_max = 90
        assert df.loc[0, "frame"] == 0
        assert df.loc[0, "p0_x"] == 60
        assert df.loc[0, "p0_y"] == 220
        assert df.loc[0, "p1_x"] == 55
        assert df.loc[0, "p1_y"] == 90

        # Frame 1: empty coordinates for both
        assert df.loc[1, "frame"] == 1
        assert pd.isna(df.loc[1, "p0_x"])
        assert pd.isna(df.loc[1, "p1_x"])

        # Frame 2:
        # obj_id 1 (p0): x_center = (12 + 110) / 2 = 61, y_max = 220
        # obj_id 2 (p1): empty
        assert df.loc[2, "frame"] == 2
        assert df.loc[2, "p0_x"] == 61
        assert df.loc[2, "p0_y"] == 220
        assert pd.isna(df.loc[2, "p1_x"])
