"""Tests for reid_yolotrack CSV naming and discovery."""

from __future__ import annotations

from pathlib import Path

from vaila.reid_yolotrack import iter_tracking_csv_paths, parse_tracking_csv_filename


def test_parse_person_id_zero_padded() -> None:
    parsed = parse_tracking_csv_filename("person_id_01.csv")
    assert parsed == ("person", 1)


def test_parse_person_id_legacy() -> None:
    parsed = parse_tracking_csv_filename("player_id3.csv")
    assert parsed == ("player", 3)


def test_parse_skips_aggregate_csvs() -> None:
    assert parse_tracking_csv_filename("all_id_detection.csv") is None
    assert parse_tracking_csv_filename("person_id_01_pose.csv") is None
    assert parse_tracking_csv_filename("yolo_reid_links.csv") is None


def test_iter_tracking_csv_paths(tmp_path: Path) -> None:
    (tmp_path / "person_id_01.csv").write_text("Frame,X_min\n0,1\n", encoding="utf-8")
    (tmp_path / "all_id_detection.csv").write_text("Frame\n", encoding="utf-8")
    hits = iter_tracking_csv_paths(tmp_path)
    assert len(hits) == 1
    assert hits[0].name == "person_id_01.csv"
