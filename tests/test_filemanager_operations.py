"""File Manager safety and CLI regression tests.

Version: 0.3.137
Update Date: 11 September 2026
"""

import os
import shlex
import subprocess
import sys
import threading

import pytest

from vaila import filemanager as fm
from vaila.task_feedback import Feedback, WorkerTask, command_text


def write(root, name, value="sample"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)
    return path


def test_copy_filters_nested_destination_and_collisions(tmp_path):
    src = tmp_path / "source files"
    write(src, "a/trial.csv", "first")
    write(src, "b/trial.csv", "second")
    write(src, "trial.CSV")
    write(src, "other.csv")
    dest = src / "output"
    write(dest, "trial.csv", "old output")
    plan = fm.build_plan("copy", src, dest, extension=".csv", patterns=["trial"])
    before = sorted(src.rglob("*"))
    assert fm.execute_plan(plan, dry_run=True).exit_code == 0
    assert sorted(src.rglob("*")) == before
    assert len(plan.targets) == 2
    result = fm.execute_plan(plan)
    assert result.completed == 2 and not result.errors
    assert {p.destination.read_text() for p in plan.targets} == {"first", "second"}
    assert len({p.destination.name for p in plan.targets}) == 2
    assert (dest / "trial.csv").read_text() == "old output"


def test_copy_literal_patterns_move_first_match(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    original = write(src, "trial.csv")
    assert not fm.build_plan("copy", src, dest, patterns=["*trial*"]).targets
    plan = fm.build_plan("copy", src, dest, patterns=["trial", "trial.csv"])
    assert len(plan.targets) == 2
    plan = fm.build_plan("move", src, dest, patterns=["trial", "trial.csv"])
    assert len(plan.targets) == 1
    assert fm.execute_plan(plan).completed == 1
    assert not original.exists()


def test_rename_collision_is_partial_failure_and_no_overwrite(tmp_path):
    write(tmp_path, "old.csv", "original")
    write(tmp_path, "new.csv", "keep")
    write(tmp_path, "old2.csv", "move me")
    plan = fm.build_plan("rename", tmp_path, text="old", replacement="new", extension=".csv")
    result = fm.execute_plan(plan)
    assert result.exit_code == 1 and result.completed == 1 and len(result.errors) == 1
    assert (tmp_path / "new.csv").read_text() == "keep"
    assert (tmp_path / "old.csv").read_text() == "original"
    assert (tmp_path / "new2.csv").read_text() == "move me"


def test_destination_created_after_preview_is_not_overwritten(tmp_path):
    src = write(tmp_path / "src", "a.csv")
    plan = fm.build_plan("export", src, tmp_path / "dest")
    write(tmp_path / "dest", "a.csv", "arrived after preview")
    result = fm.execute_plan(plan)
    assert result.exit_code == 1
    assert plan.targets[0].destination.read_text() == "arrived after preview"


def test_rename_cannot_escape_source(tmp_path):
    write(tmp_path, "old.csv")
    with pytest.raises(ValueError):
        fm.build_plan("rename", tmp_path, text="old", replacement="../outside")


def test_normalize_children_before_parents_and_collisions(tmp_path):
    write(tmp_path, "Á Folder/Some-Filé.CSV", "data")
    write(tmp_path, "á.csv", "accent")
    write(tmp_path, "a.csv", "existing")
    plan = fm.build_plan("normalize", tmp_path)
    result = fm.execute_plan(plan)
    assert result.exit_code == 0
    assert (tmp_path / "a_folder/some_file.csv").read_text() == "data"
    assert (tmp_path / "a_1.csv").read_text() == "accent"
    assert (tmp_path / "a.csv").read_text() == "existing"


@pytest.mark.parametrize(
    "kind,pattern,expected",
    [
        ("ext", ".tmp", {"one.tmp", "backup.tmp"}),
        ("name", "*backup*", {"backup.tmp", "backup.csv"}),
        ("dir", "archive", {"my_archive"}),
    ],
)
def test_remove_matching_rules(tmp_path, kind, pattern, expected):
    for name in ("one.tmp", "backup.tmp", "backup.csv", "keep.csv", "my_archive/inside.csv"):
        write(tmp_path, name)
    plan = fm.build_plan("remove", tmp_path, patterns=[pattern], removal_type=kind)
    assert {t.source.name for t in plan.targets} == expected
    assert fm.execute_plan(plan).completed == len(expected)
    assert (tmp_path / "keep.csv").exists()


def test_remove_fixed_targets_changed_directory_and_symlinks(tmp_path):
    src = tmp_path / "src"
    original = write(src, "one.tmp")
    outside = write(tmp_path / "outside", "secret.tmp")
    (src / "linked.tmp").symlink_to(outside)
    plan = fm.build_plan("remove", src, patterns=[".tmp"])
    added = write(src, "added.tmp")
    assert fm.execute_plan(plan).completed == 1
    assert not original.exists() and added.exists() and outside.exists()
    write(src, "backup/one.txt")
    directory_plan = fm.build_plan("remove", src, patterns=["backup"], removal_type="dir")
    write(src, "backup/added.txt")
    assert fm.execute_plan(directory_plan).exit_code == 1
    assert (src / "backup/one.txt").exists()


def test_changed_file_and_parent_symlink_are_rejected(tmp_path):
    src = tmp_path / "src"
    original = write(src, "a.tmp")
    plan = fm.build_plan("remove", src, patterns=[".tmp"])
    original.write_text("changed since preview")
    assert fm.execute_plan(plan).exit_code == 1
    assert original.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    plan = fm.build_plan("copy", src, tmp_path / "dest")
    (tmp_path / "dest").symlink_to(outside, target_is_directory=True)
    assert fm.execute_plan(plan).exit_code == 1
    assert not list(outside.iterdir())


def test_cancel_before_and_between_files(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    write(src, "a.csv")
    write(src, "b.csv")
    plan = fm.build_plan("move", src, dest)
    cancel = threading.Event()
    cancel.set()
    assert fm.execute_plan(plan, cancel=cancel).cancelled
    assert len(list(src.iterdir())) == 2
    cancel.clear()
    feedback = Feedback(
        "filemanager", callback=lambda line: cancel.set() if "Done:" in line else None
    )
    result = fm.execute_plan(plan, cancel=cancel, feedback=feedback)
    assert result.cancelled and result.completed == 1
    assert len(list(src.iterdir())) == 1


def test_find_tree_and_cli_codes(tmp_path, monkeypatch):
    src, dest = tmp_path / "source files", tmp_path / "reports"
    write(src, "sub/trial.csv", "12345")
    write(src, "other.txt")
    for action in ("find", "tree"):
        plan = fm.build_plan(action, src, dest, extension=".csv")
        assert fm.execute_plan(plan).exit_code == 0
        assert "sub/trial.csv" in plan.report.read_text()
    args = ["remove", "--source", str(src), "--pattern", ".csv"]
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert fm.main(args) == 1
    assert fm.main([*args, "--dry-run"]) == 0
    assert (src / "sub/trial.csv").exists()
    assert fm.main([*args, "--yes"]) == 0
    assert not (src / "sub/trial.csv").exists()


def test_vicon_reuses_converter_and_reports_parse_errors(tmp_path, monkeypatch):
    import vaila.load_vicon_csv_split_batch as converter

    src, dest = tmp_path / "src", tmp_path / "out"
    write(src, "good.csv")
    write(src, "bad.csv")
    write(src, "sub/not_first_level.csv")
    called = []

    def convert(source, destination):
        called.append(source.name)
        write(destination, source.stem + "_dev1.csv")
        return {"dev_0": object()} if source.name == "good.csv" else {"dev_0_error": "parse error"}

    monkeypatch.setattr(converter, "read_csv_devs", convert)
    plan = fm.build_plan("import-vicon", src, dest)
    assert fm.execute_plan(plan, dry_run=True).exit_code == 0
    assert not dest.exists()
    result = fm.execute_plan(plan)
    assert result.completed == 1 and result.exit_code == 1
    assert set(called) == {"good.csv", "bad.csv"}


def test_real_vicon_conversion(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "out"
    write(src, "trial.csv", "Devices\n1000\n\n")
    # No valid device must be reported as a failure, never simulated success.
    assert fm.execute_plan(fm.build_plan("import-vicon", src, dest)).exit_code == 1


def test_transfer_command_is_argument_safe_and_status_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fm.shutil, "which", lambda name: "/usr/bin/rsync" if name == "rsync" else None
    )
    cmd = fm.build_transfer_command(
        str(tmp_path) + "  ", "example.org", "analyst", "/remote/a b; echo secret"
    )
    assert "--protect-args" in cmd
    assert cmd[-1] == "analyst@example.org:/remote/a b; echo secret/"
    with pytest.raises(ValueError):
        fm.build_transfer_command(str(tmp_path), "-oProxyCommand=bad", "user", "/data")
    monkeypatch.setattr(fm.subprocess, "call", lambda argv: 23)
    assert fm.transfer(str(tmp_path), "host", "user", "/data") == 23


def test_worker_rejects_duplicate_and_reports_errors():
    task = WorkerTask()
    release = threading.Event()
    assert task.start(lambda: release.wait(2))
    assert not task.start(lambda: None)
    release.set()
    task.thread.join(3)
    events = list(task.drain())
    assert [kind for kind, _ in events] == ["result", "done"]
    assert not task.busy


def test_headless_cli_and_quoted_paths(tmp_path):
    src = tmp_path / "source with spaces"
    write(src, "data.csv")
    dest = tmp_path / "output with spaces"
    argv = [
        sys.executable,
        "-m",
        "vaila.filemanager",
        "copy",
        "--source",
        str(src),
        "--destination",
        str(dest),
        "--dry-run",
    ]
    if os.name != "nt":
        assert shlex.split(command_text(argv)) == argv
    env = dict(os.environ, DISPLAY="", MPLBACKEND="Agg")
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert not dest.exists()
    code = "import sys; import vaila.filemanager; assert 'tkinter' not in sys.modules; assert 'torch' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_same_source_destination_skips_previous_outputs(tmp_path):
    write(tmp_path, "trial.csv")
    write(tmp_path, "vaila_copy/old_output/trial.csv")
    plan = fm.build_plan("copy", tmp_path, tmp_path)
    assert len(plan.targets) == 1


def test_vicon_valid_device_data(tmp_path):
    import pandas as pd

    src, dest = tmp_path / "src", tmp_path / "out"
    write(
        src,
        "trial.csv",
        "Devices\n1000\nDevice header\nFrame,Sub Frame,Marker\nCount,Count,X\n1,0,1.5\n2,0,2.5\n",
    )
    plan = fm.build_plan("import-vicon", src, dest)
    assert fm.execute_plan(plan).exit_code == 0
    frame = pd.read_csv(next(dest.rglob("*_dev1.csv")))
    assert frame["Marker_X"].tolist() == [1.5, 2.5]
    assert "Timestamp" in frame.columns


def test_dry_run_log_not_selected_and_log_failure_status(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "out"
    write(src, "trial.txt")
    log = src / "run.txt"
    assert (
        fm.main(
            [
                "copy",
                "--source",
                str(src),
                "--destination",
                str(dest),
                "--dry-run",
                "--log-file",
                str(log),
            ]
        )
        == 0
    )
    assert not dest.exists()
    plan = fm.build_plan("copy", src, dest, feedback=Feedback("filemanager", log_file=log))
    assert [target.source.name for target in plan.targets] == ["trial.txt"]
    assert (
        fm.main(
            [
                "copy",
                "--source",
                str(src),
                "--destination",
                str(dest),
                "--log-file",
                str(tmp_path / "absent/log.txt"),
            ]
        )
        == 1
    )
