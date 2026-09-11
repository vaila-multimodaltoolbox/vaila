"""File Manager operations shared by Tkinter and CLI.

Author: Paulo Roberto Pereira Santiago
Version: 0.3.137
Update Date: 11 September 2026
License: AGPL-3.0
"""

import argparse
import ctypes
import fnmatch
import os
import re
import shutil
import stat
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from .task_feedback import Feedback
except ImportError:
    from task_feedback import Feedback


@dataclass(frozen=True)
class Target:
    source: Path
    destination: Path | None
    identity: tuple
    children: tuple = ()


@dataclass
class OperationPlan:
    action: str
    source: Path
    targets: list[Target]
    report: Path | None = None
    description: str = ""


@dataclass
class OperationResult:
    completed: int = 0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def exit_code(self):
        return 1 if self.errors else 130 if self.cancelled else 0

    def summary(self):
        return (
            f"Completed: {self.completed}; failed: {len(self.errors)}; cancelled: {self.cancelled}"
        )


def _identity(path):
    value = path.lstat()
    # Parent directory metadata changes during our own child operations.
    directory = stat.S_ISDIR(value.st_mode)
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        0 if directory else value.st_size,
        0 if directory else value.st_mtime_ns,
    )


def _clean_filename(filename):
    name, ext = os.path.splitext(filename)
    name = unicodedata.normalize("NFKD", name.lower().replace("ç", "c"))
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-z0-9_]", "", name.replace(" ", "_").replace("-", "_"))
    return re.sub(r"_+", "_", name).strip("_") + ext.lower()


def _safe_name(name):
    if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
        raise ValueError(f"Invalid resulting name: {name!r}")
    return name


def _unused(path, reserved):
    candidate, counter = path, 1
    while candidate in reserved or os.path.lexists(candidate):
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    reserved.add(candidate)
    return candidate


def _walk(source, excluded=None, cancel=None):
    def fail(error):
        raise error

    for root, dirs, files in os.walk(source, onerror=fail):
        if cancel and cancel.is_set():
            raise InterruptedError("Search cancelled")
        dirs[:] = sorted(
            d for d in dirs if not (Path(root) / d).is_symlink() and (Path(root) / d) != excluded
        )
        for name in dirs + sorted(files):
            path = Path(root) / name
            if path != excluded and not path.is_symlink():
                yield path


def _directory_snapshot(path):
    items = []

    def fail(error):
        raise error

    for root, dirs, files in os.walk(path, onerror=fail):
        for name in sorted(dirs + files):
            child = Path(root) / name
            items.append((child, _identity(child)))
    return tuple(sorted(items, key=lambda item: str(item[0])))


def build_plan(
    action,
    source,
    destination=None,
    *,
    extension="",
    patterns=(),
    removal_type="ext",
    text="",
    replacement="",
    feedback=None,
    cancel=None,
):
    """Snapshot targets without writing output or opening dialogs."""
    feedback = feedback or Feedback("filemanager")
    if not str(source).strip():
        raise ValueError("Source is required")
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().resolve() if destination else None
    if action != "export" and not source.is_dir():
        raise ValueError("Source must be a directory")
    if action in ("copy", "move", "export", "import-vicon", "find", "tree") and not destination:
        raise ValueError("Destination is required")
    if action == "export" and not source.is_file():
        raise ValueError("Export source must be a file")
    if destination and destination.exists() and not destination.is_dir():
        raise ValueError("Destination must be a directory")
    patterns = list(patterns) or [""]
    if action == "rename":
        if not text:
            raise ValueError("Text to replace cannot be empty")
        if any(c in replacement for c in "/\\\0"):
            raise ValueError("Replacement must be a filename fragment")
    if action == "remove":
        forbidden = {
            "",
            "*",
            ".",
            "/",
            "\\",
            "boot.ini",
            "ntldr",
            "ntdetect.com",
            "autoexec.bat",
            "config.sys",
            "System",
            "System32",
            ".bashrc",
            ".profile",
            ".bash_profile",
            ".bash_logout",
            "/etc/passwd",
            "/etc/shadow",
            ".DS_Store",
            "/System",
            "/Applications",
            "/Users",
            "/Library",
        }
        if len(patterns) != 1 or patterns[0] in forbidden:
            raise ValueError("Removal requires one specific, non-system pattern")
        if removal_type == "dir" and any(p in patterns[0] for p in forbidden if len(p) > 1):
            raise ValueError("System directory patterns are forbidden")
        if source == Path(source.anchor):
            raise ValueError("Removal from a filesystem root is forbidden")
    feedback(
        f"Searching: {source}; action={action}; extension={extension or 'ALL'}; patterns={patterns}"
    )
    excluded = destination if destination != source else None
    if destination == source and action in ("copy", "move"):
        excluded = source / f"vaila_{action}"
    if action == "export":
        paths = [source]
    elif action == "import-vicon":
        paths = sorted(p for p in source.iterdir() if p != excluded and not p.is_symlink())
    else:
        paths = list(_walk(source, excluded, cancel))
    if feedback.log_file:
        log_path = feedback.log_file.resolve()
        paths = [path for path in paths if path != log_path]
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    reserved, targets, moved = set(), [], set()
    if action in ("copy", "move"):
        for pattern in patterns:
            label = re.sub(r"[^\w.-]", "_", pattern.strip("_")) or "all"
            folder = _unused(
                destination / f"vaila_{action}" / f"vaila_{action}_{label}_{stamp}", reserved
            )
            for path in paths:
                if (
                    not path.is_file()
                    or not path.name.endswith(extension)
                    or pattern not in path.name
                ):
                    continue
                if action == "move" and path in moved:
                    continue
                targets.append(Target(path, _unused(folder / path.name, reserved), _identity(path)))
                moved.add(path)
    elif action == "import-vicon":
        folder = _unused(destination / f"vicon_csv_split_{datetime.now():%Y%m%d_%H%M%S}", reserved)
        for path in paths:
            if path.parent == source and path.is_file() and path.suffix == ".csv":
                targets.append(Target(path, folder / f"{path.stem}_splitdevice", _identity(path)))
    else:
        removed_dirs = []
        ordered = (
            sorted(paths, key=lambda p: (-len(p.parts), str(p))) if action == "normalize" else paths
        )
        for path in ordered:
            is_dir, dest, children = path.is_dir(), None, ()
            if action == "remove":
                pattern = patterns[0]
                if any(path.is_relative_to(parent) for parent in removed_dirs):
                    continue
                match = (
                    (is_dir and pattern in path.name)
                    if removal_type == "dir"
                    else (
                        not is_dir
                        and (
                            path.name.endswith(pattern)
                            if removal_type == "ext"
                            else fnmatch.fnmatch(path.name, pattern)
                        )
                    )
                )
                if not match:
                    continue
                if is_dir:
                    if feedback.log_file and feedback.log_file.resolve().is_relative_to(path):
                        raise ValueError("Keep --log-file outside directories selected for removal")
                    children = _directory_snapshot(path)
                    removed_dirs.append(path)
            elif action == "rename":
                if is_dir or not path.name.endswith(extension) or text not in path.name:
                    continue
                dest = path.with_name(_safe_name(path.name.replace(text, replacement)))
                if dest == path:
                    continue
            elif action == "normalize":
                name = (
                    _clean_filename(path.name + ".tmp")[:-4]
                    if is_dir
                    else _clean_filename(path.name)
                )
                name = _safe_name(name)
                if name == path.name:
                    continue
                dest = _unused(path.with_name(name), reserved)
            elif action == "export":
                dest = _unused(destination / path.name, reserved)
            elif action in ("find", "tree"):
                ext = extension if extension.startswith("*") else "*" + extension
                match = (
                    (not is_dir and path.name.endswith(extension))
                    if action == "tree"
                    else any(
                        fnmatch.fnmatch(path.name, f"*{p}*{ext}" if p else ext) for p in patterns
                    )
                )
                if not match:
                    continue
            else:
                raise ValueError(f"Unknown action: {action}")
            targets.append(Target(path, dest, _identity(path), children))
    report = (
        _unused(destination / f"vaila_{action}_{stamp}.txt", reserved)
        if action in ("find", "tree")
        else None
    )
    feedback(f"Found {len(targets)} targets. Preview ready; no files changed.")
    return OperationPlan(
        action, source, targets, report, f"extension={extension}; patterns={patterns}"
    )


def _rename_no_replace(source, destination):
    """OS no-replace semantics also protect existing empty directories."""
    if source.is_file():
        os.link(source, destination, follow_symlinks=False)
        source.unlink()
    elif sys.platform == "win32":
        os.rename(source, destination)
    else:
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            rc = libc.renamex_np(os.fsencode(source), os.fsencode(destination), 4)
        elif hasattr(libc, "renameat2"):
            rc = libc.renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        else:
            raise OSError("Atomic directory rename without replacement is unavailable on this OS")
        if rc:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(destination))


def _copy_no_replace(source, destination):
    with destination.open("xb") as output:
        try:
            with source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
        except BaseException:
            output.close()
            destination.unlink()
            raise
    shutil.copystat(source, destination)


def execute_plan(plan, *, dry_run=False, cancel=None, feedback=None):
    feedback = feedback or Feedback("filemanager")
    result = OperationResult()
    feedback(f"{'Preview' if dry_run else 'Executing'} {plan.action}: {len(plan.targets)} targets")
    report_lines = []
    for target in plan.targets:
        if cancel and cancel.is_set():
            result.cancelled = True
            break
        src, dest = target.source, target.destination
        feedback(f"{src}" + (f" -> {dest}" if dest else ""))
        if dry_run:
            continue
        try:
            if src.resolve() != src or _identity(src) != target.identity:
                raise OSError("Target changed since preview; preview again")
            if not src.is_relative_to(plan.source):
                raise ValueError("Target is outside the selected source")
            if dest:
                if dest.parent.resolve() != dest.parent:
                    raise OSError("Destination parent changed since preview")
                dest.parent.mkdir(parents=True, exist_ok=True)
            if plan.action in ("copy", "export", "move"):
                _copy_no_replace(src, dest)
                if plan.action == "move":
                    src.unlink()
            elif plan.action in ("rename", "normalize"):
                _rename_no_replace(src, dest)
            elif plan.action == "remove":
                if src.is_dir():
                    if _directory_snapshot(src) != target.children:
                        raise OSError("Directory contents changed since preview; preview again")
                    for child, identity in sorted(
                        target.children, key=lambda item: -len(item[0].parts)
                    ):
                        if _identity(child) != identity:
                            raise OSError(f"Target changed: {child}")
                        child.rmdir() if stat.S_ISDIR(identity[2]) else child.unlink()
                    src.rmdir()
                else:
                    src.unlink()
            elif plan.action == "import-vicon":
                try:
                    from .load_vicon_csv_split_batch import read_csv_devs
                except ImportError:
                    from load_vicon_csv_split_batch import read_csv_devs
                dest.mkdir()
                devices = read_csv_devs(src, dest)
                if not devices or any(key.endswith("_error") for key in devices):
                    raise ValueError(
                        "VICON parsing failed or produced no devices; inspect partial outputs"
                    )
            elif plan.action in ("find", "tree"):
                report_lines.append(str(src.relative_to(plan.source)))
            result.completed += 1
            feedback(f"Done: {src}")
        except Exception as error:
            result.errors.append(f"{src}: {error}")
            feedback.error(error)
    if plan.report and not dry_run and not result.cancelled:
        try:
            plan.report.parent.mkdir(parents=True, exist_ok=True)
            with plan.report.open("x", encoding="utf-8") as stream:
                if plan.action == "find":
                    files = [t for t in plan.targets if stat.S_ISREG(t.identity[2])]
                    stream.write(
                        f"Summary of Search Results\nPattern Searched: {plan.description}\n"
                        f"Number of Files Found: {len(files)}\n"
                        f"Total Size: {sum(t.identity[3] for t in files) / 1024**2:.2f} MB\n\nFile Tree:\n"
                    )
                stream.write("\n".join(report_lines) + "\n")
            feedback(f"Report saved: {plan.report}")
        except Exception as error:
            result.errors.append(str(error))
            feedback.error(error)
    feedback(result.summary())
    return result


def build_transfer_command(local, host, user, remote, port=22, mode="upload"):
    local = str(Path(local.strip()).expanduser().resolve(strict=True))
    if not Path(local).is_dir():
        raise ValueError("Local path must be a directory")
    host, user, remote = host.strip(), user.strip(), remote.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("Invalid SSH host")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", user):
        raise ValueError("Invalid SSH user")
    if not 1 <= int(port) <= 65535 or mode not in ("upload", "download"):
        raise ValueError("Invalid port or direction")
    if not remote.startswith("/") or any(c in remote for c in "\n\r\0"):
        raise ValueError("Use an absolute remote directory")
    endpoint = f"{user}@{host}:{remote.rstrip('/') or '/'}"
    rsync = shutil.which("rsync")
    if rsync:
        src, dest = (local, endpoint + "/") if mode == "upload" else (endpoint, local + "/")
        return [rsync, "-avzhP", "--protect-args", "-e", f"ssh -p {int(port)}", "--", src, dest]
    scp = shutil.which("scp")
    if not scp:
        raise OSError("Install rsync or OpenSSH scp to transfer files")
    if re.search(r"[^A-Za-z0-9_./ -]", remote):
        raise ValueError("This remote path requires rsync (legacy SCP shell characters)")
    src, dest = (local, endpoint) if mode == "upload" else (endpoint, local)
    return [scp, "-r", "-P", str(port), "-v", "-C", "--", src, dest]


def transfer(local, host, user, remote, port=22, mode="upload", feedback=None):
    feedback = feedback or Feedback("filemanager")
    cmd = build_transfer_command(local, host, user, remote, port, mode)
    feedback("Transfer started. Authentication and progress are in this terminal.")
    code = subprocess.call(cmd)
    feedback("Transfer completed." if code == 0 else f"Transfer failed (exit {code}).")
    return code


def build_parser():
    parser = argparse.ArgumentParser(description="vailá File Manager: shared GUI/CLI operations")
    subs = parser.add_subparsers(dest="action")
    for action in (
        "copy",
        "move",
        "remove",
        "rename",
        "normalize",
        "find",
        "tree",
        "export",
        "import-vicon",
        "transfer",
    ):
        example = (
            f'{action} --local "/data/my files" --host server --user analyst --remote /data'
            if action == "transfer"
            else f'{action} --source "/data/my files"'
            + (
                ' --destination "/data/results"'
                if action in ("copy", "move", "find", "tree", "export", "import-vicon")
                else ""
            )
            + (" --pattern .tmp --removal-type ext" if action == "remove" else "")
            + (" --text old --replacement new" if action == "rename" else "")
            + " --dry-run"
        )
        sub = subs.add_parser(action, epilog=f"Example: python -m vaila.filemanager {example}")
        sub.add_argument(
            "--debug", action="store_true", help="Include technical details and traceback"
        )
        sub.add_argument("--log-file", help="Append messages to this file")
        if action == "transfer":
            for name in ("local", "host", "user", "remote"):
                sub.add_argument(f"--{name}", required=True)
            sub.add_argument("--port", type=int, default=22)
            sub.add_argument("--mode", choices=("upload", "download"), default="upload")
            sub.add_argument("--status-file", help=argparse.SUPPRESS)
            continue
        sub.add_argument("--source", required=True, help="Source file (export) or directory")
        if action in ("copy", "move", "find", "tree", "export", "import-vicon"):
            sub.add_argument("--destination", required=True, help="Destination directory")
        if action in ("copy", "move", "rename", "find", "tree"):
            sub.add_argument(
                "--extension", default="", help="Case-sensitive suffix (.csv); blank means all"
            )
        if action in ("copy", "move", "remove", "find"):
            sub.add_argument(
                "--pattern",
                action="append",
                default=[],
                help="Copy/move: literal substring; find: glob fragment; remove: depends on type",
            )
        if action == "remove":
            sub.add_argument(
                "--removal-type",
                choices=("ext", "name", "dir"),
                default="ext",
                help="ext: suffix; name: glob; dir: substring",
            )
            sub.add_argument(
                "--yes", action="store_true", help="Confirm permanent removal of previewed targets"
            )
        if action == "rename":
            sub.add_argument("--text", required=True)
            sub.add_argument(
                "--replacement", required=True, help="New text; empty string removes old text"
            )
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Preview only; no output or changes (except optional log)",
        )
    return parser


def _plan_args(args):
    names = ("action", "source", "destination", "extension", "removal_type", "text", "replacement")
    kwargs = {name: getattr(args, name) for name in names if hasattr(args, name)}
    kwargs["patterns"] = getattr(args, "pattern", [])
    return kwargs


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.action:
        show_action("copy")
        return 0
    feedback = Feedback("filemanager", args.debug, args.log_file)
    code = 1
    try:
        feedback(f"Opened action: {args.action}")
        if args.action == "transfer":
            code = transfer(
                args.local, args.host, args.user, args.remote, args.port, args.mode, feedback
            )
        else:
            plan = build_plan(**_plan_args(args), feedback=feedback)
            if args.action in ("remove", "move", "rename", "normalize") and not args.dry_run:
                execute_plan(plan, dry_run=True, feedback=feedback)
            if args.action == "remove" and not args.dry_run and not args.yes:
                if not sys.stdin.isatty():
                    raise ValueError("Permanent removal requires --yes in noninteractive mode")
                feedback("Waiting for confirmation: permanent removal. Type yes to proceed.")
                if input().strip() != "yes":
                    feedback("Cancelled; no files changed.")
                    return 0
            code = execute_plan(plan, dry_run=args.dry_run, feedback=feedback).exit_code
    except (KeyboardInterrupt, EOFError):
        feedback("Cancelled.")
        code = 130
    except Exception as error:
        feedback.error(error)
    finally:
        if args.action == "transfer" and args.status_file:
            Path(args.status_file).write_text(str(code), encoding="utf-8")
    return code


def show_action(action):
    try:
        from .filemanager_gui import show_action as show
    except ImportError:
        from filemanager_gui import show_action as show
    return show(action)


def copy_file():
    return show_action("copy")


def move_file():
    return show_action("move")


def remove_file():
    return show_action("remove")


def rename_files():
    return show_action("rename")


def normalize_names():
    return show_action("normalize")


def find_file():
    return show_action("find")


def tree_file():
    return show_action("tree")


def export_file():
    return show_action("export")


def import_file():
    return show_action("import-vicon")


def transfer_file():
    return show_action("transfer")


def process_copy(src_directory, file_extension, patterns, destination):
    return execute_plan(
        build_plan(
            "copy", src_directory, destination, extension=file_extension or "", patterns=patterns
        )
    )


def process_move(src_directory, file_extension, patterns, destination):
    return execute_plan(
        build_plan(
            "move", src_directory, destination, extension=file_extension or "", patterns=patterns
        )
    )


if __name__ == "__main__":
    sys.exit(main())
