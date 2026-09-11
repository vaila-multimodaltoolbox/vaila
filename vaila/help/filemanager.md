# File Manager

**Version:** 0.3.137
**Updated:** 2026-09-11

Manage local files from Frame A or the command line. GUI and CLI use the same target selection and operations. Terminal messages always use **>> vaila/filemanager:** and flush immediately.

## GUI

1. Choose Copy, Move, Remove, Rename, Import, Export, Tree, Find or Transfer in Frame A. The existing Rename button opens name normalization; literal replacement remains available through the rename entry point and CLI.
2. Fill Source, Destination and applicable filter fields. Cancelling Browse leaves the form open for review.
3. Click **Preview**. The search runs in a worker. Read the complete target list in the window and terminal; preview creates no output files.
4. Click **Apply preview**. Move, rename, normalization and removal ask for confirmation of that fixed list. Changing fields requires a new preview.
5. Watch completed/failure counts and cancellation. **Cancel / Close** cancels between files; the current copy or conversion finishes first. A second execution cannot start in the same window.
6. **Help** opens this local HTML help. **Diagnostic details** adds technical messages and tracebacks.

Removal is permanent; it does not use the recycle bin. Files added after preview are not selected. A selected directory whose contents change after preview is refused. Changed files and symbolic-link substitutions are refused. Recursive operations skip symbolic links; links inside an explicitly selected removal directory are removed as links, without following them.

## Matching and outputs

- **Copy / Move:** recursive, case-sensitive filename suffix plus literal substring. .csv matches trial.csv, not trial.CSV. trial* is a literal string here. Multiple patterns use separate GUI lines or repeated --pattern. Blank filters mean all files.
- **Copy:** flattened outputs in DEST/vaila_copy/vaila_copy_PATTERN_TIMESTAMP/. Overlapping patterns receive their own copies. Collisions get _1, _2, etc.
- **Move:** same layout under vaila_move; each file belongs to its first matching pattern. Collisions get suffixes. An exclusive copy must succeed before the source is removed, including across filesystems.
- **Export:** copy one file to the destination. Existing names get suffixes; no conversion.
- **Rename:** recursively replace literal text in matching filenames. Empty replacement removes text. Existing destinations cause individual failures, never overwrites.
- **Normalize:** files and directories, children first. Lowercase, remove accents and punctuation, replace spaces/hyphens with underscores, lowercase extensions. Directory dots are removed; the last file extension is retained. Collisions get suffixes; unsafe/empty resulting names are refused.
- **Remove:** ext means case-sensitive suffix; name uses shell-style wildcards such as *backup*; dir means literal directory-name substring, including contents. Broad/system patterns remain forbidden.
- **Tree:** recursive relative file paths matching a suffix, saved as vaila_tree_TIMESTAMP.txt.
- **Find:** glob name fragments plus extension, including matching directories. vaila_find_TIMESTAMP.txt records relative paths, file count and aggregate file size in MB.
- **Import VICON:** only first-level .csv files. Reuses the existing device splitter. Outputs vicon_csv_split_TIMESTAMP/STEM_splitdevice/STEM_devN.csv, including cleaned headers and Timestamp. Parse failures or zero devices are failures; inspect partial outputs. Other import formats are unavailable.
- A destination nested in the source is excluded from searching. When destination equals source, the operation's copy/move output tree is excluded. Searches finish before outputs are created.

## CLI examples

Every subcommand provides --help, --debug and optional --log-file. Every local operation supports --dry-run, including reports and VICON. The optional log is the only file a dry run may write.

~~~bash
python -m vaila.filemanager copy --source "/data/my files" --destination "/data/out" --extension .csv --pattern trial --dry-run
python -m vaila.filemanager move --source "/data/my files" --destination "/data/out" --extension .csv --pattern trial
python -m vaila.filemanager remove --source "/data/my files" --removal-type name --pattern "*backup*" --dry-run
python -m vaila.filemanager remove --source "/data/my files" --removal-type name --pattern "*backup*" --yes
python -m vaila.filemanager rename --source "/data/my files" --extension .csv --text old --replacement new --dry-run
python -m vaila.filemanager normalize --source "/data/my files" --dry-run
python -m vaila.filemanager find --source "/data/my files" --destination "/data/reports" --pattern "trial*" --extension .csv
python -m vaila.filemanager tree --source "/data/my files" --destination "/data/reports" --extension .csv
python -m vaila.filemanager export --source "/data/my file.csv" --destination "/data/out" --dry-run
python -m vaila.filemanager import-vicon --source "/data/vicon" --destination "/data/out" --debug --log-file "/data/import.log"
python -m vaila.filemanager transfer --local "/data/my files" --host server.example --user analyst --remote /data/incoming --port 22 --mode upload
~~~

Omit --dry-run to apply local operations. Removal without --yes prompts only in an interactive terminal; noninteractive removal requires --yes. GUI removal commands always include --dry-run for safe replay. Commands use native shell quoting (POSIX shell or Windows cmd).

Exit codes: 0 success or interactive confirmation declined; 1 failure including partial failures; 2 invalid arguments; 130 interrupted/cancelled execution. Transfer propagates the external tool's exit status. Completed work may remain after cancellation or partial failure.

## SSH transfer

The GUI collects local directory, absolute remote directory, SSH host/user/port and upload/download direction. Authentication and progress run in an interactive terminal. Passwords are never collected in GUI fields or written to logs.

**Terminal opened** is not transfer completion. The GUI waits for the CLI process's status file and displays **Transfer completed** only for exit zero. If the terminal closes before reporting, status remains unknown. Closing the GUI with unknown status does not cancel the transfer.

The CLI uses the current terminal, preferring rsync with protected arguments, otherwise OpenSSH scp. Both transfer the selected directory itself. SCP rejects remote shell metacharacters; use rsync for such paths. Remote-file replacement follows the transfer tool's normal behavior; local preview protections do not apply to remote transfers.

## Diagnosis

Essential stages, parameters, matches, changes, failures and summaries always appear in the terminal. Add --debug for technical detail and traceback, and --log-file to retain messages. Keep logs outside selected removal directories. Passwords, cookies and tokens are redacted.

Project agents can invoke /debug to reproduce safely with temporary files and simulated backends. It never automatically removes real data, transfers files or downloads media.

Implementation: filemanager.py owns selection/execution; filemanager_gui.py owns Tk widgets; task_feedback.py owns terminal feedback and queued worker events. Processing functions contain no dialogs; workers never update widgets.
