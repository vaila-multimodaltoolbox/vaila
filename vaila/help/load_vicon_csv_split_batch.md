# VICON CSV Split Batch Processor

**Version:** 0.3.137
**Updated:** 2026-09-11

Split first-level VICON Nexus CSV files into device CSVs. The existing converter cleans and merges headers, adds the source file creation Timestamp and preserves device data.

Use **Frame A → Import** for explicit parameters, preview and background execution. Other formats are unavailable. See [File Manager](filemanager.md) for cancellation, diagnostics and output safety.

~~~bash
python -m vaila.filemanager import-vicon --source "/data/VICON CSV" --destination "/data/output" --dry-run
python -m vaila.filemanager import-vicon --source "/data/VICON CSV" --destination "/data/output" --debug
~~~

Outputs: vicon_csv_split_TIMESTAMP/STEM_splitdevice/STEM_devN.csv. Only first-level .csv sources are converted. No devices or any parser-error block means failure; inspect partial files. The File Manager never reuses an existing output directory.

The original standalone directory chooser remains available through python -m vaila.load_vicon_csv_split_batch. Tkinter is imported only when opening that chooser, so read_csv_devs can run without Tk or a display. Prefer the File Manager CLI for explicit arguments and reliable batch failure status.
