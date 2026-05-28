# API Functions


## vaila_deadlift

- `process_deadlift_kinematics(df, fps, factor)`: calculates frame-by-frame deadlift metrics, including arm verticality, bar-over-midfoot error, hip/knee angular velocities, and synchronism ratio.
- `evaluate_initial_pull_synchronism(df, phases)`: checks the first 15% of the concentric pull for premature knee extension relative to hip opening.
- `identify_deadlift_phases(df)`: estimates start, bottom, and end frames from shoulder vertical velocity.
- `classify_variant_at_bottom(df, bottom_frame)`: classifies stiff-legged deadlift, RDL, conventional deadlift, or mixed variant.
- `generate_html_report(df, phases, variant, plot_files, output_dir, base_name)`: writes the diagnostic HTML report.
- `process_mediapipe_deadlift_data(input_file, output_dir)`: end-to-end CSV processing entry point.
- `main_gui()`: Tkinter folder-selection workflow used by the Frame B Deadlift button.
