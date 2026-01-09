# vailá Sprint Analysis (20m) - Help

Welcome to the **vailá Sprint Analysis** module. This tool provides a comprehensive biomechanical and performance analysis of 20-meter linear sprints using data collected by the vailá Tracker. It is designed to automatically process multiple runs, generate visual reports, and compile a database for team-wide analysis.

## 🚀 Workflow

1.  **Launch**: Open `vaila.py`, verify your settings, and click the **Sprint** button (bottom left of the main interface).
2.  **Select Mode**:
    - **Time Sprint (20m)**: Choose this for standard 20m linear sprints.
    - *COD 90 Degree (20m)*: (Coming Soon) Support for Change of Direction tests.
3.  **Select Data Folder**: Choose the directory containing your `.toml` tracking files.
    - **Crucial Tip**: For automatic video frame extraction (0m, 5m, etc.), ensure the video files are located in the **same folder** as the `.toml` files, or at the path specified inside the TOML.
4.  **Processing**: The script will iterate through every `.toml` file found, calculate kinematics, and generate reports.
5.  **Output**: Upon completion, the `vaila_sprint_reports` folder will open automatically.

---

## 📂 Output Structure (`vaila_sprint_reports`)

All results are organized in a structured manner to facilitate both individual feedback and group analysis.

### 1. Main Dashboard (`general_report.html`)
**Target Audience: Head Coach, Physical Trainer**
- **Purpose**: Talent identification and team monitoring.
- **Practical Use**: fast identification of the fastest player in the squad. Use the **Rankings** to select players for specific tactical roles (e.g., wingers vs. defenders).
- **Global Database** (`vaila_sprint_database.csv`): A master file containing every data point. **Practical Use**: Physical trainers can import this into Excel/PowerBI to track season-long progress or compare squads (e.g., U17 vs Pro).

### 2. Individual Athlete Reports
**Target Audience: The Athlete, Performance Analyst**
A dedicated subfolder is created for each analyzed file (e.g., `Silva_analysis...`). Inside, you will find specific files:

#### A. The Interactive Report (`*_report_sprint20m.html`)
**What is it?** A single file containing the complete visual analysis of the run.
**What's inside?**
- **Speed Curve**: Show the athlete *where* they reached top speed. In football, reaching top speed earlier (acceleration) is often more important than the top speed itself.
- **Usain Bolt Comparison**: Educational tool to show the difference between an elite sprinter's profile and the player's profile.
- **Video Evidence**: Extracts frames at 0m, 5m, 10m, 15m, and 20m.
    - **0m**: Check "Set" position. Center of gravity height.
    - **5m**: Check drive phase angle (approx 45°).
    - **20m**: Check upright posture and mechanics at max speed.

#### B. The Data Files (`*_data.xlsx` / `*_data.csv`)
**What are they?** Raw numerical data for every split calculated.
**Columns included:**
1.  **distance_cumulative**: Distance point (e.g., 5.0, 10.0, 15.0, 20.0 meters).
2.  **duration**: Time taken to cover that specific segment.
3.  **speed_ms** & **speed_kmh**: Average speed in that segment.
4.  **acceleration_ms2**: Acceleration value for that segment.
**Practical Use**:
- Import into **Excel** to calculate specialized metrics like "Fatigue Index" (drop in speed).
- Compare the **0-10m split** specifically, which is critical for multidirectional sports.

#### C. The Images (`*.png`)
- **Plots**: High-resolution images of the speed and acceleration graphs (useful for WhatsApp/Instagram reports).
- **Frames**: The individual extracted video frames (0m, 5m, etc.).

---

## 📈 Understanding the Metrics

### Speed (Velocity)
- **Unit**: Reported in **km/h** (standard for communication) and **m/s** (scientific standard).
- **Interpretation**:
    - **Max Speed**: The highest momentary velocity reached. In a 20m sprint, this typically occurs near the end.
    - **Reference**: Usain Bolt's peak speed was ~44.72 km/h (12.42 m/s). Top soccer players often reach 32-36 km/h.

### Acceleration
- **Unit**: Meters per second squared (m/s²).
- **Interpretation**: How quickly the athlete gains speed.
    - **Start Phase (0-5m)**: should show the highest acceleration values (explosive power).
    - **Transition Phase**: Acceleration decreases as speed increases.
    - **Zero Acceleration**: Means the athlete has reached their maximum constant speed.

---

## 🛠 Troubleshooting

- **"No video frames extracted"**:
  - The script looks for the video file name stored in the `.toml`.
  - **Fix**: Copy the original video files (e.g., `run1.mp4`) into the same folder where your `.toml` files are located before running the analysis.
- **Logo missing**:
  - The report looks for `vaila.png` in `docs/images/` or locally. Ensure the project structure is intact.
