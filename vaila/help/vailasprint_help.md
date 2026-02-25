# vaila Sprint Analysis (20m) - Help

Welcome to the **vaila Sprint Analysis** module. This tool provides a comprehensive biomechanical and performance analysis of 20-meter linear sprints and Change of Direction (COD 180°) tests using data collected by the vaila Tracker. It is designed to automatically process multiple runs, generate visual reports, and compile a database for team-wide analysis.

## 🚀 Workflow

1.  **Launch**: Open `vaila.py`, verify your settings, and click the **Sprint** button (bottom left of the main interface).
2.  **Select Mode**:
    - **Time Sprint (20m)**: Choose this for standard 20m linear sprints.
    - **2X COD 180 degree (20m)**: Support for Change of Direction tests (2x10m with 180° turn).
3.  **Select Data Folder**: Choose the directory containing your `.toml` tracking files.
    - **Crucial Tip**: For automatic video frame extraction (0m, 5m, etc.), ensure the video files are located in the **same folder** as the `.toml` files, or at the path specified inside the TOML.
4.  **Processing**: The script will iterate through every `.toml` file found, calculate kinematics, and generate reports.
5.  **Output**: Upon completion, the `vaila_sprint_reports` folder will open automatically.

---

## 📂 Output Structure (`vaila_sprint_reports`)

All results are organized in a structured manner to facilitate both individual feedback and group analysis.

### 1. Main Dashboard (`general_report.html`)

**Target Audience: Head Coach, Physical Trainer**

The general report now includes comprehensive team analysis:

#### Key Statistics Banner

- **Athletes Count**: Total number of athletes analyzed
- **Total Runs**: Number of runs processed
- **Top Speed**: Maximum speed achieved with **athlete name and run ID**
- **Best Time**: Fastest 20m time with **athlete name and run ID**

#### Team Statistics Section

- **Average Speed**: Team mean speed (km/h)
- **Speed Std Dev**: Variability in team performance
- **Average Time**: Team mean time (seconds)
- **Time Std Dev**: Consistency measure

#### Visual Performance Analysis

Depending on the number of processed runs, the charts adapt to show the most relevant information:

- **Dumbbell Chart** (2 runs): Compare Run 1 vs Run 2 for each athlete. Green lines = improved, Red lines = declined.
- **Multi-Run Dot Plot** (3+ runs): Shows an athlete's performance range across all runs, highlighting consistency.
- **Improvement Scatter Plot** (2 runs): Points above diagonal show improvement from Run 1 to Run 2.
- **Sequential Improvement Scatter Plot** (3+ runs): Visualizes step-by-step progress (Run 1 -> 2, Run 2 -> 3) with directional arrows.
- **Performance Heatmap** (3+ runs): Complete metrics matrix with color-coded values for in-depth analysis.

#### K-Means Cluster Analysis (3 Levels)

Athletes are automatically classified into 3 performance groups:

- **High Performers** (Green): Top tier athletes
- **Medium Performers** (Orange): Average performers
- **Low Performers** (Red): Athletes needing development

**Beeswarm Plot**: Visual distribution of athletes by cluster, showing individual data points colored by performance level.

#### Z-Score Analysis

Standardized scores showing how each athlete compares to the group average:

- **Dark Green (Z > 1.5)**: Excellent - significantly above average
- **Green (0.5 < Z < 1.5)**: Good - above average
- **Yellow (-0.5 < Z < 0.5)**: Average
- **Red (-1.5 < Z < -0.5)**: Below average
- **Dark Red (Z < -1.5)**: Low - significantly below average

**Composite Z-Score**: Combined score for overall performance ranking.

#### Performance Rankings

- **Ranking by Max Speed**: Fastest athletes first
- **Ranking by Total Time**: Quickest completion times first

#### Global Database (`vaila_sprint_database.csv`)

A master file containing every data point. **Practical Use**: Physical trainers can import this into Excel/PowerBI to track season-long progress or compare squads (e.g., U17 vs Pro).

### 2. Individual Athlete Reports

**Target Audience: The Athlete, Performance Analyst**
A dedicated subfolder is created for each analyzed file (e.g., `Silva_analysis...`). Inside, you will find specific files:

#### A. The Interactive Report (`*_report_sprint20m.html`)

**What is it?** A single file containing the complete visual analysis of the run.
**What's inside?**

- **Speed Curve**: Show the athlete _where_ they reached top speed. In football, reaching top speed earlier (acceleration) is often more important than the top speed itself.
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

### Z-Score

- **Unit**: Standard deviations from the mean.
- **Interpretation**: How an individual compares to the group.
  - **Positive Z-Score**: Above average performance
  - **Negative Z-Score**: Below average performance
  - **|Z| > 2**: Exceptional or concerning (outlier)

### Cluster Assignment

- **Method**: K-means clustering on max speed and total time
- **Interpretation**: Data-driven grouping of similar performers
  - Useful for creating training groups
  - Identifying athletes ready for promotion or needing support

---

## 🛠 Troubleshooting

- **"No video frames extracted"**:
  - The script looks for the video file name stored in the `.toml`.
  - **Fix**: Copy the original video files (e.g., `run1.mp4`) into the same folder where your `.toml` files are located before running the analysis.
- **Logo missing**:
  - The report looks for `vaila.png` in `docs/images/` or locally. Ensure the project structure is intact.
- **K-means clustering disabled**:
  - Requires scikit-learn package. Install with: `pip install scikit-learn`
- **Statistical profiling disabled**:
  - Requires ydata-profiling package. Install with: `pip install ydata-profiling`

---

## 📦 Dependencies

**Required:**

- pandas, numpy, matplotlib, seaborn
- scipy (for Z-score calculations)
- toml (for reading TOML files)
- tkinter (for GUI dialogs)

**Optional:**

- opencv-python (cv2): For video frame extraction
- scikit-learn: For K-means clustering analysis
- ydata-profiling: For statistical profiling reports
