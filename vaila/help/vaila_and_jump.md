# vaila_and_jump

## 📋 Module Information

- **Category:** Analysis
- **File:** `vaila\vaila_and_jump.py`
- **Lines:** 4514
- **Size:** ~150000 characters
- **Version:** 0.1.1
- **Author:** Prof. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes
- **Last Update:** 04 Jan 2026

## 📖 Description

This script processes jump data from multiple .csv files in a specified directory, performing biomechanical calculations based on either the time of flight or the jump height. The results are saved in a new output directory with a timestamp for each processed file.

For MediaPipe data, the script automatically inverts y-coordinates (1.0 - y) to transform from screen coordinates (where y increases downward) to biomechanical coordinates (where y increases upward). This allows proper visualization and analysis of the jumping motion.

### Features:

- **Three calculation modes:**
  - Based on time of flight (calculates jump height)
  - Based on measured jump height (uses the height directly)
  - Processes MediaPipe pose estimation data:
    - Automatically inverts y-coordinates for proper biomechanical analysis
    - Converts normalized coordinates to meters using shank length as reference
    - Calculates center of gravity (CG) position for accurate jump height

- **Jump Performance Metrics:**
  - Force
  - Liftoff Force (Thrust)
  - Velocity (takeoff)
  - Potential Energy
  - Kinetic Energy
  - Average Power (if contact time is provided)
  - Relative Power (W/kg)
  - Jump Performance Index (JPI)
  - Total time (if time of flight and contact time are available)

- **Advanced Kinematic Analysis (NEW in v0.1.1):**
  - **FPPA (Frontal Plane Projection Angle):** Rigorous 2D vector-based calculation
    - Vector v1: HIP → KNEE (Femur)
    - Vector v2: KNEE → ANKLE (Tibia)
    - Calculates internal angle at the knee joint
    - Positive values = Valgus (adduction, medial collapse)
    - Negative values = Varus (abduction, lateral collapse)
    - **Right side sign inversion:** For consistent visualization, right side FPPA is inverted to match left side convention (both show valgus as positive)
  
  - **Landing Phase Analysis:** Multiple time points based on scientific literature
    - Initial Contact (IC) at landing
    - IC + 40ms
    - IC + 100ms
    - Max Valgus (within 0.2s post-landing window)
  
  - **Risk Classification:** Evidence-based categorization
    - **< 5°:** Good Alignment (Green)
    - **5° - 10°:** Moderate Risk (Yellow)
    - **> 10°:** High Risk / Excessive Dynamic Valgus (Red)
  
  - **Valgus Ratio:** Knee-Hip Separation Ratio (Knee Dist / Hip Dist)
    - Ratio < 0.8 indicates excessive knee approximation (High Risk)
  
  - **KASR (Knee-to-Ankle Separation Ratio):** Additional alignment metric
  
  - **Landing Stability:** Horizontal CG sway in 0.4s post-landing window

- **Visualizations:**
  - Jump phase analysis plots
  - CG and feet position analysis
  - Power curves
  - **FPPA Time Series Plot (NEW):** Shows adduction/abduction angles over time with key events highlighted
  - **Valgus Event Stick Figures (NEW):** Multi-moment landing analysis with FPPA values, knee/ankle separation, and risk classification
  - Animated GIF of jump cycle
  - Comprehensive HTML report with all metrics and visualizations

- **Output Files:**
  - CSV files with jump metrics
  - Full processed data with all original and calculated values
  - Calibrated data in meters with proper coordinate orientation
  - Visualizations of the jump performance
  - HTML report summarizing all analysis

## 🔧 Main Functions

**Total functions found:** 30+

### Core Calculation Functions:
- `calculate_force` - Calculate weight force based on mass
- `calculate_jump_height` - Calculate jump height from time of flight
- `calculate_power` - Calculate power output during jump
- `calculate_velocity` - Calculate takeoff velocity
- `calculate_kinetic_energy` - Calculate kinetic energy
- `calculate_potential_energy` - Calculate potential energy
- `calculate_average_power` - Calculate average power output
- `calculate_liftoff_force` - Calculate total liftoff force (thrust)
- `calculate_time_of_flight` - Calculate time of flight from jump height

### Phase Identification:
- `calculate_baseline` - Calculate baseline using first n frames
- `identify_jump_phases` - Identify all jump phases (squat, takeoff, peak, landing)

### Kinematic Analysis (NEW):
- `calculate_kinematics` - Calculate advanced kinematic metrics for injury screening
  - FPPA (Frontal Plane Projection Angle) using rigorous 2D vector method
  - Valgus Ratio (Knee-Hip Separation Ratio)
  - KASR (Knee-to-Ankle Separation Ratio)
  - Landing Stability (Sway)
  - Risk classification based on scientific evidence

### Visualization Functions:
- `plot_valgus_ratio` - Generate plot of Knee-Hip Separation Ratio over time
- `plot_fppa_time_series` - Generate time series plot of FPPA angles with key events
- `plot_valgus_event` - Generate annotated stick figure plots for Squat and Landing events
  - Multi-moment landing analysis (IC, IC+40ms, IC+100ms, Max Valgus)
  - Displays FPPA values with risk classification
  - Shows knee and ankle separation distances
- `generate_jump_plots` - Generate power curve visualizations
- `plot_jump_phases_analysis` - Generate visualization showing jump phases with colored regions
- `plot_jump_cg_feet_analysis` - Generate visualization showing CG and feet positions
- `generate_jump_animation_gif` - Generate animated GIF of jump cycle
- `plot_jump_stickfigures_subplot` - Plot sequence of stick figures at key phases
- `plot_jump_stickfigures_with_cg` - Plot stick figures with CG pathway

### Data Processing:
- `process_mediapipe_data` - Process MediaPipe data and generate visualizations
- `process_all_mediapipe_files` - Process all MediaPipe CSV files in directory
- `process_jump_data` - Process jump data from input file
- `calc_fator_convert_mediapipe` - Calculate conversion factor from normalized to meters
- `calc_fator_convert_mediapipe_simple` - Simplified conversion factor calculation
- `calculate_cg_frame` - Calculate center of gravity for a frame

### Reporting:
- `generate_html_report` - Generate comprehensive HTML report with all metrics and plots
- `generate_normalized_diagnostic_plot` - Generate diagnostic plot showing normalized CG position

### Helper Functions:
- `draw_fppa_overlay` - Draw FPPA overlay on matplotlib axes (optional visualization)
- `_get_fppa_risk_classification` - Classify FPPA angle into risk categories
- `_format_fppa_with_risk` - Format FPPA angle with color coding for HTML report

## 📊 Key Metrics Explained

### FPPA (Frontal Plane Projection Angle)
The FPPA is calculated using a rigorous 2D vector-based method:
- **Vector v1 (Femur):** From HIP to KNEE
- **Vector v2 (Tibia):** From KNEE to ANKLE
- **Internal Angle:** Calculated at the knee joint using dot product and cross product
- **Convention:** 
  - 0° = straight alignment (180° internal angle)
  - Positive = Valgus (knee collapses medially/inward)
  - Negative = Varus (knee collapses laterally/outward)
- **Right Side:** Sign is inverted for consistent visualization (both sides show valgus as positive)

### Risk Classification
Based on scientific evidence (DOI: 10.1016/j.heliyon.2024):
- **< 5°:** Good Alignment (Green) - Low risk
- **5° - 10°:** Moderate Risk (Yellow) - Requires attention
- **> 10°:** High Risk / Excessive Dynamic Valgus (Red) - Significant risk factor

### Landing Phase Analysis
Based on "Mechanisms for Noncontact Anterior Cruciate Ligament Injury":
- **Initial Contact (IC):** Moment of landing
- **IC + 40ms:** Critical window for ACL injury risk
- **IC + 100ms:** Extended landing phase analysis
- **Max Valgus:** Maximum valgus angle within 0.2s post-landing window

## 📝 Usage

1. Run the script: `python vaila_and_jump.py`
2. Select the target directory containing .csv files
3. Choose data type:
   - **Option 1:** Time of Flight Data
   - **Option 2:** Jump Height Data
   - **Option 3:** MediaPipe Shank Length Data
4. For MediaPipe data, provide:
   - Subject mass (kg)
   - Video FPS (frames per second)
   - Shank length (m) for calibration
5. The script processes all files and generates comprehensive reports

## 📁 Input File Formats

### 1. Time-of-flight based format:
```
mass_kg,time_of_flight_s,contact_time_s
75.0,0.45,0.22
80.2,0.42,0.25
```

### 2. Jump-height based format:
```
mass_kg,height_m,contact_time_s
75.0,0.25,0.22
80.2,0.22,0.25
```

### 3. MediaPipe pose estimation format:
CSV file with MediaPipe pose landmark coordinates (frame_index, nose_x, nose_y, etc.)

## 🎯 Output

The script generates:
- **CSV files:** Jump metrics, calibrated data, database entries
- **Visualizations:** PNG plots of all analyses
- **HTML Report:** Comprehensive report with all metrics, plots, and risk classifications
- **Animated GIF:** Jump cycle animation (if imageio is available)

## 📚 Scientific References

- Santiago, P. R. P., et al. (2024). vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox. arXiv preprint arXiv:2410.07238
- Mechanisms for Noncontact Anterior Cruciate Ligament Injury (DOI: 10.1016/j.heliyon.2024)
- Samozino, P., et al. (2008). A simple method for measuring force, velocity and power output during squat jump. Journal of Biomechanics, 41(14), 2940-2945.

## ⚠️ Notes

- Ensure all necessary libraries are installed (pandas, numpy, matplotlib, tkinter)
- For accurate results, MediaPipe landmark detection should be of good quality
- Shank length calibration is critical for accurate metric calculations
- FPPA calculations use rigorous vector-based methods for clinical validity
- Right side FPPA values are inverted for consistent visualization (both sides show valgus as positive)

## 📄 License

This script is licensed under the GNU General Public License v3.0.

---

📅 **Last Updated:** 04 Jan 2026  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
