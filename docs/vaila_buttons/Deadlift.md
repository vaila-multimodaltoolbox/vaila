# Biomechanical Specification Document for Computer Vision Pose Estimation: Deadlift & Romanian Deadlift (RDL)

This document establishes the technical, kinematic, and mathematical parameters required to configure a MediaPipe Pose Landmarker algorithm for validating and differentiating the Deadlift and the Romanian Deadlift (RDL).


---

## 1. MediaPipe Pose Landmark Mapping

The computer vision pipeline utilizes the following 33-landmark topology coordinates (X, Y, Z, Visibility) to compute spatial vectors:

* **Cervical / Head:** `LEFT_EAR` (7), `RIGHT_EAR` (8)
* **Shoulder Girdle:** `LEFT_SHOULDER` (11), `RIGHT_SHOULDER` (12)
* **Pelvic Girdle (Hips):** `LEFT_HIP` (23), `RIGHT_HIP` (24)
* **Upper Extremities (Load Proxy):** `LEFT_WRIST` (15), `RIGHT_WRIST` (16)
* **Lower Extremities:** `LEFT_KNEE` (25), `RIGHT_KNEE` (26), `LEFT_ANKLE` (27), `RIGHT_ANKLE` (28)

---

## 2. Mathematical Framework: Vectorization & Angular Tracking

Joint angles are computed in the 2D sagittal plane or 3D coordinate space using the dot product of two intersecting vectors or via the `atan2` function. For any joint center $B$ between endpoints $A$ and $C$:

```python
import math

def calculate_joint_angle(A, B, C):
    """
    Computes the absolute angle at vertex B given coordinates A, B, and C.
    Coordinates are passed as [x, y] or [x, y, z].
    """
    rad = math.atan2(C[1] - B[1], C[0] - B[0]) - math.atan2(A[1] - B[1], A[0] - B[0])
    angle = abs(math.degrees(rad))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle
```

3. Kinematic Parameters for Romanian Deadlift (RDL) Validation1️⃣ Hip-Width StanceBiomechanical Protocol: Feet must be placed strictly hip-width apart to ensure a vertical force vector through the sagittal plane and baseline stability. Sumo (wide) or adducted (touching) stances are restricted.Algorithmic Rule: The Euclidean distance along the X-axis between LEFT_ANKLE (27) and RIGHT_ANKLE (28) must match the distance between LEFT_HIP (23) and RIGHT_HIP (24).Threshold:$$\Delta = | \text{Dist}_{\text{Ankle}} - \text{Dist}_{\text{Hip}} | \le 0.10 \times \text{Dist}_{\text{Hip}}$$2️⃣ Bar Path Proximity (Scraping the Legs)Biomechanical Protocol: The barbell must remain in close proximity to the thighs and shins. Increasing the horizontal distance creates an unnecessary moment arm relative to the hip joint, massively increasing shear stress on the L4-S1 lumbar vertebrae.Algorithmic Rule: In a sagittal/profile view, the horizontal position (Z-axis or X-axis depending on camera orientation) of WRIST (15/16) must maintain a strict offset relative to the segment line bounding KNEE (25/26) and ANKLE (27/28).Threshold: Maximum horizontal displacement $\le 5\text{ cm}$ (normalized against femur length). Displace $> 7.5\text{ cm}$ flags a CRITICAL_WARNING: MOMENT_ARM_TOO_LARGE.3️⃣ Neutral Thoracolumbar SpineBiomechanical Protocol: Preservation of the natural physiological lordosis under axial load. Spinal flexion (rounding) or extreme hyperextension must be penalized.Algorithmic Rule: The tracking vector created from the midpoint of SHOULDER (11/12) to the midpoint of HIP (23/24) must maintain structural linearity.Threshold: Angular deviation within the thoracolumbar segment must be $< 5^\circ$ throughout both eccentric and concentric phases.4️⃣ Vertical Tibias (Shin Alignment)Biomechanical Protocol: To isolate the posterior chain (hamstrings and gluteus maximus) during the hip hinge, anterior translation of the knee must be restricted. Shins must remain vertical.Algorithmic Rule: Calculate the angle between the vector ANKLE $\rightarrow$ KNEE and the horizontal ground plane vector.Threshold: The angle must remain between $80^\circ$ and $90^\circ$ during the eccentric descent. Angles $< 80^\circ$ indicate faulty forward knee travel.5️⃣ Neutral Cervical SpineBiomechanical Protocol: The head and gaze must track smoothly with the inclination of the torso. Looking up fixedly at a mirror induces cervical hyperextension, stressing the upper trapezius.Algorithmic Rule: The vector passing through EAR (7/8) and SHOULDER (11/12) must remain strictly parallel to the torso vector (SHOULDER $\rightarrow$ HIP).Threshold: Relative angular divergence must stay within $0^\circ \pm 8^\circ$.6️⃣ Movement Initiation via Hip HingeBiomechanical Protocol: The eccentric phase must begin with the posterior translation of the pelvis (pushing hips back). Dropping the torso vertically first shifts the load completely onto the lower back.Algorithmic Rule: At the onset of the eccentric phase (triggered by a decrease in SHOULDER Y-position), the horizontal coordinate of the HIP landmark must exhibit an immediate backward vector translation.Threshold: Horizontal hip displacement must precede or happen simultaneously with vertical shoulder displacement.4. Algorithmic Differentiation MatrixWhen the tracking payload crosses below the patella plane (KNEE_LEVEL), the system uses these definitive thresholds to classify the exercise variation:Kinematic MetricStiff-Legged DeadliftRomanian Deadlift (RDL)Conventional DeadliftKnee Flexion Angle ($\theta_{\text{knee}}$)Near-total extension:$0^\circ \le \theta \le 5^\circ$Controlled, fixed flexion:$10^\circ \le \theta \le 15^\circ$High starting flexion:$75^\circ \le \theta \le 100^\circ$Shin-to-Ground AngleStrictly Vertical:$88^\circ - 90^\circ$Highly Vertical:$80^\circ - 90^\circ$Anteriorly Inclined:$60^\circ - 75^\circ$Relative Hip Height ($Y_{\text{hip}}$)Maximum Elevation (High Z/Y axis placement)Intermediate Position (Maintained above knee level)Low Position (Dropped close to or below knee level)Primary Muscle TargetHamstrings (distal emphasis) and Erector SpinaeGluteus Maximus and Hamstrings (proximal emphasis)Integrated Posterior Chain and Quadriceps5. Sample Integration Code block for LLM Engine

```
def classify_deadlift_variant(landmarks, img_width, img_height):
    # Normalized coordinate assignment
    hip = [landmarks[23].x, landmarks[23].y]
    knee = [landmarks[25].x, landmarks[25].y]
    ankle = [landmarks[27].x, landmarks[27].y]
    
    # Calculate key joint metrics
    knee_angle = calculate_joint_angle(hip, knee, ankle)
    
    # Measure shin vector relative to artificial horizontal ground [ankle_x + offset, ankle_y]
    shin_angle = calculate_joint_angle(knee, ankle, [ankle[0] + 0.1, ankle[1]])
    
    # Conditional logic classification tree
    if knee_angle < 5.0 and shin_angle >= 85.0:
        return "STIFF_LEGGED_DEADLIFT"
    elif 10.0 <= knee_angle <= 16.0 and 80.0 <= shin_angle <= 90.0:
        return "ROMANIAN_DEADLIFT_CORRECT"
    elif knee_angle > 45.0 and shin_angle < 78.0:
        return "CONVENTIONAL_DEADLIFT"
    else:
        return "UNKNOWN_OR_DEVIATING_FORM"
```

Kinematic Metric,Stiff-Legged Deadlift,Romanian Deadlift (RDL),Conventional Deadlift
Knee Flexion Angle (θknee​),Near-total extension:0∘≤θ≤5∘,"Controlled, fixed flexion:10∘≤θ≤15∘",High starting flexion:75∘≤θ≤100∘
Shin-to-Ground Angle,Strictly Vertical:88∘−90∘,Highly Vertical:80∘−90∘,Anteriorly Inclined:60∘−75∘
Relative Hip Height (Yhip​),Maximum Elevation (High Z/Y axis placement),Intermediate Position (Maintained above knee level),Low Position (Dropped close to or below knee level)
Primary Muscle Target,Hamstrings (distal emphasis) and Erector Spinae,Gluteus Maximus and Hamstrings (proximal emphasis),Integrated Posterior Chain and Quadriceps
