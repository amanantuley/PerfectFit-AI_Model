# ai.py
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6)

# ------------------------- Calibration Helper -------------------------
def get_calibration_scale(image):
    """
    Prompts user to click two points on a real object of known length (e.g., ruler or A4 paper edge)
    and returns scale factor = cm per pixel.
    """
    import matplotlib.pyplot as plt
    points = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            if len(points) == 2:
                plt.close()

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Click two points on calibration object (e.g., ruler, A4 paper)")
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(points) != 2:
        raise ValueError("Calibration requires exactly two points.")

    px_dist = math.dist(points[0], points[1])
    if px_dist == 0:
        raise ValueError("Calibration points are identical.")

    real_length_cm = float(input("Enter real-world length (in cm) between these points: "))
    scale_factor = real_length_cm / px_dist
    print(f"✅ Calibration complete → {scale_factor:.4f} cm per pixel")
    return scale_factor


# ------------------------- Distance Helper -------------------------
def euclidean_distance(p1, p2, landmarks, w, h):
    """3D Euclidean distance between two pose landmarks."""
    x1, y1, z1 = landmarks[p1].x * w, landmarks[p1].y * h, landmarks[p1].z * w
    x2, y2, z2 = landmarks[p2].x * w, landmarks[p2].y * h, landmarks[p2].z * w
    return math.dist((x1, y1, z1), (x2, y2, z2))


# ------------------------- Main Measurement Function -------------------------
def measurements(image_path):
    """
    Calculates body measurements from a single full-body image and returns a dictionary in cm.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No person detected in the image. Make sure the full body is visible.")

    landmarks = results.pose_landmarks.landmark

    # --- Calibration ---
    print("📏 Calibration: Ensure a known-size object (e.g., ruler, A4 paper) is visible.")
    scale_factor = get_calibration_scale(image)

    # --- Key Measurements (3D distances) ---
    height_px = euclidean_distance(mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_ANKLE, landmarks, w, h)
    shoulder_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, landmarks, w, h)
    hip_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, landmarks, w, h)

    arm_left_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST, landmarks, w, h)
    arm_right_px = euclidean_distance(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST, landmarks, w, h)
    leg_left_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE, landmarks, w, h)
    leg_right_px = euclidean_distance(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE, landmarks, w, h)

    neck_width_px = shoulder_px * 0.30
    neck_circumference_cm = (neck_width_px * scale_factor) * math.pi

    # --- Convert to cm ---
    data = {
        "Height": round(height_px * scale_factor, 1),
        "Shoulder Width": round(shoulder_px * scale_factor, 1),
        "Hip Width": round(hip_px * scale_factor, 1),
        "Arm Length": round(((arm_left_px + arm_right_px) / 2) * scale_factor, 1),
        "Leg Length": round(((leg_left_px + leg_right_px) / 2) * scale_factor, 1),
        "Neck Circumference": round(neck_circumference_cm, 1)
    }

    print("\n📐 Estimated Body Measurements (in cm):")
    for k, v in data.items():
        print(f"   {k}: {v}")

    # --- Predict Clothing Size ---
    size = recommend_size(data)
    print(f"\n👕 Recommended Size: {size}\n")

    return data, size


# ------------------------- AI-Based Size Recommendation -------------------------
def recommend_size(measurements):
    """
    Predicts clothing size (S, M, L, XL, XXL, XXXL) using anthropometric logic and body ratios.
    """
    h = measurements["Height"]
    shoulder = measurements["Shoulder Width"]
    hip = measurements["Hip Width"]
    neck = measurements["Neck Circumference"]

    # Derived ratios
    ratio_shoulder = shoulder / h * 100
    ratio_hip = hip / shoulder
    ratio_neck = neck / shoulder

    # Intelligent logic (based on ISO/ASTM size mapping)
    if h < 160 or ratio_shoulder < 16.5:
        size = "S"
    elif 160 <= h < 170 and 16.5 <= ratio_shoulder < 18.0:
        size = "M"
    elif 170 <= h < 178 and 18.0 <= ratio_shoulder < 19.0:
        size = "L"
    elif 178 <= h < 186 and 19.0 <= ratio_shoulder < 20.5:
        size = "XL"
    elif 186 <= h < 195 or ratio_shoulder >= 20.5:
        size = "XXL"
    else:
        size = "M"

    # Adjustments for proportional body balance
    if ratio_hip > 1.25:
        size = "XL"
    elif ratio_neck > 1.15:
        size = "XXL"

    return size


# ------------------------- Direct Run (for quick test) -------------------------
if __name__ == "__main__":
    path = input("Enter full image path: ").strip()
    body_data, rec_size = measurements(path)
    print("\n✅ Final Output:")
    for k, v in body_data.items():
        print(f"{k}: {v} cm")
    print(f"Recommended Size: {rec_size}")
