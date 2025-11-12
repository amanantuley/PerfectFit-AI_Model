# ai.py
import cv2
import mediapipe as mp
import numpy as np
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.6)

# ------------------------- Utilities -------------------------
def auto_detect_a4(image):
    """
    Try to detect a rectangular A4 sheet in the image.
    Returns pixel_height_of_a4 or None if not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        eps = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area > best_area:
                best_area = area
                best = (approx, w, h)
    if best is None:
        return None
    _, w, h = best
    # A4 ratio ~ sqrt(2) ~ 1.414 (height/width depends on orientation)
    ratio = max(w, h) / (min(w, h) + 1e-6)
    if 1.2 < ratio < 1.6 and best_area > 5000:
        # prefer height if portrait orientation
        pixel_height = max(w, h)
        return pixel_height
    return None

def get_calibration_scale(image, allow_manual=True):
    """
    Attempt automatic A4 detection; if fails and allow_manual True, fallback to manual click.
    Returns scale_factor (cm per pixel).
    """
    # Try auto detect
    pix_h = auto_detect_a4(image)
    if pix_h:
        # A4 height is 29.7 cm (portrait) or 21.0 cm (landscape). We assume largest dimension corresponds to 29.7
        real_cm = 29.7
        scale = real_cm / pix_h
        print(f"Auto-calibration (A4) succeeded: {scale:.4f} cm/px")
        return scale

    if not allow_manual:
        return None

    # fallback to manual two-point click
    import matplotlib.pyplot as plt
    points = []
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            if len(points) == 2:
                plt.close()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Click two points on a calibration object (e.g., ruler or A4).")
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    if len(points) != 2:
        raise ValueError("Calibration requires two points.")
    px_dist = math.dist(points[0], points[1])
    if px_dist == 0:
        raise ValueError("Calibration points are identical.")
    real_length_cm = float(input("Enter real-world length (cm) between these points: "))
    scale_factor = real_length_cm / px_dist
    print(f"Manual calibration complete: {scale_factor:.4f} cm/px")
    return scale_factor

def euclidean_distance_px(p1, p2, landmarks, w, h):
    x1, y1 = landmarks[p1].x * w, landmarks[p1].y * h
    x2, y2 = landmarks[p2].x * w, landmarks[p2].y * h
    return math.dist((x1, y1), (x2, y2))

def landmark_visible(landmark, threshold=0.55):
    # MediaPipe landmark may have visibility field; treat missing as visible
    v = getattr(landmark, 'visibility', None)
    if v is None:
        return True
    return v >= threshold

# ------------------------- Main Measurement Function -------------------------
def measurements(image_path):
    """
    Compute body measurements (cm) using calibration and landmark checks.
    Returns (data_dict, recommended_size, scale_cm_per_px)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        raise ValueError("No person detected. Make sure full body is visible.")

    landmarks = results.pose_landmarks.landmark

    # Calibration
    print("Attempting automatic calibration (A4 detection). If that fails, you can manually calibrate.")
    scale_cm_per_px = get_calibration_scale(image)
    if scale_cm_per_px is None:
        raise RuntimeError("Calibration failed.")

    # Ensure required landmarks visible
    required = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
    ]
    for r in required:
        if not landmark_visible(landmarks[r]):
            raise ValueError(f"Landmark {r.name} not sufficiently visible. Retake image with full body visible and good lighting.")

    # Height: use top-most (min y) vs bottom-most (max y among ankles/feet)
    ys = [lm.y * h for lm in landmarks]
    top_y = min(ys)
    # find bottom using ankles, foot indices prefer if visible
    ankles = []
    for idx in (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX):
        try:
            if landmark_visible(landmarks[idx]):
                ankles.append(landmarks[idx].y * h)
        except:
            pass
    if not ankles:
        raise ValueError("No ankle/foot landmarks sufficiently visible.")
    bottom_y = max(ankles)
    height_px = bottom_y - top_y
    height_cm = round(height_px * scale_cm_per_px, 1)

    # Shoulder width
    shoulder_px = euclidean_distance_px(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, landmarks, w, h)
    shoulder_cm = round(shoulder_px * scale_cm_per_px, 1)

    # Hip width
    hip_px = euclidean_distance_px(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, landmarks, w, h)
    hip_cm = round(hip_px * scale_cm_per_px, 1)

    # Arms: average of left and right
    arm_left_px = euclidean_distance_px(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST, landmarks, w, h)
    arm_right_px = euclidean_distance_px(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_WRIST, landmarks, w, h)
    arm_cm = round(((arm_left_px + arm_right_px) / 2) * scale_cm_per_px, 1)

    # Legs: average left/right hip->ankle
    leg_left_px = euclidean_distance_px(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE, landmarks, w, h)
    leg_right_px = euclidean_distance_px(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE, landmarks, w, h)
    leg_cm = round(((leg_left_px + leg_right_px) / 2) * scale_cm_per_px, 1)

    neck_circ_cm = round((shoulder_px * 0.30 * scale_cm_per_px) * math.pi, 1)

    data = {
        "Height": height_cm,
        "Shoulder Width": shoulder_cm,
        "Hip Width": hip_cm,
        "Arm Length": arm_cm,
        "Leg Length": leg_cm,
        "Neck Circumference": neck_circ_cm
    }

    print("\nEstimated measurements (cm):")
    for k, v in data.items():
        print(f"  {k}: {v}")

    size = recommend_size(data)
    print(f"\nRecommended Size: {size}\n")

    return data, size, scale_cm_per_px

# ------------------------- AI-Based Size Recommendation -------------------------
def recommend_size(measurements):
    h = measurements["Height"]
    shoulder = measurements["Shoulder Width"]
    hip = measurements["Hip Width"]
    neck = measurements["Neck Circumference"]

    ratio_shoulder = shoulder / h * 100
    ratio_hip = hip / shoulder if shoulder > 0 else 1.0
    ratio_neck = neck / shoulder if shoulder > 0 else 0.5

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

    if ratio_hip > 1.25:
        size = "XL"
    elif ratio_neck > 1.15:
        size = "XXL"

    return size

# ------------------------- Direct Run -------------------------
if __name__ == "__main__":
    path = input("Enter full image path: ").strip()
    data, size, scale = measurements(path)
    print("Scale (cm/px):", scale)
    for k, v in data.items():
        print(f"{k}: {v} cm")
    print("Recommended Size:", size)
