# ai.py
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.6)

def get_calibration_scale(image):
    """
    Prompts user to click two points on a real-world object of known size (e.g., ruler).
    Returns scale factor: cm per pixel.
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

    real_length_cm = float(input("Enter real-world length (in cm) between the two points: "))
    scale_factor = real_length_cm / px_dist
    print(f"Calibration complete → {scale_factor:.4f} cm per pixel")
    return scale_factor


def euclidean_distance(p1, p2, landmarks, w, h):
    """3D distance between two landmarks (x,y,z)."""
    x1, y1, z1 = landmarks[p1].x * w, landmarks[p1].y * h, landmarks[p1].z * w
    x2, y2, z2 = landmarks[p2].x * w, landmarks[p2].y * h, landmarks[p2].z * w
    return math.dist((x1, y1, z1), (x2, y2, z2))


def measurements(image_path):
    """
    Calculates body measurements from a single full-body image.
    Returns dictionary of measurements in centimeters.
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No person detected in the image.")

    landmarks = results.pose_landmarks.landmark

    # --- Calibration ---
    print("Calibration: Ensure a known object (e.g., ruler, A4 paper) is visible in the image.")
    scale_factor = get_calibration_scale(image)

    # --- Key Measurements (using both sides & 3D distances) ---
    height_px = euclidean_distance(mp_pose.PoseLandmark.NOSE,
                                   mp_pose.PoseLandmark.LEFT_ANKLE,
                                   landmarks, w, h)

    shoulder_width_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                           mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                           landmarks, w, h)

    hip_width_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                      mp_pose.PoseLandmark.RIGHT_HIP,
                                      landmarks, w, h)

    # Arm length (average of left & right)
    arm_left_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                     mp_pose.PoseLandmark.LEFT_WRIST,
                                     landmarks, w, h)
    arm_right_px = euclidean_distance(mp_pose.PoseLandmark.RIGHT_SHOULDER,
                                      mp_pose.PoseLandmark.RIGHT_WRIST,
                                      landmarks, w, h)
    arm_length_px = (arm_left_px + arm_right_px) / 2

    # Leg length (average of left & right)
    leg_left_px = euclidean_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                     mp_pose.PoseLandmark.LEFT_ANKLE,
                                     landmarks, w, h)
    leg_right_px = euclidean_distance(mp_pose.PoseLandmark.RIGHT_HIP,
                                      mp_pose.PoseLandmark.RIGHT_ANKLE,
                                      landmarks, w, h)
    leg_length_px = (leg_left_px + leg_right_px) / 2

    # Approximate neck circumference from shoulder width
    neck_width_px = shoulder_width_px * 0.30
    neck_circumference_cm = (neck_width_px * scale_factor) * math.pi

    # --- Convert to cm ---
    measurements_cm = {
        "Height": round(height_px * scale_factor, 1),
        "Neck Circumference": round(neck_circumference_cm, 1),
        "Shoulder Width": round(shoulder_width_px * scale_factor, 1),
        "Hip Width": round(hip_width_px * scale_factor, 1),
        "Arm Length": round(arm_length_px * scale_factor, 1),
        "Leg Length": round(leg_length_px * scale_factor, 1)
    }

    return measurements_cm
