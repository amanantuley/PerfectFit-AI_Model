# ai.py
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def get_calibration_scale(image):
    """
    Prompts user to click two points on the image corresponding to a real object of known length.
    Returns the scale factor (cm per pixel).
    """
    import matplotlib.pyplot as plt
    points = []
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            points.append((event.xdata, event.ydata))
            if len(points) == 2:
                plt.close()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Click two points on the calibration object (e.g., ruler)")
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    if len(points) != 2:
        raise ValueError("Calibration requires exactly two points.")
    px_dist = math.dist(points[0], points[1])
    real_length_cm = float(input("Enter the real-world length (in cm) between the two points: "))
    if px_dist == 0:
        raise ValueError("Calibration points are identical.")
    scale_factor = real_length_cm / px_dist
    print(f"Calibration complete: {scale_factor:.4f} cm per pixel.")
    return scale_factor

def measurements(image_path):
    """
    Calculates body measurements from a single full-body image.
    Returns measurements in centimeters.
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

    def pixel_distance(p1, p2):
        """Euclidean distance between two landmarks in pixels."""
        x1, y1 = int(landmarks[p1].x * w), int(landmarks[p1].y * h)
        x2, y2 = int(landmarks[p2].x * w), int(landmarks[p2].y * h)
        return math.dist((x1, y1), (x2, y2))


    # Calibration step for accurate scaling
    print("Calibration: Please ensure a real object of known length (e.g., ruler, A4 paper) is visible in the image.")
    scale_factor = get_calibration_scale(image)


    # Key measurements
    height_px = pixel_distance(mp_pose.PoseLandmark.NOSE,
                               mp_pose.PoseLandmark.LEFT_ANKLE)
    hip_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                  mp_pose.PoseLandmark.RIGHT_HIP)
    shoulder_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                       mp_pose.PoseLandmark.RIGHT_SHOULDER)
    neck_width_px = shoulder_width_px * 0.3  # approx. neck width
    neck_circumference_cm = (neck_width_px * scale_factor) * math.pi
    arm_length_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                   mp_pose.PoseLandmark.LEFT_WRIST)
    leg_length_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                   mp_pose.PoseLandmark.LEFT_ANKLE)

    # Convert to cm
    measurements_cm = {
        "Height": round(height_px * scale_factor, 1),
        "Neck Circumference": round(neck_circumference_cm, 1),
        "Shoulder Width": round(shoulder_width_px * scale_factor, 1),
        "Hip Width": round(hip_width_px * scale_factor, 1),
        "Arm Length": round(arm_length_px * scale_factor, 1),
        "Leg Length": round(leg_length_px * scale_factor, 1)
    }

    return measurements_cm
