# ai.py
import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

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

    # Reference: Shoulder width in pixels
    shoulder_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                       mp_pose.PoseLandmark.RIGHT_SHOULDER)
    if shoulder_width_px == 0:
        raise ValueError("Invalid shoulder width detected.")

    # Scaling — assume average shoulder width is 40 cm
    scale_factor = 40 / shoulder_width_px

    # Key measurements
    height_px = pixel_distance(mp_pose.PoseLandmark.NOSE,
                               mp_pose.PoseLandmark.LEFT_ANKLE)

    hip_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                  mp_pose.PoseLandmark.RIGHT_HIP)

    neck_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                   mp_pose.PoseLandmark.RIGHT_SHOULDER) * 0.3  # approx. neck width
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
