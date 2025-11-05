import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime
import math

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Storage
CSV_FILE = "perfectfit_measurements.csv"
IMAGE_DIR = "captured_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# ------------------------- Helper Functions -------------------------

def get_name_input():
    name = input("Enter your name for verification: ").strip()
    return name if name else "Anonymous"


def pixel_distance(p1, p2, landmarks, w, h):
    """Distance between two landmarks in pixels."""
    x1, y1 = landmarks[p1].x * w, landmarks[p1].y * h
    x2, y2 = landmarks[p2].x * w, landmarks[p2].y * h
    return math.dist((x1, y1), (x2, y2))


def extract_measurements(landmarks, h, w):
    """Calculate real-world body measurements from landmarks."""
    shoulder_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, landmarks, w, h)
    if shoulder_px == 0:
        return None

    # Assume average shoulder = 40 cm for scale calibration
    scale_factor = 40 / shoulder_px

    height_px = pixel_distance(mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_ANKLE, landmarks, w, h)
    hip_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, landmarks, w, h)
    arm_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_WRIST, landmarks, w, h)
    leg_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE, landmarks, w, h)
    neck_circum_cm = (shoulder_px * 0.3 * scale_factor) * math.pi

    return {
        "Height(cm)": round(height_px * scale_factor, 1),
        "Shoulder(cm)": round(shoulder_px * scale_factor, 1),
        "Hip(cm)": round(hip_px * scale_factor, 1),
        "Arm(cm)": round(arm_px * scale_factor, 1),
        "Leg(cm)": round(leg_px * scale_factor, 1),
        "Neck(cm)": round(neck_circum_cm, 1)
    }


def recommend_size(meas):
    """Predicts clothing size using proportional ratios and height thresholds."""
    height = meas["Height(cm)"]
    shoulder = meas["Shoulder(cm)"]
    hip = meas["Hip(cm)"]
    neck = meas["Neck(cm)"]

    shoulder_ratio = (shoulder / height) * 100
    hip_ratio = hip / shoulder
    neck_ratio = neck / shoulder

    # Core anthropometric logic (based on global size standards)
    if height < 160 or shoulder_ratio < 16.5:
        size = "S"
    elif 160 <= height < 170 and 16.5 <= shoulder_ratio < 18.0:
        size = "M"
    elif 170 <= height < 178 and 18.0 <= shoulder_ratio < 19.0:
        size = "L"
    elif 178 <= height < 186 and 19.0 <= shoulder_ratio < 20.5:
        size = "XL"
    elif height >= 186 or shoulder_ratio >= 20.5:
        size = "XXL"
    else:
        size = "M"

    # Fine-tuning using body proportions
    if hip_ratio > 1.25:
        size = "XL"
    elif neck_ratio > 1.15 and size != "XXL":
        size = "XL"

    return size


def save_to_csv(name, measurements, size, image_filename):
    """Store data in CSV file."""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name"] + list(measurements.keys()) + ["Predicted Size", "Image", "Timestamp"])
        writer.writerow([name] + list(measurements.values()) + [
            size,
            image_filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])


# ------------------------- Main Camera Loop -------------------------

def main():
    cap = cv2.VideoCapture(0)
    stable_frames = []
    required_stable_frames = 12
    saved = False

    print("\n🧠 PerfectFit AI — Live Measurement + Size Recommender")
    print("Stand straight in full view. Stay still for 3 seconds...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            meas = extract_measurements(landmarks, h, w)

            if meas:
                stable_frames.append(meas)
                if len(stable_frames) > required_stable_frames:
                    stable_frames.pop(0)

                y = 30
                for k, v in meas.items():
                    cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    y += 25

                if len(stable_frames) == required_stable_frames:
                    stds = {k: np.std([f[k] for f in stable_frames]) for k in meas}
                    if all(s < 1.0 for s in stds.values()):  # stable
                        avg = {k: round(np.mean([f[k] for f in stable_frames]), 2) for k in meas}
                        size = recommend_size(avg)
                        y += 20
                        cv2.putText(frame, f"Recommended Size: {size}", (10, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                        if not saved:
                            name = get_name_input()
                            image_filename = f"{IMAGE_DIR}/{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(image_filename, frame)
                            save_to_csv(name, avg, size, image_filename)
                            saved = True
                            print(f"\n✅ Data saved for {name}: {size}")
                            cv2.putText(frame, "Saved successfully!", (10, y + 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if saved:
            cv2.putText(frame, "Press ESC to exit", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        cv2.imshow("PerfectFit AI", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
