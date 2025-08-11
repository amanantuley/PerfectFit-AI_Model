import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import csv
import os
from datetime import datetime
import math

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
recognizer = sr.Recognizer()

# File to save measurements
CSV_FILE = "measurements_data.csv"
IMAGE_DIR = "captured_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Helper: Get voice input
def get_voice_input():
    print("Say your name for verification:")
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=5)
    try:
        name = recognizer.recognize_google(audio)
        print(f"Verified Name: {name}")
        return name
    except:
        return None

# Helper: Extract measurements
def extract_measurements(landmarks, h, w):
    def pixel_distance(p1, p2):
        """Distance between two points in pixels."""
        x1, y1 = landmarks[p1].x * w, landmarks[p1].y * h
        x2, y2 = landmarks[p2].x * w, landmarks[p2].y * h
        return math.dist((x1, y1), (x2, y2))

    # Shoulder width (reference for scale)
    shoulder_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                       mp_pose.PoseLandmark.RIGHT_SHOULDER)
    if shoulder_width_px == 0:
        return None

    # Assume average shoulder width = 40 cm
    scale_factor = 40 / shoulder_width_px

    height_px = pixel_distance(mp_pose.PoseLandmark.NOSE,
                               mp_pose.PoseLandmark.LEFT_ANKLE)
    hip_width_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                  mp_pose.PoseLandmark.RIGHT_HIP)
    neck_width_px = shoulder_width_px * 0.3
    neck_circumference_cm = (neck_width_px * scale_factor) * math.pi
    arm_length_px = pixel_distance(mp_pose.PoseLandmark.LEFT_SHOULDER,
                                   mp_pose.PoseLandmark.LEFT_WRIST)
    leg_length_px = pixel_distance(mp_pose.PoseLandmark.LEFT_HIP,
                                   mp_pose.PoseLandmark.LEFT_ANKLE)

    return {
        "Height(cm)": round(height_px * scale_factor, 1),
        "Neck Circumference(cm)": round(neck_circumference_cm, 1),
        "Shoulder Width(cm)": round(shoulder_width_px * scale_factor, 1),
        "Hip Width(cm)": round(hip_width_px * scale_factor, 1),
        "Arm Length(cm)": round(arm_length_px * scale_factor, 1),
        "Leg Length(cm)": round(leg_length_px * scale_factor, 1)
    }

# Helper: Save to CSV
def save_to_csv(name, measurements, image_filename):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name"] + list(measurements.keys()) + ["Image", "Timestamp"])
        writer.writerow([name] + list(measurements.values()) + [
            image_filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

# Start Webcam
cap = cv2.VideoCapture(0)
stable_frames = []
required_stable_frames = 10
verified = False
stored = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark
        current_measurements = extract_measurements(landmarks, h, w)

        if current_measurements:
            stable_frames.append(current_measurements)
            if len(stable_frames) > required_stable_frames:
                stable_frames.pop(0)

            # Show current measurement on screen
            y_pos = 30
            for key, value in current_measurements.items():
                cv2.putText(frame, f"{key}: {value}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30

            # Check if measurements are stable
            if len(stable_frames) == required_stable_frames:
                stds = {k: np.std([f[k] for f in stable_frames]) for k in current_measurements}
                if all(s < 1.0 for s in stds.values()):  # stable
                    if not verified:
                        name = get_voice_input()
                        if name:
                            verified = True
                            avg_measurements = {
                                k: round(np.mean([f[k] for f in stable_frames]), 2)
                                for k in current_measurements
                            }

                            # Save image
                            image_filename = f"{IMAGE_DIR}/{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(image_filename, frame)

                            # Save data
                            save_to_csv(name, avg_measurements, image_filename)
                            stored = True
                            print("Measurement & image stored successfully.")
                            cv2.putText(frame, f"Saved for {name}", (10, y_pos + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if stored:
        cv2.putText(frame, "Process Complete. Press ESC to exit.", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AI Tailor - Live Measurement", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
