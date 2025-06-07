import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import csv
import os
from datetime import datetime
from ai import measurements

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
    def point(lm): return np.array([landmarks[lm].x * w, landmarks[lm].y * h])
    shoulder = np.linalg.norm(point(mp_pose.PoseLandmark.LEFT_SHOULDER) - point(mp_pose.PoseLandmark.RIGHT_SHOULDER))
    hip = np.linalg.norm(point(mp_pose.PoseLandmark.LEFT_HIP) - point(mp_pose.PoseLandmark.RIGHT_HIP))
    height = np.linalg.norm(point(mp_pose.PoseLandmark.NOSE) - point(mp_pose.PoseLandmark.RIGHT_ANKLE))
    return {
        "shoulder_width_cm": round(shoulder / 10, 2),
        "hip_width_cm": round(hip / 10, 2),
        "height_cm": round(height / 10, 2)
    }

# Helper: Save to CSV
def save_to_csv(name, measurements, image_filename):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Shoulder(cm)", "Hip(cm)", "Height(cm)", "Image", "Timestamp"])
        writer.writerow([
            name,
            measurements["shoulder_width_cm"],
            measurements["hip_width_cm"],
            measurements["height_cm"],
            image_filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

# Start Webcam
cap = cv2.VideoCapture(0)
frame_count = 0
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
        measurements = extract_measurements(landmarks, h, w)

        # Add current measurement to buffer
        stable_frames.append(measurements)
        if len(stable_frames) > required_stable_frames:
            stable_frames.pop(0)

        # Show current measurement
        y_pos = 30
        for key, value in measurements.items():
            cv2.putText(frame, f"{key}: {value} cm", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30

        # Check if measurements are stable (low variation)
        if len(stable_frames) == required_stable_frames:
            stds = {k: np.std([f[k] for f in stable_frames]) for k in measurements}
            if all(s < 1.0 for s in stds.values()):  # < 1cm variation is acceptable
                if not verified:
                    name = get_voice_input()
                    if name:
                        verified = True
                        avg_measurements = {
                            k: round(np.mean([f[k] for f in stable_frames]), 2)
                            for k in measurements
                        }

                        # Save image
                        image_filename = f"{IMAGE_DIR}/{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                        cv2.imwrite(image_filename, frame)

                        # Save data
                        save_to_csv(name, avg_measurements, image_filename)
                        stored = True
                        print("Measurement & image stored successfully.")

                        # Display confirmation
                        cv2.putText(frame, f"Saved for {name}", (10, y_pos + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if stored:
        cv2.putText(frame, "Process Complete. Press ESC to exit.", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AI Tailor - Live Measurement", frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
