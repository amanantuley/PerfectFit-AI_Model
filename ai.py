import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import csv
import os
import sqlite3
from datetime import datetime

# Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
recognizer = sr.Recognizer()

CSV_FILE = "measurements_data.csv"
IMAGE_DIR = "uploads"
DB_FILE = "measurements.db"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Get voice input
def get_voice_input():
    print("Please say your name for verification...")
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=5)
    try:
        name = recognizer.recognize_google(audio)
        print(f"Detected Name: {name}")
        return name
    except sr.UnknownValueError:
        return "Could not understand the name"
    except sr.RequestError:
        return "API unavailable"

# Measurement logic
def get_measurements(landmarks, h, w):
    def get_point(landmark):
        return np.array([landmarks[landmark].x * w, landmarks[landmark].y * h])
    shoulder = np.linalg.norm(get_point(mp_pose.PoseLandmark.LEFT_SHOULDER) - get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER))
    hip = np.linalg.norm(get_point(mp_pose.PoseLandmark.LEFT_HIP) - get_point(mp_pose.PoseLandmark.RIGHT_HIP))
    height = np.linalg.norm(get_point(mp_pose.PoseLandmark.NOSE) - get_point(mp_pose.PoseLandmark.RIGHT_ANKLE))
    return {
        "shoulder_width_cm": round(shoulder / 10, 2),
        "hip_width_cm": round(hip / 10, 2),
        "height_cm": round(height / 10, 2)
    }

# Save to CSV
def save_to_csv(name, measurements, image_file):
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "Shoulder(cm)", "Hip(cm)", "Height(cm)", "Image File", "Timestamp"])
        writer.writerow([
            name,
            measurements["shoulder_width_cm"],
            measurements["hip_width_cm"],
            measurements["height_cm"],
            image_file,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

# Save to SQLite
def save_to_db(name, measurements, image_file):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            shoulder REAL,
            hip REAL,
            height REAL,
            image TEXT,
            timestamp TEXT
        )
    ''')
    cursor.execute('''
        INSERT INTO measurements (name, shoulder, hip, height, image, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        name,
        measurements["shoulder_width_cm"],
        measurements["hip_width_cm"],
        measurements["height_cm"],
        image_file,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

# Camera loop
cap = cv2.VideoCapture(0)
verified = False
saved = False

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
        measurements = get_measurements(landmarks, h, w)

        y_pos = 30
        for key, value in measurements.items():
            cv2.putText(frame, f"{key}: {value} cm", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30

        if not verified:
            name = get_voice_input()
            if name.lower() not in ["could not understand the name", "api unavailable"]:
                verified = True

                # Save image
                image_filename = f"{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                cv2.imwrite(image_path, frame)

                # Save to CSV and DB
                save_to_csv(name, measurements, image_filename)
                save_to_db(name, measurements, image_filename)

                cv2.putText(frame, f"Verified: {name}", (10, y_pos + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                saved = True
            else:
                cv2.putText(frame, "Voice not recognized!", (10, y_pos + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if saved:
        cv2.putText(frame, "Captured & Saved", (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow('AI Tailor - Live Measurement + Storage', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
