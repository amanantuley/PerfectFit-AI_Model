import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_measurements(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        h, w, _ = image.shape
        landmarks = results.pose_landmarks.landmark

        shoulder_width = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w,
                                                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h]) -
                                        np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w,
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h]))

        hip_width = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w,
                                             landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h]) -
                                   np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * w,
                                             landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * h]))

        height = np.linalg.norm(np.array([landmarks[mp_pose.PoseLandmark.NOSE].x * w,
                                          landmarks[mp_pose.PoseLandmark.NOSE].y * h]) -
                                np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h]))

        return {
            "shoulder_width_cm": round(shoulder_width / 10, 2),
            "hip_width_cm": round(hip_width / 10, 2),
            "height_cm": round(height / 10, 2)
        }
    else:
        return {"error": "No body detected"}
