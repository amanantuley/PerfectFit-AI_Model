from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import base64
import mediapipe as mp
from ai import extract_measurements
app = Flask(__name__)

# Folder to save uploaded/captured images
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mp_pose = mp.solutions.pose

def process_image(image_path, user_height_cm):
    # Read image from path
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Pose for landmark detection
    with mp_pose.Pose(static_image_mode=True) as pose:
        result = pose.process(image_rgb)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark

        # Get key landmarks
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

        image_height, image_width, _ = image.shape

        def denorm(pt):
            # Convert normalized [0,1] landmark to pixel coords
            return int(pt.x * image_width), int(pt.y * image_height)

        ls_px, rs_px = denorm(left_shoulder), denorm(right_shoulder)
        lh_px, rh_px = denorm(left_hip), denorm(right_hip)
        le_px, la_px = denorm(left_ear), denorm(left_ankle)

        # Calculate pixel distances
        shoulder_width_px = np.linalg.norm(np.array(ls_px) - np.array(rs_px))
        hip_width_px = np.linalg.norm(np.array(lh_px) - np.array(rh_px))
        torso_height_px = np.linalg.norm(np.array(le_px) - np.array(la_px))

        # Prevent division by zero
        if torso_height_px == 0:
            return None

        # Calculate scale from user height in cm to torso pixel height
        scale = user_height_cm / torso_height_px

        # Convert pixel measurements to cm
        return {
            'shoulder_width_cm': round(shoulder_width_px * scale, 2),
            'hip_width_cm': round(hip_width_px * scale, 2),
            'torso_height_cm': round(torso_height_px * scale, 2)
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files or request.files['photo'].filename == '':
        return "No photo uploaded."

    photo = request.files['photo']

    # Validate height input
    try:
        user_height_cm = float(request.form['height'])
        if user_height_cm <= 0:
            return "Please enter a valid height greater than zero."
    except (ValueError, KeyError):
        return "Please enter a valid height in centimeters."

    path = os.path.join(app.config['UPLOAD_FOLDER'], photo.filename)
    photo.save(path)

    measurements = process_image(path, user_height_cm)
    if not measurements:
        return "Unable to detect human body properly. Try another image."

    return render_template('results.html', measurements=measurements)

@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    image_data = request.form.get('image')
    try:
        user_height_cm = float(request.form['height'])
        if user_height_cm <= 0:
            return "Please enter a valid height greater than zero."
    except (ValueError, KeyError):
        return "Please enter a valid height in centimeters."

    if not image_data:
        return "No image data received."

    # Decode base64 image
    header, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return "Failed to decode image data."

    filename = 'captured.jpg'
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cv2.imwrite(path, image)

    measurements = process_image(path, user_height_cm)
    if not measurements:
        return "Unable to detect human body properly. Try again."

    return render_template('results.html', measurements=measurements)

if __name__ == '__main__':
    app.run(debug=True)
