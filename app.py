from flask import Flask, render_template, request
import os
import sqlite3
import base64
from PIL import Image
from io import BytesIO
from ai import extract_measurements

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------
# Routes
# --------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['photo']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    result = extract_measurements(filepath)
    if "error" not in result:
        save_to_db(file.filename, result)
        return f"Measurements Saved: {result}"
    else:
        return result["error"]

@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    img_data = request.form['image']
    img_data = img_data.replace('data:image/png;base64,', '')
    img_bytes = base64.b64decode(img_data)

    img = Image.open(BytesIO(img_bytes))
    filepath = os.path.join(UPLOAD_FOLDER, 'captured.png')
    img.save(filepath)

    result = extract_measurements(filepath)
    if "error" not in result:
        save_to_db('captured.png', result)
        return f"Captured Measurements: {result}"
    else:
        return result["error"]

# --------------------
# Save to DB
# --------------------

def save_to_db(filename, data):
    conn = sqlite3.connect('measurements.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            shoulder_width REAL,
            hip_width REAL,
            height REAL
        )
    ''')
    c.execute("INSERT INTO users (filename, shoulder_width, hip_width, height) VALUES (?, ?, ?, ?)",
              (filename, data['shoulder_width_cm'], data['hip_width_cm'], data['height_cm']))
    conn.commit()
    conn.close()

# --------------------
# Run
# --------------------

if __name__ == '__main__':
    app.run(debug=True)
