import cv2
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import subprocess
import sys


app = Flask(__name__)

# Load models
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
model = load_model('drowsiness_model.h5')

# Variables for drowsiness detection
COUNTER = 0
ALERT_ACTIVE = False
TOTAL_BLINKS = 0
EYE_AR_THRESH = 0.3 # Threshold to determine a blink
EYE_AR_CONSEC_FRAMES = 10 # Number of consecutive frames the eye must be below the threshold to trigger an alert

# Track previous alert state to detect rising edge
PREV_ALERT_ACTIVE = False

def trigger_screen_flash():
    """Spawn a short-lived full-screen flash in a separate process."""
    try:
        flags = 0
        if sys.platform.startswith('win') and hasattr(subprocess, 'CREATE_NO_WINDOW'):
            flags = subprocess.CREATE_NO_WINDOW
        subprocess.Popen(
            [sys.executable, 'flash_screen.py'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags
        )
    except Exception:
        # Fail silently to avoid breaking the video loop
        pass

def generate_frames():
    cap = cv2.VideoCapture(0)
    global COUNTER, ALERT_ACTIVE, PREV_ALERT_ACTIVE
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Flip the frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

                eye_status = 'Open' # Default status
                if len(eyes) == 0:
                    eye_status = 'Closed'

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                    eye_roi = roi_color[ey:ey+eh, ex:ex+ew]

                    # Prepare image for model
                    final_image = cv2.resize(eye_roi, (224, 224))
                    final_image = np.expand_dims(final_image, axis=0)
                    final_image = final_image / 255.0

                    # Predict with the model
                    prediction = model.predict(final_image)
                    if prediction[0][0] > 0.5: # Assuming class 1 is 'Open'
                        eye_status = 'Open'
                    else:
                        eye_status = 'Closed'

                if eye_status == 'Closed':
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        ALERT_ACTIVE = True
                else:
                    COUNTER = 0
                    ALERT_ACTIVE = False

            # Rising-edge: fire system flash once when alert just became active
            if ALERT_ACTIVE and not PREV_ALERT_ACTIVE:
                try:
                    trigger_screen_flash()
                except Exception:
                    pass
            PREV_ALERT_ACTIVE = ALERT_ACTIVE

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert_status')
def alert_status():
    return jsonify(alert=ALERT_ACTIVE)

@app.route('/debug/alert_on')
def debug_alert_on():
    global ALERT_ACTIVE, PREV_ALERT_ACTIVE
    ALERT_ACTIVE = True
    if not PREV_ALERT_ACTIVE:
        try:
            trigger_screen_flash()
        except Exception:
            pass
    PREV_ALERT_ACTIVE = True
    return jsonify(status="on", alert=ALERT_ACTIVE)

@app.route('/debug/alert_off')
def debug_alert_off():
    global ALERT_ACTIVE
    ALERT_ACTIVE = False
    return jsonify(status="off", alert=ALERT_ACTIVE)

if __name__ == '__main__':
    app.run(debug=True)