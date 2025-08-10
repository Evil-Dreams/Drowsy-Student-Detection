import cv2
from flask import Flask, render_template, Response, jsonify


app = Flask(__name__)

# Load models
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Variables for drowsiness detection
COUNTER = 0
ALERT_ACTIVE = False
TOTAL_BLINKS = 0
EYE_AR_THRESH = 0.3 # Threshold to determine a blink
EYE_AR_CONSEC_FRAMES = 10 # Number of consecutive frames the eye must be below the threshold to trigger an alert

def generate_frames():
    cap = cv2.VideoCapture(0)
    global COUNTER, ALERT_ACTIVE
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
            eye_status = 'Closed' # Default to closed, change if eyes are found

            for (x, y, w, h) in faces:
                # We don't draw the face rectangle anymore
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

                if len(eyes) > 0:
                    eye_status = 'Open'
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

                if eye_status == 'Closed':
                    COUNTER += 1
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        ALERT_ACTIVE = True
                else:
                    COUNTER = 0
                    ALERT_ACTIVE = False



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

if __name__ == '__main__':
    app.run(debug=True)
