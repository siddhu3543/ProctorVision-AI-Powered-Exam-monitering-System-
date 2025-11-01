<<<<<<< HEAD
import cv2, time
from flask import Flask, render_template, Response, redirect, url_for, request, session
from flask_socketio import SocketIO
from flask_cors import CORS
from ultralytics import YOLO
import mediapipe as mp
import os

app = Flask(__name__)
app.secret_key = "exam_secret"
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load models
yolo = YOLO("yolov8n.pt")
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=5)

# ---------------- Routes ----------------

@app.route("/")
def rules_page():
    message = session.pop("message", None)
    return render_template("rules.html", message=message)

@app.route("/start_exam", methods=["POST"])
def start_exam():
    return redirect(url_for("student"))

@app.route("/student")
def student():
    return render_template("student.html")

@app.route("/end_exam", methods=["POST"])
def end_exam():
    session["message"] = "✅ Test Completed Successfully!"
    return redirect(url_for("rules_page"))

# ---------------- Video + Detection ----------------

def gen_frames():
    cap = cv2.VideoCapture(0)
    last_gaze_alert = 0
    last_phone_alert = 0
    last_face_alert = 0
    gaze_start = None
    gaze_direction = "Forward"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ----- 1️⃣ YOLO: Phone Detection -----
        phone_detected = False
        results = yolo.predict(frame, imgsz=320, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = r.names.get(cls, "")
                if "phone" in name or "cell phone" in name:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "PHONE", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        if phone_detected and time.time() - last_phone_alert > 3:
            print("Emitting phone alert")  # Debug print
            socketio.emit("alert", {
                "student_id": "student_1",
                "message": "📱 Mobile phone detected!",
                "timestamp": time.strftime("%H:%M:%S"),
                "type": "phone"
            })
            last_phone_alert = time.time()

        # ----- 2️⃣ Mediapipe: Face Detection -----
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)

        if res.multi_face_landmarks:
            face_count = len(res.multi_face_landmarks)

            # Multiple faces detected
            if face_count > 1 and time.time() - last_face_alert > 3:
                socketio.emit("alert", {
                    "student_id": "student_1",
                    "message": f"⚠️ Multiple people detected ({face_count})!",
                    "timestamp": time.strftime("%H:%M:%S"),
                    "type": "face"
                })
                last_face_alert = time.time()

            # Gaze tracking for first face
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            left_eye = (int(lm[33].x * w), int(lm[33].y * h))
            right_eye = (int(lm[263].x * w), int(lm[263].y * h))
            center = (lm[33].x + lm[263].x) / 2

            cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)

            direction = "Forward"
            if center < 0.43:
                direction = "Right"
            elif center > 0.57:
                direction = "Left"

            cv2.putText(frame, f"Gaze: {direction}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Gaze alert if not looking forward for >3s
            if direction != "Forward":
                if gaze_direction != direction:
                    gaze_direction = direction
                    gaze_start = time.time()
                elif time.time() - gaze_start > 3:
                    if time.time() - last_gaze_alert > 3:
                        socketio.emit("alert", {
                            "student_id": "student_1",
                            "message": f"👀 Looking {direction} for too long!",
                            "timestamp": time.strftime("%H:%M:%S"),
                            "type": "gaze"
                        })
                        last_gaze_alert = time.time()
            else:
                gaze_start = None
                gaze_direction = "Forward"

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- Main ----------------
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
    # Production
    else:
        app.run()
=======
import cv2, time, os, base64
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO
import mediapipe as mp

app = Flask(__name__)
socketio = SocketIO(app)

# Load models
yolo = YOLO("yolov8n.pt")
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

@app.route("/")
def student():
    return render_template("student.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Video streaming route
def gen_frames():
    cap = cv2.VideoCapture(0)
    last_gaze_alert = 0
    gaze_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. YOLO phone detection
        phone_detected = False
        results = yolo.predict(frame, imgsz=320, conf=0.4, verbose=False)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = r.names.get(cls, "")
                if "phone" in name or "cell phone" in name:
                    phone_detected = True
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, "PHONE", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if phone_detected:
            socketio.emit("alert", {
                "student_id": "student_1",
                "message": "📱 Phone detected!",
                "timestamp": time.strftime("%H:%M:%S")
            })

        # 2. Mediapipe face mesh (gaze detection)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_face.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            left_eye = (int(lm[33].x * w), int(lm[33].y * h))
            right_eye = (int(lm[263].x * w), int(lm[263].y * h))
            center = (lm[33].x + lm[263].x) / 2

            # Draw eyes
            cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)

            gaze_flag = False
            direction = "Forward"
            if center < 0.43:
                direction = "Right"
                gaze_flag = True
            elif center > 0.57:
                direction = "Left"
                gaze_flag = True

            cv2.putText(frame, f"Gaze: {direction}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            if gaze_flag:
                if gaze_start is None:
                    gaze_start = time.time()
                elif time.time() - gaze_start > 2.0:
                    if time.time() - last_gaze_alert > 5:
                        socketio.emit("alert", {
                            "student_id": "student_1",
                            "message": f"👀 Student looking {direction} too long!",
                            "timestamp": time.strftime("%H:%M:%S")
                        })
                        last_gaze_alert = time.time()
                        gaze_start = None
            else:
                gaze_start = None

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
>>>>>>> 3a36a026638dac90c29c6e3cc068d494f5a109ed
