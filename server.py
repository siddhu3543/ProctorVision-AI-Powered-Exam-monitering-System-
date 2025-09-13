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
