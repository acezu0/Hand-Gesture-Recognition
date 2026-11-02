"""
Web-Based Interface for the Hand Recognition System.
Streams real-time video with gesture HUD and interactive web controls using Flask.
Run with: python web_app.py
Then open http://localhost:5000 in your browser.
"""

import time
import cv2
from flask import Flask, render_template_string, Response, jsonify, request

from hand_detector import HandDetector
from canvas import AirCanvas
from gesture_actions import GestureFilter, ActionController
from main import create_demo_frame

app = Flask(__name__)

# Global instances
detector = HandDetector(model_path="gesture_recognizer.task", num_hands=2, min_confidence=0.5)
canvas = AirCanvas(width=960, height=540)
gesture_filter = GestureFilter(history_len=5)
action_controller = ActionController(width=960, height=540)

# Global states
current_mode = 1  # 1: HUD, 2: Canvas, 3: Controller
show_hud = True
current_stats = {
    "fps": 30.0,
    "hands": 0,
    "gesture": "None",
    "handedness": "None",
    "fingers": 0,
    "pinch": False,
    "slider": 50
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Hand Recognition Dashboard</title>
    <style>
        :root {
            --bg: #0f111a;
            --card-bg: #1a1d2e;
            --accent: #00e5ff;
            --accent-glow: rgba(0, 229, 255, 0.3);
            --text: #e0e6ed;
            --text-dim: #8b9bb4;
            --border: #2a2f4c;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 20px; }
        .container { max-width: 1300px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 15px; }
        h1 { font-size: 1.6rem; color: var(--accent); text-shadow: 0 0 10px var(--accent-glow); display: flex; align-items: center; gap: 10px; }
        .badge { background: #262c49; color: #a5b4fc; padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .video-box { background: #000; border: 2px solid var(--border); border-radius: 12px; overflow: hidden; position: relative; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .video-box img { width: 100%; height: auto; display: block; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; margin-bottom: 18px; }
        .card h2 { font-size: 1.1rem; margin-bottom: 14px; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 8px; }
        .btn-group { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
        button { background: #232845; color: var(--text); border: 1px solid var(--border); padding: 8px 14px; border-radius: 8px; cursor: pointer; transition: 0.2s; font-size: 0.85rem; font-weight: 500; }
        button:hover { background: #323961; border-color: var(--accent); }
        button.active { background: var(--accent); color: #000; font-weight: bold; box-shadow: 0 0 12px var(--accent-glow); }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat-item { background: #131625; padding: 10px 14px; border-radius: 8px; border-left: 3px solid var(--accent); }
        .stat-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-val { font-size: 1.2rem; font-weight: bold; margin-top: 4px; color: #fff; }
        .guide-item { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; }
        .guide-item span:first-child { color: #94a3b8; }
        .guide-item span:last-child { color: var(--accent); font-weight: 500; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🖐️ AI Hand Recognition System <span class="badge">Live Stream</span></h1>
            <div>MediaPipe Tasks API + OpenCV</div>
        </header>

        <div class="grid">
            <div>
                <div class="video-box">
                    <img src="/video_feed" alt="Live Stream" />
                </div>
            </div>

            <div>
                <div class="card">
                    <h2>Interaction Modes</h2>
                    <div class="btn-group">
                        <button id="btn-mode1" class="active" onclick="setMode(1)">1: HUD Recognition</button>
                        <button id="btn-mode2" onclick="setMode(2)">2: Air Canvas</button>
                        <button id="btn-mode3" onclick="setMode(3)">3: Controller</button>
                    </div>
                    <div class="btn-group">
                        <button onclick="toggleHud()">Toggle Skeleton HUD</button>
                        <button onclick="clearCanvas()">Clear Canvas</button>
                    </div>
                </div>

                <div class="card">
                    <h2>Live Detection Telemetry</h2>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-label">Recognized Gesture</div>
                            <div class="stat-val" id="stat-gesture" style="color: #4ade80;">--</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Handedness</div>
                            <div class="stat-val" id="stat-hand">--</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Fingers Up</div>
                            <div class="stat-val" id="stat-fingers">0 / 5</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-label">Pinch State</div>
                            <div class="stat-val" id="stat-pinch">No</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Gesture Reference Guide</h2>
                    <div class="guide-item"><span>Pointing (1 Finger)</span><span>Draw in Canvas</span></div>
                    <div class="guide-item"><span>Peace / Victory</span><span>Play / Pause</span></div>
                    <div class="guide-item"><span>Thumbs Up / Down</span><span>Volume +/-</span></div>
                    <div class="guide-item"><span>Pinch & Slide</span><span>Adjust Slider</span></div>
                    <div class="guide-item"><span>Rock / Horns</span><span>Mute Audio</span></div>
                    <div class="guide-item"><span>Open Palm</span><span>Stop / Erase</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setMode(mode) {
            fetch('/api/set_mode?mode=' + mode).then(() => {
                document.querySelectorAll('#btn-mode1, #btn-mode2, #btn-mode3').forEach((btn, idx) => {
                    btn.classList.toggle('active', (idx + 1) === mode);
                });
            });
        }
        function toggleHud() { fetch('/api/toggle_hud'); }
        function clearCanvas() { fetch('/api/clear_canvas'); }

        // Live stats polling
        setInterval(() => {
            fetch('/api/stats').then(r => r.json()).then(data => {
                document.getElementById('stat-gesture').innerText = data.gesture;
                document.getElementById('stat-hand').innerText = data.handedness;
                document.getElementById('stat-fingers').innerText = data.fingers + ' / 5';
                document.getElementById('stat-pinch').innerText = data.pinch ? 'YES' : 'NO';
                document.getElementById('stat-pinch').style.color = data.pinch ? '#f87171' : '#94a3b8';
            }).catch(() => {});
        }, 200);
    </script>
</body>
</html>
"""

def generate_frames():
    global current_stats, current_mode, show_hud
    cap = cv2.VideoCapture(0)
    is_demo = not cap.isOpened()
    if is_demo:
        print("[Web App] Physical webcam not available. Using dynamic demo mode.")

    w, h = 960, 540
    frame_counter = 0

    while True:
        frame_counter += 1
        if is_demo:
            frame = create_demo_frame(frame_counter, width=w, height=h)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = create_demo_frame(frame_counter, width=w, height=h)
            else:
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (w, h))

        # Detect
        hands = detector.detect(frame)
        primary_gesture = hands[0]["gesture"] if hands else "None"
        stable_gesture = gesture_filter.update(primary_gesture)

        # Update stats
        if hands:
            current_stats["hands"] = len(hands)
            current_stats["gesture"] = stable_gesture
            current_stats["handedness"] = hands[0]["handedness"]
            current_stats["fingers"] = hands[0]["finger_count"]
            current_stats["pinch"] = hands[0]["pinch"]["is_pinch"]
        else:
            current_stats["hands"] = 0
            current_stats["gesture"] = "None"
            current_stats["handedness"] = "None"
            current_stats["fingers"] = 0
            current_stats["pinch"] = False

        # Mode execution
        if current_mode == 1:
            if show_hud:
                frame = detector.draw_landmarks(frame, hands, draw_hud=True)
        elif current_mode == 2:
            canvas.update(hands)
            frame = canvas.draw_ui(frame)
            if show_hud:
                frame = detector.draw_landmarks(frame, hands, draw_hud=False)
        elif current_mode == 3:
            action_controller.update(hands, stable_gesture)
            if show_hud:
                frame = detector.draw_landmarks(frame, hands, draw_hud=True)
            frame = action_controller.draw_ui(frame)

        # Encode JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def get_stats():
    return jsonify(current_stats)


@app.route('/api/set_mode')
def set_mode():
    global current_mode
    mode = int(request.args.get('mode', 1))
    current_mode = mode
    return jsonify({"success": True, "mode": current_mode})


@app.route('/api/toggle_hud')
def toggle_hud():
    global show_hud
    show_hud = not show_hud
    return jsonify({"success": True, "show_hud": show_hud})


@app.route('/api/clear_canvas')
def clear_canvas_api():
    canvas.clear()
    return jsonify({"success": True})


if __name__ == '__main__':
    print("\n=======================================================")
    print("  Starting Hand Recognition Web Server on port 5000...  ")
    print("  Open: http://localhost:5000 in your browser          ")
    print("=======================================================\n")
    app.run(host='0.0.0.0', port=5000, threaded=True)

