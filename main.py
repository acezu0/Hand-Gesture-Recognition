"""
Real-Time Hand Recognition and Gesture Interaction System.
Supports:
- Mode 1: Hand & Gesture Diagnostic HUD (21 landmarks, handedness, confidence, finger status)
- Mode 2: Virtual Air Canvas (Fingertip air drawing, colors, brush sizing, eraser)
- Mode 3: Gesture Action Controller (Virtual slider, media controls, event triggers)

Controls:
- '1', '2', '3': Switch Modes
- 'h': Toggle Landmark HUD
- 'c': Clear Canvas
- 's': Save Snapshot Screenshot
- 'm': Toggle Camera Mirroring
- 'q' or ESC: Exit
"""

import os
import sys
import time
import math
import argparse
import cv2
import numpy as np

from hand_detector import HandDetector
from canvas import AirCanvas
from gesture_actions import GestureFilter, ActionController


def create_demo_frame(frame_num, width=1280, height=720):
    """
    Generates an animated synthetic hand frame for demo/testing when no physical webcam is connected.
    Simulates natural hand movement, finger states, and gestures over time.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient tech background
    for y in range(0, height, 4):
        val = int(20 + 25 * (y / height))
        frame[y:y+4, :] = (val // 2, val // 3, val)

    # Grid overlay
    for x in range(0, width, 80):
        cv2.line(frame, (x, 0), (x, height), (35, 30, 45), 1)
    for y in range(0, height, 80):
        cv2.line(frame, (0, y), (width, y), (35, 30, 45), 1)

    t = frame_num * 0.05
    cx = int(width * 0.45 + math.sin(t * 0.8) * (width * 0.15))
    cy = int(height * 0.55 + math.cos(t * 0.6) * (height * 0.1))

    # Cycle through gesture states every 100 frames
    phase = (frame_num // 80) % 4

    # Palm base
    cv2.circle(frame, (cx, cy), 55, (220, 190, 170), -1)

    # Fingers
    # phase 0: Open Palm (all 5 fingers up)
    # phase 1: Pointing (Index up, others curled)
    # phase 2: Peace / Victory (Index + Middle up)
    # phase 3: Pinch (Thumb + Index close together)

    finger_configs = [
        # (angle_offset, length, is_extended)
        (-0.7, 85, True if phase in (0, 3) else False),               # Thumb
        (-0.35, 110, True if phase in (0, 1, 2, 3) else False),        # Index
        (0.0, 120, True if phase in (0, 2) else False),               # Middle
        (0.35, 105, True if phase == 0 else False),                   # Ring
        (0.7, 90, True if phase == 0 else False),                     # Pinky
    ]

    for angle_offset, length, is_ext in finger_configs:
        curr_len = length if is_ext else 40
        # If pinch mode, bring thumb and index tips close
        if phase == 3 and angle_offset in (-0.7, -0.35):
            curr_len = 80
            angle = -0.52 + (0.1 if angle_offset == -0.7 else -0.1)
        else:
            angle = -math.pi / 2 + angle_offset

        tip_x = int(cx + curr_len * math.cos(angle))
        tip_y = int(cy + curr_len * math.sin(angle))
        cv2.line(frame, (cx, cy - 20), (tip_x, tip_y), (220, 190, 170), 22, cv2.LINE_AA)
        cv2.circle(frame, (tip_x, tip_y), 12, (220, 190, 170), -1, cv2.LINE_AA)

    # Demo Banner at Top
    cv2.rectangle(frame, (width // 2 - 280, 15), (width // 2 + 280, 55), (40, 20, 80), -1)
    cv2.rectangle(frame, (width // 2 - 280, 15), (width // 2 + 280, 55), (0, 200, 255), 2)
    cv2.putText(
        frame,
        "DEMO SIMULATION MODE (No Camera Attached)",
        (width // 2 - 260, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return frame


def render_header(frame, mode, fps, hands_count, is_mirror):
    """Draws top information banner across all modes."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (18, 18, 22), -1)
    cv2.line(frame, (0, 50), (w, 50), (60, 60, 75), 1)

    modes = [
        ("1: Recognition HUD", mode == 1),
        ("2: Air Canvas", mode == 2),
        ("3: Controller", mode == 3),
    ]

    # Mode Selector Badges
    x_offset = 20
    for label, is_active in modes:
        badge_w = len(label) * 9 + 20
        bg_col = (0, 140, 255) if is_active else (35, 35, 45)
        text_col = (255, 255, 255) if is_active else (160, 160, 160)
        cv2.rectangle(frame, (x_offset, 10), (x_offset + badge_w, 40), bg_col, -1)
        if is_active:
            cv2.rectangle(frame, (x_offset, 10), (x_offset + badge_w, 40), (255, 255, 255), 1)
        cv2.putText(frame, label, (x_offset + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_col, 1, cv2.LINE_AA)
        x_offset += badge_w + 12

    # Stats on the right
    stats_text = f"FPS: {fps:.1f}  |  Hands: {hands_count}  |  Mirror: {'ON' if is_mirror else 'OFF'}  |  [Q]uit"
    (tw, _), _ = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(frame, stats_text, (w - tw - 20, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="AI Hand Recognition & Gesture Control System")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file (optional)")
    parser.add_argument("--demo", action="store_true", help="Force synthetic demo simulation mode")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height (default: 720)")
    parser.add_argument("--model", type=str, default="gesture_recognizer.task", help="Path to gesture recognizer model")
    args = parser.parse_args()

    print("=======================================================")
    print("      HAND RECOGNITION & GESTURE CONTROL SYSTEM        ")
    print("=======================================================")
    print(f"Loading MediaPipe model: {args.model}...")

    try:
        detector = HandDetector(model_path=args.model, num_hands=2, min_confidence=0.5)
        print("✓ Hand Detector initialized successfully.")
    except Exception as e:
        print(f"Error initializing Hand Detector: {e}")
        sys.exit(1)

    # Initialize Modules
    canvas = AirCanvas(width=args.width, height=args.height)
    gesture_filter = GestureFilter(history_len=5)
    action_controller = ActionController(width=args.width, height=args.height)

    # Camera / Video Initialization
    cap = None
    is_demo = args.demo

    if not is_demo and args.video:
        if os.path.exists(args.video):
            cap = cv2.VideoCapture(args.video)
            print(f"✓ Video source loaded: {args.video}")
        else:
            print(f"Video file not found: {args.video}, falling back to camera.")

    if not is_demo and cap is None:
        print(f"Attempting to open camera index {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened() and args.camera != 1:
            # Try index 1 as fallback
            print("Camera 0 failed, trying camera 1...")
            cap = cv2.VideoCapture(1)

        if not cap.isOpened():
            print("\n[NOTE] No physical webcam detected or camera stream could not be opened.")
            print("Switching automatically to interactive DEMO SIMULATION MODE.")
            print("The system will demonstrate tracking, gesture recognition, and air drawing.\n")
            is_demo = True
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            print("✓ Webcam connected and streaming.")

    # State variables
    mode = 1  # 1: HUD, 2: Air Canvas, 3: Controller
    show_hud = True
    mirror_frame = True
    frame_counter = 0
    fps_start_time = time.time()
    fps = 30.0

    window_name = "AI Hand Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    print("\n-------------------------------------------------------")
    print("Application Controls:")
    print("  '1' : Mode 1 - Detailed Hand & Gesture HUD")
    print("  '2' : Mode 2 - Virtual Air Canvas Drawing")
    print("  '3' : Mode 3 - Gesture Action Controller")
    print("  'h' : Toggle Landmark Skeleton & Bounding Boxes")
    print("  'c' : Clear Air Canvas")
    print("  's' : Save Snapshot to disk")
    print("  'm' : Toggle Camera Mirroring")
    print("  'q' or ESC : Exit")
    print("-------------------------------------------------------\n")

    try:
        while True:
            frame_counter += 1

            # 1. Capture or Generate Frame
            if is_demo:
                frame = create_demo_frame(frame_counter, width=args.width, height=args.height)
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Loop video or notify
                    if args.video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    if not ret or frame is None:
                        print("End of stream or camera disconnected.")
                        break

            if mirror_frame and not is_demo:
                frame = cv2.flip(frame, 1)

            # Resize if needed to ensure consistency
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            # 2. Hand Detection & Landmark Analysis
            hands = detector.detect(frame)

            # Determine dominant gesture and update filter
            primary_gesture = hands[0]["gesture"] if hands else "None"
            stable_gesture = gesture_filter.update(primary_gesture)

            # 3. Mode Processing
            if mode == 1:
                # Mode 1: HUD & Diagnostics
                if show_hud:
                    frame = detector.draw_landmarks(frame, hands, draw_hud=True)

            elif mode == 2:
                # Mode 2: Air Canvas
                action = canvas.update(hands)
                frame = canvas.draw_ui(frame)
                if show_hud:
                    # Draw minimal fingertips
                    frame = detector.draw_landmarks(frame, hands, draw_hud=False)

            elif mode == 3:
                # Mode 3: Gesture Controller
                action_controller.update(hands, stable_gesture)
                if show_hud:
                    frame = detector.draw_landmarks(frame, hands, draw_hud=True)
                frame = action_controller.draw_ui(frame)

            # 4. FPS Calculation
            if frame_counter % 10 == 0:
                elapsed = time.time() - fps_start_time
                if elapsed > 0:
                    fps = 10.0 / elapsed
                fps_start_time = time.time()

            # 5. Header Overlay
            render_header(frame, mode, fps, len(hands), mirror_frame)

            # 6. Display Frame
            cv2.imshow(window_name, frame)

            # 7. Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # 'q' or ESC
                break
            elif key == ord('1'):
                mode = 1
                print("Switched to Mode 1: Recognition HUD")
            elif key == ord('2'):
                mode = 2
                print("Switched to Mode 2: Virtual Air Canvas")
            elif key == ord('3'):
                mode = 3
                print("Switched to Mode 3: Gesture Controller")
            elif key == ord('h'):
                show_hud = not show_hud
                print(f"HUD Overlay: {'Enabled' if show_hud else 'Disabled'}")
            elif key == ord('c'):
                canvas.clear()
                print("Canvas cleared.")
            elif key == ord('m'):
                mirror_frame = not mirror_frame
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.png"
                cv2.imwrite(filename, frame)
                print(f"✓ Snapshot saved: {filename}")

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("System shutdown cleanly.")


if __name__ == "__main__":
    main()

