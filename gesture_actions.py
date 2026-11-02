"""
Gesture Action Controller & Temporal Filter Module.
Provides temporal smoothing to eliminate gesture jitter and implements
an interactive controller for Mode 3 (virtual slider, media control triggers).
"""

import time
from collections import deque
import cv2
import numpy as np


class GestureFilter:
    """Temporal filter to smooth out single-frame misclassifications."""
    def __init__(self, history_len=5):
        self.history = deque(maxlen=history_len)
        self.current_stable_gesture = "None"

    def update(self, detected_gesture):
        self.history.append(detected_gesture)
        # Find most frequent gesture in history
        counts = {}
        for g in self.history:
            counts[g] = counts.get(g, 0) + 1
        most_common = max(counts, key=counts.get)
        # Require majority
        if counts[most_common] >= len(self.history) // 2 + 1:
            self.current_stable_gesture = most_common
        return self.current_stable_gesture


class ActionController:
    """
    Mode 3: Interactive gesture-based control panel.
    Features:
    - Virtual Pinch Slider (adjusts value 0-100% by sliding pinched hand up/down or varying distance)
    - Virtual buttons triggered by steady gestures (Play/Pause, Next/Prev, Mute)
    - Action log showing recent events
    """
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.slider_value = 50.0  # 0 to 100
        self.is_playing = False
        self.is_muted = False
        self.last_action_time = 0
        self.action_cooldown = 1.0  # seconds between triggering discrete actions
        self.status_message = "Ready - Show gestures to control"
        self.status_time = time.time()
        self.action_log = []

    def update(self, hands_info, stable_gesture):
        current_time = time.time()

        if not hands_info:
            return

        hand = hands_info[0]
        pinch = hand["pinch"]
        lm = hand["landmarks"]

        # 1. Virtual Slider Control via Pinch
        # When pinching, vertical position of hand controls slider value (higher up = higher value)
        if pinch["is_pinch"]:
            py = pinch["pinch_point"][1]
            # Map vertical position from range [120, height - 100] to [100, 0]
            val = 100.0 * (1.0 - (py - 120) / max(10, self.height - 220))
            self.slider_value = max(0.0, min(100.0, val))
            self.set_message(f"Slider Adjusted: {int(self.slider_value)}%")

        # 2. Discrete Actions with Cooldown
        if current_time - self.last_action_time >= self.action_cooldown:
            if stable_gesture == "Thumbs Up":
                self.slider_value = min(100.0, self.slider_value + 10)
                self.trigger_action(f"Volume Up -> {int(self.slider_value)}%")
            elif stable_gesture == "Thumbs Down":
                self.slider_value = max(0.0, self.slider_value - 10)
                self.trigger_action(f"Volume Down -> {int(self.slider_value)}%")
            elif stable_gesture in ("Peace / Victory", "Victory"):
                self.is_playing = not self.is_playing
                self.trigger_action("Play / Pause Toggle: " + ("PLAYING" if self.is_playing else "PAUSED"))
            elif stable_gesture == "Rock / Horns":
                self.is_muted = not self.is_muted
                self.trigger_action("Mute Toggle: " + ("MUTED" if self.is_muted else "UNMUTED"))
            elif stable_gesture == "OK Sign":
                self.trigger_action("Confirmed: OK Gesture Accepted")

    def trigger_action(self, msg):
        self.last_action_time = time.time()
        self.set_message(msg)
        timestamp = time.strftime("%H:%M:%S")
        self.action_log.append(f"[{timestamp}] {msg}")
        if len(self.action_log) > 6:
            self.action_log.pop(0)

    def set_message(self, msg):
        self.status_message = msg
        self.status_time = time.time()

    def draw_ui(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # 1. Right-side Control Panel
        panel_w = 340
        panel_x = w - panel_w - 20
        panel_y = 70
        panel_h = h - 90

        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (25, 25, 25), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 200, 255), 2)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Panel Header
        cv2.putText(frame, "GESTURE CONTROLLER", (panel_x + 20, panel_y + 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

        # 2. Interactive Vertical Slider
        slider_x = panel_x + 40
        slider_y = panel_y + 65
        slider_h = 160
        slider_w = 26

        # Slider background track
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (60, 60, 60), -1)
        # Slider fill
        fill_h = int(slider_h * (self.slider_value / 100.0))
        cv2.rectangle(
            frame,
            (slider_x, slider_y + slider_h - fill_h),
            (slider_x + slider_w, slider_y + slider_h),
            (0, 200, 0),
            -1
        )
        cv2.rectangle(frame, (slider_x, slider_y), (slider_x + slider_w, slider_y + slider_h), (200, 200, 200), 2)

        # Slider labels
        cv2.putText(frame, f"Value: {int(self.slider_value)}%", (slider_x + slider_w + 15, slider_y + slider_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "(Pinch & drag vertically)", (slider_x + slider_w + 15, slider_y + slider_h // 2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)

        # 3. Media Status Indicators
        state_y = slider_y + slider_h + 35
        play_color = (0, 255, 0) if self.is_playing else (100, 100, 100)
        mute_color = (0, 0, 255) if self.is_muted else (0, 255, 0)

        cv2.putText(frame, f"State: {'PLAYING' if self.is_playing else 'PAUSED'}", (panel_x + 20, state_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, play_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Audio: {'MUTED' if self.is_muted else 'ACTIVE'}", (panel_x + 180, state_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mute_color, 1, cv2.LINE_AA)

        # 4. Gesture Trigger Cheatsheet
        guide_y = state_y + 30
        cv2.line(frame, (panel_x + 15, guide_y), (panel_x + panel_w - 15, guide_y), (80, 80, 80), 1)
        cv2.putText(frame, "Gestures & Shortcuts:", (panel_x + 20, guide_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        shortcuts = [
            ("Pinch + Move", "Drag Slider"),
            ("Thumb Up/Down", "Vol +/- 10%"),
            ("Peace / Victory", "Play / Pause"),
            ("Rock / Horns", "Mute / Unmute"),
            ("OK Sign", "Confirm Action")
        ]
        for idx, (gest, desc) in enumerate(shortcuts):
            sy = guide_y + 40 + idx * 20
            cv2.putText(frame, f"{gest}:", (panel_x + 20, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (panel_x + 160, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 200), 1, cv2.LINE_AA)

        # 5. Live Action Event Log
        log_y = guide_y + 150
        cv2.line(frame, (panel_x + 15, log_y), (panel_x + panel_w - 15, log_y), (80, 80, 80), 1)
        cv2.putText(frame, "Event Log:", (panel_x + 20, log_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        for idx, log_entry in enumerate(self.action_log[-4:]):
            ly = log_y + 36 + idx * 18
            cv2.putText(frame, log_entry, (panel_x + 20, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # 6. Status Message Banner at Bottom Left
        if time.time() - self.status_time < 3.0:
            bw = 480
            cv2.rectangle(frame, (20, h - 50), (20 + bw, h - 15), (20, 20, 20), -1)
            cv2.rectangle(frame, (20, h - 50), (20 + bw, h - 15), (0, 255, 100), 1)
            cv2.putText(frame, f"> {self.status_message}", (30, h - 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1, cv2.LINE_AA)

        return frame

