"""
Virtual Air Canvas Module.
Allows drawing in the air using hand gestures, tracking the index fingertip.
Supports multiple vibrant colors, eraser, brush sizes, and stroke smoothing.
"""

import time
import cv2
import numpy as np


class AirCanvas:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Palette colors (BGR)
        self.colors = [
            {"name": "Cyan", "bgr": (255, 255, 0)},
            {"name": "Green", "bgr": (0, 255, 100)},
            {"name": "Magenta", "bgr": (255, 0, 255)},
            {"name": "Yellow", "bgr": (0, 255, 255)},
            {"name": "Orange", "bgr": (0, 140, 255)},
            {"name": "White", "bgr": (255, 255, 255)},
            {"name": "Eraser", "bgr": (0, 0, 0)},
        ]
        self.current_color_idx = 0
        self.brush_size = 6
        self.eraser_size = 35

        self.prev_point = None
        self.header_height = 80
        self.buttons = []
        self._init_toolbar()

        # Selection hover timer
        self.hover_btn_idx = None
        self.hover_start_time = 0
        self.hover_duration = 0.4  # seconds to confirm button click by hovering

    def _init_toolbar(self):
        """Creates clickable/hoverable toolbar buttons across the top."""
        total_buttons = len(self.colors) + 1  # colors + clear button
        btn_w = min(110, (self.width - 20) // total_buttons)
        btn_h = 50
        y1 = 15
        y2 = y1 + btn_h

        self.buttons = []
        for idx, col in enumerate(self.colors):
            x1 = 20 + idx * (btn_w + 10)
            x2 = x1 + btn_w
            self.buttons.append({
                "type": "color" if col["name"] != "Eraser" else "eraser",
                "idx": idx,
                "label": col["name"],
                "bgr": col["bgr"],
                "rect": (x1, y1, x2, y2)
            })

        # Clear button at the end
        clear_x1 = 20 + len(self.colors) * (btn_w + 10)
        clear_x2 = clear_x1 + btn_w
        self.buttons.append({
            "type": "clear",
            "idx": -1,
            "label": "CLEAR",
            "bgr": (40, 40, 200),
            "rect": (clear_x1, y1, clear_x2, y2)
        })

    def update(self, hands_info):
        """
        Processes detected hands, updates drawing, and checks toolbar interactions.
        Returns the action performed (if any).
        """
        action = None
        if not hands_info:
            self.prev_point = None
            self.hover_btn_idx = None
            return action

        # Use primary hand (first detected)
        hand = hands_info[0]
        lm = hand["landmarks"]
        index_tip = (lm[8][0], lm[8][1])
        fingers = hand["fingers_up"]
        pinch = hand["pinch"]
        gesture = hand["gesture"]

        ix, iy = index_tip

        # 1. Check if hovering/clicking in the Top Toolbar area
        if iy < self.header_height:
            self.prev_point = None  # Don't draw while inside toolbar
            for btn in self.buttons:
                x1, y1, x2, y2 = btn["rect"]
                if x1 <= ix <= x2 and y1 <= iy <= y2:
                    # Hovering over button
                    if self.hover_btn_idx == btn["idx"]:
                        if time.time() - self.hover_start_time >= self.hover_duration or pinch["is_pinch"]:
                            action = self._trigger_button(btn)
                            self.hover_btn_idx = None
                    else:
                        self.hover_btn_idx = btn["idx"]
                        self.hover_start_time = time.time()
                    return action
            self.hover_btn_idx = None
            return action

        self.hover_btn_idx = None

        # 2. Gesture Actions in Canvas Area
        # If open palm (all 5 fingers up), don't draw (free movement)
        if gesture == "Open Palm" or sum(fingers) >= 4:
            self.prev_point = None
            return action

        # If 2 fingers up (Index + Middle), Selection / Hover Mode (cursor moves, no drawing)
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            self.prev_point = None
            return action

        # Drawing Mode: Only Index finger up OR active Pinch
        is_drawing = (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]) or pinch["is_pinch"]

        if is_drawing:
            draw_pt = pinch["pinch_point"] if pinch["is_pinch"] else index_tip

            if self.prev_point is None:
                self.prev_point = draw_pt

            is_eraser = (self.colors[self.current_color_idx]["name"] == "Eraser")
            color = (0, 0, 0) if is_eraser else self.colors[self.current_color_idx]["bgr"]
            thickness = self.eraser_size if is_eraser else self.brush_size

            # Draw smooth line between consecutive points
            cv2.line(self.canvas, self.prev_point, draw_pt, color, thickness, cv2.LINE_AA)
            cv2.circle(self.canvas, draw_pt, thickness // 2, color, -1, cv2.LINE_AA)
            self.prev_point = draw_pt
        else:
            self.prev_point = None

        return action

    def _trigger_button(self, btn):
        if btn["type"] == "clear":
            self.clear()
            return "Cleared Canvas"
        else:
            self.current_color_idx = btn["idx"]
            return f"Selected {btn['label']}"

    def clear(self):
        self.canvas[:] = 0

    def draw_ui(self, frame):
        """
        Blends drawing canvas onto video frame and renders top toolbar.
        """
        # Ensure dimensions match
        fh, fw = frame.shape[:2]
        if self.canvas.shape[:2] != (fh, fw):
            self.canvas = cv2.resize(self.canvas, (fw, fh))
            self.width, self.height = fw, fh
            self._init_toolbar()

        # 1. Blend canvas into frame
        # Grayscale mask of canvas
        canvas_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Black-out area of drawing in original frame, then add canvas strokes
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        blended = cv2.add(frame_bg, canvas_fg)

        # 2. Render Toolbar Background
        header_overlay = blended.copy()
        cv2.rectangle(header_overlay, (0, 0), (fw, self.header_height), (15, 15, 15), -1)
        cv2.addWeighted(header_overlay, 0.75, blended, 0.25, 0, blended)

        # 3. Render Buttons
        for btn in self.buttons:
            x1, y1, x2, y2 = btn["rect"]
            is_active = (btn["idx"] == self.current_color_idx)
            is_hovered = (btn["idx"] == self.hover_btn_idx)

            # Button background
            btn_bg = btn["bgr"] if btn["label"] != "Eraser" else (60, 60, 60)
            if btn["type"] == "clear":
                btn_bg = (30, 30, 180)

            border_color = (0, 255, 255) if (is_active or is_hovered) else (100, 100, 100)
            thickness = 3 if is_active else 1

            cv2.rectangle(blended, (x1, y1), (x2, y2), btn_bg, -1)
            cv2.rectangle(blended, (x1, y1), (x2, y2), border_color, thickness)

            # Text
            text_color = (0, 0, 0) if btn["label"] in ("Yellow", "White", "Cyan", "Green") else (255, 255, 255)
            font_scale = 0.45
            (tw, th), _ = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = x1 + (x2 - x1 - tw) // 2
            ty = y1 + (y2 - y1 + th) // 2
            cv2.putText(blended, btn["label"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

            # Hover progress bar indicator
            if is_hovered and self.hover_start_time > 0:
                elapsed = time.time() - self.hover_start_time
                pct = min(1.0, elapsed / self.hover_duration)
                bar_w = int((x2 - x1) * pct)
                cv2.rectangle(blended, (x1, y2 - 4), (x1 + bar_w, y2), (0, 255, 0), -1)

        return blended

