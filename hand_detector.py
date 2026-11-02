"""
Hand Detector and Gesture Recognition Module.
Uses MediaPipe Tasks Vision GestureRecognizer and HandLandmarker models
with custom geometric feature extraction (finger states, pinch distance, custom gestures).
"""

import os
import tempfile
# Ensure Matplotlib cache uses writable temp directory on Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_mp"))

import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 21 Hand Landmark indices reference
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# Landmark connections for skeleton rendering
HAND_CONNECTIONS = [
    # Thumb
    (WRIST, THUMB_CMC), (THUMB_CMC, THUMB_MCP), (THUMB_MCP, THUMB_IP), (THUMB_IP, THUMB_TIP),
    # Index
    (WRIST, INDEX_MCP), (INDEX_MCP, INDEX_PIP), (INDEX_PIP, INDEX_DIP), (INDEX_DIP, INDEX_TIP),
    # Middle
    (INDEX_MCP, MIDDLE_MCP), (MIDDLE_MCP, MIDDLE_PIP), (MIDDLE_PIP, MIDDLE_DIP), (MIDDLE_DIP, MIDDLE_TIP),
    # Ring
    (MIDDLE_MCP, RING_MCP), (RING_MCP, RING_PIP), (RING_PIP, RING_DIP), (RING_DIP, RING_TIP),
    # Pinky
    (RING_MCP, PINKY_MCP), (PINKY_MCP, PINKY_PIP), (PINKY_PIP, PINKY_DIP), (PINKY_DIP, PINKY_TIP),
    # Palm base
    (WRIST, PINKY_MCP)
]


class HandDetector:
    def __init__(self, model_path="gesture_recognizer.task", num_hands=2, min_confidence=0.5):
        """
        Initializes the MediaPipe Gesture Recognizer.
        """
        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            full_model_path = os.path.join(base_dir, model_path)
            if os.path.exists(full_model_path):
                model_path = full_model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. Please ensure 'gesture_recognizer.task' exists."
            )

        self.model_path = model_path
        self.num_hands = num_hands
        self.min_confidence = min_confidence

        # Configure MediaPipe GestureRecognizer
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def detect(self, bgr_frame):
        """
        Processes a BGR OpenCV frame and returns a list of detected hand dictionaries.
        """
        h, w, _ = bgr_frame.shape
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.recognizer.recognize(mp_image)
        hands = []

        if not result.hand_landmarks:
            return hands

        for i, landmarks_list in enumerate(result.hand_landmarks):
            # 1. Pixel and Normalized Landmarks
            lm_px = []
            lm_norm = []
            xs, ys = [], []

            for lm in landmarks_list:
                px_x = int(lm.x * w)
                px_y = int(lm.y * h)
                lm_px.append((px_x, px_y, lm.z))
                lm_norm.append((lm.x, lm.y, lm.z))
                xs.append(px_x)
                ys.append(px_y)

            # 2. Bounding Box with padding
            pad = 20
            xmin = max(0, min(xs) - pad)
            ymin = max(0, min(ys) - pad)
            xmax = min(w, max(xs) + pad)
            ymax = min(h, max(ys) + pad)
            bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

            # 3. Handedness
            handedness = "Hand"
            hand_score = 0.0
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                top_hand = result.handedness[i][0]
                handedness = top_hand.category_name
                hand_score = top_hand.score

            # 4. Built-in MediaPipe Gesture
            gesture_name = "None"
            gesture_score = 0.0
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                top_gesture = result.gestures[i][0]
                gesture_name = top_gesture.category_name
                gesture_score = top_gesture.score

            # 5. Geometric Finger Analysis
            fingers_up = self._get_fingers_up(lm_px, handedness)
            finger_count = sum(fingers_up)

            # 6. Pinch Analysis (Thumb Tip 4 to Index Tip 8)
            pinch_info = self._get_pinch_info(lm_px, (w, h))

            # 7. Enhanced Custom Gesture Fusion
            custom_gesture = self._classify_gesture(
                fingers_up, pinch_info, gesture_name, gesture_score, lm_px
            )

            hand_dict = {
                "id": i,
                "handedness": handedness,
                "hand_score": hand_score,
                "landmarks": lm_px,
                "normalized_landmarks": lm_norm,
                "bbox": bbox,
                "model_gesture": gesture_name,
                "model_score": gesture_score,
                "fingers_up": fingers_up,  # [thumb, index, middle, ring, pinky]
                "finger_count": finger_count,
                "pinch": pinch_info,
                "gesture": custom_gesture,
            }
            hands.append(hand_dict)

        return hands

    def _get_fingers_up(self, lm, handedness):
        """
        Determines which fingers are open/extended [Thumb, Index, Middle, Ring, Pinky].
        Uses distance-based and joint-angle checks robust to hand rotation.
        """
        fingers = [0, 0, 0, 0, 0]

        # Wrist and MCP positions
        wrist = np.array([lm[WRIST][0], lm[WRIST][1]])
        pinky_mcp = np.array([lm[PINKY_MCP][0], lm[PINKY_MCP][1]])
        thumb_mcp = np.array([lm[THUMB_MCP][0], lm[THUMB_MCP][1]])
        thumb_tip = np.array([lm[THUMB_TIP][0], lm[THUMB_TIP][1]])
        thumb_ip = np.array([lm[THUMB_IP][0], lm[THUMB_IP][1]])

        # 1. Thumb: Check distance from pinky MCP and projection
        d_tip_pinky = np.linalg.norm(thumb_tip - pinky_mcp)
        d_ip_pinky = np.linalg.norm(thumb_ip - pinky_mcp)
        d_mcp_pinky = np.linalg.norm(thumb_mcp - pinky_mcp)

        if d_tip_pinky > d_ip_pinky and d_tip_pinky > d_mcp_pinky * 1.15:
            fingers[0] = 1

        # 2. Fingers: Index (8), Middle (12), Ring (16), Pinky (20)
        finger_indices = [
            (INDEX_TIP, INDEX_PIP, INDEX_MCP),
            (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            (RING_TIP, RING_PIP, RING_MCP),
            (PINKY_TIP, PINKY_PIP, PINKY_MCP),
        ]

        for idx, (tip, pip, mcp) in enumerate(finger_indices, start=1):
            p_tip = np.array([lm[tip][0], lm[tip][1]])
            p_pip = np.array([lm[pip][0], lm[pip][1]])
            p_mcp = np.array([lm[mcp][0], lm[mcp][1]])

            # Distance from wrist: extended finger has tip farther than PIP
            d_tip_wrist = np.linalg.norm(p_tip - wrist)
            d_pip_wrist = np.linalg.norm(p_pip - wrist)
            d_mcp_wrist = np.linalg.norm(p_mcp - wrist)

            # Also check vertical projection if hand is generally upright
            if d_tip_wrist > d_pip_wrist * 1.1 and d_pip_wrist > d_mcp_wrist * 0.9:
                fingers[idx] = 1

        return fingers

    def _get_pinch_info(self, lm, frame_size):
        """
        Calculates pinch state and distance between thumb tip and index tip.
        """
        w, h = frame_size
        thumb_tip = np.array([lm[THUMB_TIP][0], lm[THUMB_TIP][1]], dtype=float)
        index_tip = np.array([lm[INDEX_TIP][0], lm[INDEX_TIP][1]], dtype=float)
        wrist = np.array([lm[WRIST][0], lm[WRIST][1]], dtype=float)
        middle_mcp = np.array([lm[MIDDLE_MCP][0], lm[MIDDLE_MCP][1]], dtype=float)

        # Scale distance relative to palm size for resolution-independence
        palm_size = max(np.linalg.norm(middle_mcp - wrist), 30.0)
        dist_px = float(np.linalg.norm(thumb_tip - index_tip))
        normalized_dist = dist_px / palm_size

        mid_point = (int((thumb_tip[0] + index_tip[0]) / 2), int((thumb_tip[1] + index_tip[1]) / 2))
        is_pinch = normalized_dist < 0.28  # Close threshold

        return {
            "is_pinch": is_pinch,
            "distance_px": round(dist_px, 1),
            "normalized_distance": round(normalized_dist, 3),
            "pinch_point": mid_point,
            "thumb_point": (int(thumb_tip[0]), int(thumb_tip[1])),
            "index_point": (int(index_tip[0]), int(index_tip[1]))
        }

    def _classify_gesture(self, fingers, pinch, model_gesture, model_score, lm):
        """
        Combines deep learning model predictions with geometric rules for high accuracy.
        """
        thumb, index, middle, ring, pinky = fingers
        finger_count = sum(fingers)

        # 1. High-confidence model gestures take precedence if consistent
        if model_score > 0.75 and model_gesture not in ("None", "Unrecognized"):
            if model_gesture == "Victory" and index and middle and not ring and not pinky:
                return "Peace / Victory"
            if model_gesture == "Thumb_Up" and thumb and not index and not middle and not ring and not pinky:
                return "Thumbs Up"
            if model_gesture == "Thumb_Down" and thumb and not index and not middle and not ring and not pinky:
                return "Thumbs Down"
            if model_gesture == "Closed_Fist" and finger_count == 0:
                return "Closed Fist"
            if model_gesture == "Open_Palm" and finger_count >= 4:
                return "Open Palm"
            if model_gesture == "Pointing_Up" and index and not middle and not ring and not pinky:
                return "Pointing"
            if model_gesture == "ILoveYou":
                return "I Love You"

        # 2. Geometric custom gesture rules
        if pinch["is_pinch"]:
            if middle and ring and pinky:
                return "OK Sign"
            if not middle and not ring and not pinky:
                return "Pinch Click"
            return "Pinch"

        if finger_count == 0:
            return "Closed Fist"

        if fingers == [0, 1, 0, 0, 0]:
            return "Pointing"

        if fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            return "Peace / Victory"

        if fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
            return "Rock / Horns"

        if fingers == [1, 0, 0, 0, 1]:
            return "Call Me"

        if fingers == [1, 1, 0, 0, 0]:
            return "Gun Gesture"

        if fingers == [1, 0, 0, 0, 0]:
            # Check orientation of thumb (up vs down)
            thumb_tip_y = lm[THUMB_TIP][1]
            thumb_mcp_y = lm[THUMB_MCP][1]
            if thumb_tip_y < thumb_mcp_y:
                return "Thumbs Up"
            else:
                return "Thumbs Down"

        if fingers == [0, 1, 1, 1, 0]:
            return "Three Fingers"

        if fingers == [0, 1, 1, 1, 1]:
            return "Four Fingers"

        if finger_count == 5:
            return "Open Palm"

        # Fallback to model gesture if available
        if model_gesture not in ("None", "Unrecognized") and model_score > 0.5:
            return model_gesture.replace("_", " ")

        return f"{finger_count} Finger{'s' if finger_count != 1 else ''}"

    def draw_landmarks(self, frame, hands_info, draw_hud=True):
        """
        Draws a modern, high-tech HUD overlay on the frame with skeleton bones,
        glowing fingertip nodes, dynamic pinch lines, and info tags.
        """
        overlay = frame.copy()

        # Modern color palette
        COLOR_BONE = (245, 180, 0)       # Cyan / Electric Blue (BGR)
        COLOR_JOINT = (255, 255, 255)    # White
        COLOR_FINGERTIP = (0, 240, 255)  # Bright Yellow-Amber
        COLOR_PINCH = (0, 69, 255)       # Orange-Red for active pinch
        COLOR_BOX = (50, 205, 50)        # Lime Green

        for hand in hands_info:
            lm = hand["landmarks"]
            bx, by, bw, bh = hand["bbox"]

            # 1. Draw Skeleton Bones
            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = (lm[start_idx][0], lm[start_idx][1])
                p2 = (lm[end_idx][0], lm[end_idx][1])
                cv2.line(overlay, p1, p2, COLOR_BONE, 2, cv2.LINE_AA)

            # 2. Draw Landmark Joints
            for idx, pt in enumerate(lm):
                p = (pt[0], pt[1])
                if idx in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
                    # Fingertip glow
                    cv2.circle(overlay, p, 7, COLOR_FINGERTIP, -1, cv2.LINE_AA)
                    cv2.circle(overlay, p, 9, (0, 140, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(overlay, p, 4, COLOR_JOINT, -1, cv2.LINE_AA)
                    cv2.circle(overlay, p, 5, COLOR_BONE, 1, cv2.LINE_AA)

            # 3. Dynamic Pinch Connection Line
            pinch = hand["pinch"]
            if pinch["distance_px"] < 120:
                p_thumb = pinch["thumb_point"]
                p_index = pinch["index_point"]
                line_color = COLOR_PINCH if pinch["is_pinch"] else (180, 180, 180)
                line_thickness = 3 if pinch["is_pinch"] else 1
                cv2.line(overlay, p_thumb, p_index, line_color, line_thickness, cv2.LINE_AA)
                mid_pt = pinch["pinch_point"]
                cv2.circle(overlay, mid_pt, 5, line_color, -1, cv2.LINE_AA)

            if draw_hud:
                # 4. Stylized Corner Bounding Box
                self._draw_corner_box(overlay, bx, by, bw, bh, COLOR_BOX)

                # 5. Hand Info Tag Banner
                handedness = hand["handedness"]
                score_pct = int(hand["hand_score"] * 100)
                gesture = hand["gesture"]
                f_count = hand["finger_count"]

                label_top = f"{handedness} Hand ({score_pct}%)"
                label_bot = f"{gesture} | {f_count} Up"

                tag_w = max(len(label_top), len(label_bot)) * 9 + 20
                tag_h = 44
                tag_x = bx
                tag_y = max(10, by - tag_h - 6)

                # Semi-transparent dark background badge
                cv2.rectangle(
                    overlay,
                    (tag_x, tag_y),
                    (tag_x + tag_w, tag_y + tag_h),
                    (20, 20, 20),
                    -1
                )
                cv2.rectangle(
                    overlay,
                    (tag_x, tag_y),
                    (tag_x + tag_w, tag_y + tag_h),
                    COLOR_BOX,
                    1
                )
                cv2.putText(
                    overlay,
                    label_top,
                    (tag_x + 8, tag_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                    cv2.LINE_AA
                )
                cv2.putText(
                    overlay,
                    label_bot,
                    (tag_x + 8, tag_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Blend overlay with original frame for clean anti-aliased look
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        return frame

    def _draw_corner_box(self, img, x, y, w, h, color, corner_len=18, thickness=2):
        """Draws aesthetic corner brackets around a bounding box."""
        # Top-Left
        cv2.line(img, (x, y), (x + corner_len, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + corner_len), color, thickness, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x + w, y), (x + w - corner_len, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y), (x + w, y + corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x, y + h), (x + corner_len, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y + h), (x, y + h - corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x + w, y + h), (x + w - corner_len, y + h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x + w, y + h), (x + w, y + h - corner_len), color, thickness, cv2.LINE_AA)
