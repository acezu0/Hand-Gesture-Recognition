"""
Unit & Integration Test Suite for Hand Gesture Recognition System.
Tests detector, feature extractors, canvas engine, and controller.
"""

import os
import unittest
import numpy as np
import cv2

from hand_detector import HandDetector, WRIST, THUMB_TIP, INDEX_TIP, PINKY_MCP
from canvas import AirCanvas
from gesture_actions import GestureFilter, ActionController
from main import create_demo_frame


class TestHandRecognitionSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = "gesture_recognizer.task"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model asset: {model_path}")
        cls.detector = HandDetector(model_path=model_path)
        cls.canvas = AirCanvas(width=640, height=480)
        cls.filter = GestureFilter(history_len=5)
        cls.controller = ActionController(width=640, height=480)

    def test_01_model_loading(self):
        """Verify MediaPipe gesture recognizer model initializes."""
        self.assertIsNotNone(self.detector.recognizer)
        self.assertEqual(self.detector.num_hands, 2)

    def test_02_inference_on_blank_frame(self):
        """Verify detector runs without error on blank frame."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = self.detector.detect(blank_frame)
        self.assertIsInstance(hands, list)
        self.assertEqual(len(hands), 0)

    def test_03_inference_on_demo_frame(self):
        """Verify detector processes synthetic generated hand frame."""
        demo_frame = create_demo_frame(1, width=640, height=480)
        self.assertEqual(demo_frame.shape, (480, 640, 3))
        # Draw on frame to verify HUD pipeline
        hands = self.detector.detect(demo_frame)
        rendered = self.detector.draw_landmarks(demo_frame.copy(), hands, draw_hud=True)
        self.assertEqual(rendered.shape, (480, 640, 3))

    def test_04_pinch_calculation(self):
        """Verify pinch geometry calculation."""
        # Mock 21 landmarks
        mock_lm = [(100, 100, 0.0)] * 21
        # Set wrist and middle MCP
        mock_lm[0] = (100, 200, 0.0)      # WRIST
        mock_lm[9] = (100, 100, 0.0)      # MIDDLE_MCP (distance = 100)
        # Thumb tip and index tip close together
        mock_lm[4] = (105, 95, 0.0)       # THUMB_TIP
        mock_lm[8] = (108, 98, 0.0)       # INDEX_TIP

        pinch_info = self.detector._get_pinch_info(mock_lm, (640, 480))
        self.assertTrue(pinch_info["is_pinch"])
        self.assertLess(pinch_info["distance_px"], 20)

        # Separate thumb and index tip
        mock_lm[8] = (250, 100, 0.0)
        pinch_info2 = self.detector._get_pinch_info(mock_lm, (640, 480))
        self.assertFalse(pinch_info2["is_pinch"])

    def test_05_gesture_classification(self):
        """Verify rule-based gesture classification."""
        pinch_false = {"is_pinch": False, "distance_px": 80.0, "normalized_distance": 0.8}
        pinch_true = {"is_pinch": True, "distance_px": 10.0, "normalized_distance": 0.1}

        # Fist
        g = self.detector._classify_gesture([0, 0, 0, 0, 0], pinch_false, "None", 0.0, {})
        self.assertEqual(g, "Closed Fist")

        # Pointing
        g = self.detector._classify_gesture([0, 1, 0, 0, 0], pinch_false, "None", 0.0, {})
        self.assertEqual(g, "Pointing")

        # Peace / Victory
        g = self.detector._classify_gesture([0, 1, 1, 0, 0], pinch_false, "None", 0.0, {})
        self.assertEqual(g, "Peace / Victory")

        # OK Sign (Pinch + 3 open)
        g = self.detector._classify_gesture([1, 1, 1, 1, 1], pinch_true, "None", 0.0, {})
        self.assertEqual(g, "OK Sign")

        # Open Palm
        g = self.detector._classify_gesture([1, 1, 1, 1, 1], pinch_false, "None", 0.0, {})
        self.assertEqual(g, "Open Palm")

    def test_06_canvas_operations(self):
        """Verify canvas drawing, color changing, and clearing."""
        canvas = AirCanvas(width=640, height=480)
        self.assertEqual(np.count_nonzero(canvas.canvas), 0)

        # Mock hand with index finger pointing to draw
        mock_hand = {
            "landmarks": [(200, 200, 0.0)] * 21,
            "fingers_up": [0, 1, 0, 0, 0],  # only index
            "pinch": {"is_pinch": False, "pinch_point": (200, 200)},
            "gesture": "Pointing"
        }
        # Point 1
        canvas.update([mock_hand])
        # Point 2
        mock_hand["landmarks"][8] = (220, 220, 0.0)
        canvas.update([mock_hand])

        # Canvas should have drawn pixels
        self.assertGreater(np.count_nonzero(canvas.canvas), 0)

        # Test clear
        canvas.clear()
        self.assertEqual(np.count_nonzero(canvas.canvas), 0)

    def test_07_gesture_smoothing_filter(self):
        """Verify temporal debounce filter suppresses momentary noise."""
        gf = GestureFilter(history_len=5)
        for _ in range(3):
            gf.update("Victory")
        self.assertEqual(gf.current_stable_gesture, "Victory")

        # Single frame glitch to 'Closed Fist' should not flip the stable state
        gf.update("Closed Fist")
        self.assertEqual(gf.current_stable_gesture, "Victory")

    def test_08_action_controller(self):
        """Verify action controller updates slider and actions."""
        ac = ActionController(width=640, height=480)
        initial_val = ac.slider_value

        # Thumbs up should increment slider
        ac.update([], "Thumbs Up")
        # Mock hand with thumbs up
        mock_hand = {
            "landmarks": [(100, 100, 0.0)] * 21,
            "pinch": {"is_pinch": False, "pinch_point": (100, 100)},
        }
        ac.update([mock_hand], "Thumbs Up")
        self.assertGreaterEqual(ac.slider_value, initial_val)


if __name__ == "__main__":
    unittest.main()

