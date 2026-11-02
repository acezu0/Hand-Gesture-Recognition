"""
Model Downloader Utility.
Checks if 'gesture_recognizer.task' exists; if not, downloads it from Google's official MediaPipe repository.
"""

import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
MODEL_NAME = "gesture_recognizer.task"

def ensure_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, MODEL_NAME)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000:
        print(f"✓ Model '{MODEL_NAME}' is present ({os.path.getsize(model_path):,} bytes).")
        return model_path

    print(f"Downloading MediaPipe Gesture Recognizer model from:\n{MODEL_URL}...")
    def report_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            print(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB)", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=report_hook)
    print("\n✓ Download completed successfully.")
    return model_path

if __name__ == "__main__":
    ensure_model()

