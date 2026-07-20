"""
detection.py – YOLOv8 Animal Detection Wrapper
Smart Dairy Livestock Monitoring System

Uses YOLOv8 nano model to detect animals (cattle, sheep, horse, dog, cow, etc.)
Filters COCO classes relevant to livestock.
"""

import cv2
import numpy as np

# Livestock-relevant COCO class IDs
# 16=bird, 17=cat, 18=dog, 19=horse, 20=sheep, 21=cow, 22=elephant
LIVESTOCK_CLASSES = {17: 'Cat', 18: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cow', 22: 'Elephant'}
# For demo without actual livestock, also include person(0) so webcam always shows detections
DEMO_CLASSES = {0: 'Animal', 14: 'Bird', 15: 'Cat', 16: 'Dog', 19: 'Horse', 20: 'Sheep', 21: 'Cattle'}

MIN_CONFIDENCE = 0.35  # minimum detection confidence threshold


class Detector:
    """
    YOLOv8-based object detector for livestock surveillance.
    Falls back to demo mode (detects any object) if no livestock are found.
    """

    def __init__(self, model_size: str = 'yolov8n.pt', demo_mode: bool = True):
        """
        Args:
            model_size: YOLOv8 model variant ('yolov8n.pt' is fastest)
            demo_mode : If True, detect all objects (useful for webcam demo without cattle)
        """
        self.demo_mode = demo_mode
        self.model = None
        self._load_model(model_size)

    def _load_model(self, model_size: str):
        """Load YOLOv8 model. Downloads automatically on first run (~6 MB)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_size)
            print(f"[Detector] YOLOv8 model loaded: {model_size}")
        except ImportError:
            print("[Detector] WARNING: ultralytics not installed. Using mock detector.")
            self.model = None
        except Exception as e:
            print(f"[Detector] ERROR loading model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Returns:
            List of detection dicts:
            {
                'bbox': [x1, y1, x2, y2],   # pixel coords
                'confidence': float,          # 0–1
                'class_id': int,
                'label': str                  # human-readable class name
            }
        """
        if self.model is None:
            return self._mock_detect(frame)

        try:
            results = self.model(frame, verbose=False, conf=MIN_CONFIDENCE)[0]
        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return []

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # In demo mode accept everything; otherwise filter livestock only
            if self.demo_mode:
                label = results.names.get(cls_id, f'Class{cls_id}')
            else:
                if cls_id not in LIVESTOCK_CLASSES:
                    continue
                label = LIVESTOCK_CLASSES[cls_id]

            detections.append({
                'bbox':       [x1, y1, x2, y2],
                'confidence': round(conf, 3),
                'class_id':   cls_id,
                'label':      label
            })

        return detections

    # ── Mock detector for environments without ultralytics ───────────────────
    def _mock_detect(self, frame: np.ndarray) -> list[dict]:
        """Returns a simulated detection so the UI stays functional."""
        h, w = frame.shape[:2]
        return [{
            'bbox':       [w//4, h//4, 3*w//4, 3*h//4],
            'confidence': 0.82,
            'class_id':   21,
            'label':      'Cattle (Demo)'
        }]


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Draw bounding boxes and labels on the frame.
    Returns annotated copy of frame.
    """
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        conf  = det['confidence']

        # Colour by label
        colour = (0, 220, 80)   # green default
        if 'Cattle' in label or 'Cow' in label:
            colour = (255, 165, 0)    # orange
        elif 'Demo' in label:
            colour = (100, 200, 255)  # light-blue

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        text = f"{label}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out
