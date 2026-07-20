"""
video_stream.py – Camera Capture, MJPEG Generator & Stats Provider
Smart Dairy Livestock Monitoring System

Manages the main processing pipeline:
  Camera → YOLOv8 Detection → DeepSORT Tracking →
  Behaviour Analysis → Anomaly Detection → Annotated MJPEG frame
"""

import cv2
import time
import threading
import numpy as np

from behaviour_tracking.detection      import Detector, draw_detections
from behaviour_tracking.tracker        import Tracker
from behaviour_tracking.behaviour_analyzer import BehaviourAnalyzer, BEHAVIOUR_COLOURS
from behaviour_tracking.anomaly_detector   import AnomalyDetector

BEHAVIOUR_COLOURS_BGR = BEHAVIOUR_COLOURS   # already BGR


class VideoStream:
    """
    Singleton-style pipeline controller.
    Call get_instance() to obtain the shared instance.
    """

    _instance = None
    _lock      = threading.Lock()

    def __init__(self, source=0, demo_mode=True):
        self.source      = source          # 0 = webcam, or path to video file
        self.demo_mode   = demo_mode
        self.running     = False
        self.cap         = None
        self._thread     = None

        # Pipeline components
        self.detector  = Detector(demo_mode=demo_mode)
        self.tracker   = Tracker()
        self.analyzer  = BehaviourAnalyzer()
        self.anomaly   = AnomalyDetector()

        # Shared state (thread-safe via lock)
        self._frame_lock  = threading.Lock()
        self._latest_jpeg = None           # bytes for MJPEG
        self._stats       = {              # latest snapshot for API
            'animal_count':  0,
            'behaviours':    {},
            'tracks':        [],
            'fps':           0.0,
            'alerts':        [],
            'alert_counts':  {'INFO': 0, 'WARNING': 0, 'CRITICAL': 0},
            'source_active': False,
        }
        self._stats_lock  = threading.Lock()

    # ── Singleton accessor ─────────────────────────────────────────────────
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self, source=None):
        if self.running:
            return
        if source is not None:
            self.source = source
        self.running = True
        self._thread = threading.Thread(target=self._pipeline_loop, daemon=True)
        self._thread.start()
        print(f"[VideoStream] Started. Source={self.source}")

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        print("[VideoStream] Stopped.")

    def switch_source(self, source):
        """Hot-switch camera/video source without restarting pipeline."""
        self.source = source
        if self.cap:
            self.cap.release()
            self.cap = None

    # ── Main pipeline loop (runs in background thread) ─────────────────────
    def _pipeline_loop(self):
        fps_counter  = 0
        fps_t        = time.time()
        current_fps  = 0.0

        while self.running:
            # Open / reopen capture
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
                if not self.cap.isOpened():
                    # No camera – generate placeholder frame
                    frame = self._placeholder_frame("No Camera Detected")
                    self._publish(frame, [], current_fps)
                    time.sleep(0.1)
                    continue
                else:
                    self.anomaly.add_system_alert('INFO', f'Camera source connected: {self.source}')

            ret, frame = self.cap.read()
            if not ret:
                # End of video file → loop back
                if isinstance(self.source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = self._placeholder_frame("Camera Feed Lost")
                self._publish(frame, [], current_fps)
                time.sleep(0.05)
                continue

            # ── Run detection ────────────────────────────────────────
            detections = self.detector.detect(frame)

            # ── Run tracking ─────────────────────────────────────────
            track_list = self.tracker.update(frame, detections)

            # ── Behaviour analysis ───────────────────────────────────
            self.analyzer.analyse(self.tracker.active_tracks)

            # ── Anomaly detection ────────────────────────────────────
            self.anomaly.update(self.tracker.active_tracks)

            # ── FPS ──────────────────────────────────────────────────
            fps_counter += 1
            if time.time() - fps_t >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_t       = time.time()

            # ── Annotate frame ───────────────────────────────────────
            annotated = self._annotate(frame, track_list, current_fps)

            self._publish(annotated, track_list, current_fps)

    # ── Frame annotation ───────────────────────────────────────────────────
    def _annotate(self, frame: np.ndarray, track_list: list, fps: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        for t in track_list:
            x1, y1, x2, y2 = t['bbox']
            beh    = t.get('behaviour', 'Detecting…')
            vel    = t.get('velocity', 0)
            tid    = t['track_id']
            colour = BEHAVIOUR_COLOURS_BGR.get(beh, (160, 160, 160))

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

            # ID label
            id_text = f"#{tid} {beh}"
            (tw, th), _ = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), colour, -1)
            cv2.putText(out, id_text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

            # Centroid dot
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(out, (cx, cy), 4, colour, -1)

            # Velocity badge
            vel_text = f"v={vel:.1f}"
            cv2.putText(out, vel_text, (x1, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

            # Alert indicator
            if t.get('alert'):
                cv2.putText(out, '⚠ ALERT', (x1, y1 - th - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # HUD overlay
        hud_lines = [
            f"Animals: {len(track_list)}",
            f"FPS: {fps:.0f}",
            f"Mode: {'Demo' if self.demo_mode else 'Live'}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(out, line, (10, 24 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 1, cv2.LINE_AA)

        # Timestamp
        ts = time.strftime('%Y-%m-%d  %H:%M:%S')
        cv2.putText(out, ts, (w - 220, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        return out

    @staticmethod
    def _placeholder_frame(msg: str) -> np.ndarray:
        """Generate a dark placeholder frame when camera is unavailable."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (18, 28, 46)   # dark blue
        cv2.putText(frame, msg, (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)
        cv2.putText(frame, 'SmartDairy AI Monitor', (120, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 160, 120), 1)
        ts = time.strftime('%H:%M:%S')
        cv2.putText(frame, ts, (260, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        return frame

    # ── Internal publish ───────────────────────────────────────────────────
    def _publish(self, frame, track_list, fps):
        # Encode JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with self._frame_lock:
            self._latest_jpeg = jpeg.tobytes()

        # Update stats snapshot
        beh_summary = BehaviourAnalyzer.summary(self.tracker.active_tracks)
        serialisable_tracks = []
        for t in track_list:
            serialisable_tracks.append({
                'track_id':  t['track_id'],
                'behaviour': t.get('behaviour', 'Detecting…'),
                'velocity':  t.get('velocity', 0),
                'alert':     t.get('alert'),
                'age':       t.get('age', 0),
                'label':     t.get('label', 'Animal'),
                'conf':      t.get('conf', 0),
            })

        with self._stats_lock:
            self._stats = {
                'animal_count':  len(track_list),
                'behaviours':    beh_summary,
                'tracks':        serialisable_tracks,
                'fps':           fps,
                'alerts':        self.anomaly.recent_alerts(10),
                'alert_counts':  self.anomaly.alert_counts(),
                'source_active': self.cap is not None and self.cap.isOpened() if self.cap else False,
            }

    # ── Public accessors ───────────────────────────────────────────────────
    def get_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def get_stats(self) -> dict:
        with self._stats_lock:
            import copy
            return copy.deepcopy(self._stats)

    # ── MJPEG generator ────────────────────────────────────────────────────
    def mjpeg_generator(self):
        """Yield multipart MJPEG frames for Flask streaming response."""
        while True:
            jpeg = self.get_jpeg()
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.04)   # ~25 fps cap
