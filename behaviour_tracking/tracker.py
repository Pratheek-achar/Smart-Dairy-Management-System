"""
tracker.py – DeepSORT Animal Tracking Wrapper
Smart Dairy Livestock Monitoring System

Wraps deep-sort-realtime to assign persistent unique IDs to each animal
across video frames and maintain their movement history.
"""

import time
import numpy as np
from collections import defaultdict, deque

# Maximum frames to keep in movement history per animal
HISTORY_LEN = 60   # ~2 seconds at 30fps


class Track:
    """Represents a single tracked animal with movement history."""

    def __init__(self, track_id: int, bbox: list, label: str, conf: float):
        self.track_id   = track_id
        self.label      = label
        self.conf       = conf
        self.bbox       = bbox              # current [x1,y1,x2,y2]
        self.centroid   = self._centroid(bbox)
        self.history    = deque(maxlen=HISTORY_LEN)  # (cx,cy) per frame
        self.history.append(self.centroid)
        self.first_seen = time.time()
        self.last_seen  = time.time()
        self.behaviour  = 'Detecting…'
        self.alert      = None

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def update(self, bbox: list, label: str, conf: float):
        self.bbox      = bbox
        self.label     = label
        self.conf      = conf
        self.centroid  = self._centroid(bbox)
        self.history.append(self.centroid)
        self.last_seen = time.time()

    # ── Kinematic features ─────────────────────────────────────────────────
    def mean_velocity(self) -> float:
        """Average pixel displacement per frame over recent history."""
        pts = list(self.history)
        if len(pts) < 2:
            return 0.0
        dists = [np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                 for i in range(1, len(pts))]
        return float(np.mean(dists))

    def velocity_variance(self) -> float:
        """Variance of per-frame displacements — high = erratic movement."""
        pts = list(self.history)
        if len(pts) < 3:
            return 0.0
        dists = [np.hypot(pts[i][0]-pts[i-1][0], pts[i][1]-pts[i-1][1])
                 for i in range(1, len(pts))]
        return float(np.var(dists))

    def direction_changes(self) -> int:
        """Count of significant direction reversals in history."""
        pts = list(self.history)
        if len(pts) < 3:
            return 0
        changes = 0
        for i in range(2, len(pts)):
            dx1 = pts[i-1][0] - pts[i-2][0]
            dy1 = pts[i-1][1] - pts[i-2][1]
            dx2 = pts[i][0]   - pts[i-1][0]
            dy2 = pts[i][1]   - pts[i-1][1]
            dot = dx1*dx2 + dy1*dy2
            if dot < -5:      # angle > 90°
                changes += 1
        return changes

    def inactive_duration(self) -> float:
        """Seconds since animal last had meaningful movement."""
        return time.time() - self.last_seen

    @property
    def age_seconds(self) -> float:
        return time.time() - self.first_seen


class Tracker:
    """
    Multi-animal tracker using DeepSORT.
    Falls back to simple centroid-based tracking if DeepSORT is unavailable.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 2):
        self.max_age  = max_age
        self.min_hits = min_hits
        self._ds      = self._init_deepsort(max_age, min_hits)
        self.tracks: dict[int, Track] = {}   # track_id → Track
        self._next_id = 1                    # fallback ID counter

    @staticmethod
    def _init_deepsort(max_age, min_hits):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
            ds = DeepSort(max_age=max_age, n_init=min_hits, nn_budget=100)
            print("[Tracker] DeepSORT initialised.")
            return ds
        except ImportError:
            print("[Tracker] deep-sort-realtime not found — using fallback tracker.")
            return None

    # ── Public API ─────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray, detections: list[dict]) -> list[dict]:
        """
        Update tracker with new detections.

        Args:
            frame      : Current BGR frame (needed by DeepSORT appearance model)
            detections : List from Detector.detect()

        Returns:
            List of active track dicts:
            {
                'track_id': int,
                'bbox'    : [x1,y1,x2,y2],
                'label'   : str,
                'conf'    : float,
                'behaviour': str,
                'alert'   : str | None,
                'velocity': float
            }
        """
        if self._ds is not None:
            return self._update_deepsort(frame, detections)
        return self._update_fallback(detections)

    # ── DeepSORT path ──────────────────────────────────────────────────────
    def _update_deepsort(self, frame, detections):
        # Convert to DeepSORT format: ([left,top,w,h], confidence, label)
        raw = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            raw.append(([x1, y1, x2-x1, y2-y1], d['confidence'], d['label']))

        try:
            ds_tracks = self._ds.update_tracks(raw, frame=frame)
        except Exception as e:
            print(f"[Tracker] DeepSORT update error: {e}")
            return []

        active_ids = set()
        for t in ds_tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = list(map(int, t.to_ltrb()))
            lbl  = t.det_class or 'Animal'
            conf = float(t.det_conf) if t.det_conf else 0.5

            if tid not in self.tracks:
                self.tracks[tid] = Track(tid, ltrb, lbl, conf)
            else:
                self.tracks[tid].update(ltrb, lbl, conf)
            active_ids.add(tid)

        # Remove stale tracks
        self.tracks = {k: v for k, v in self.tracks.items() if k in active_ids}
        return self._build_output()

    # ── Fallback centroid tracker ──────────────────────────────────────────
    def _update_fallback(self, detections):
        if not detections:
            self.tracks.clear()
            return []

        # Simple nearest-centroid matching
        unmatched_tracks = set(self.tracks.keys())
        used_dets = set()

        for i, det in enumerate(detections):
            cx = (det['bbox'][0] + det['bbox'][2]) // 2
            cy = (det['bbox'][1] + det['bbox'][3]) // 2
            best_id, best_dist = None, 80   # px threshold

            for tid in list(unmatched_tracks):
                tc = self.tracks[tid].centroid
                d  = np.hypot(cx - tc[0], cy - tc[1])
                if d < best_dist:
                    best_dist, best_id = d, tid

            if best_id is not None:
                self.tracks[best_id].update(det['bbox'], det['label'], det['confidence'])
                unmatched_tracks.discard(best_id)
                used_dets.add(i)
            else:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = Track(tid, det['bbox'], det['label'], det['confidence'])

        # Age out unmatched tracks
        for tid in unmatched_tracks:
            del self.tracks[tid]

        return self._build_output()

    def _build_output(self):
        return [{
            'track_id':  t.track_id,
            'bbox':      t.bbox,
            'label':     t.label,
            'conf':      t.conf,
            'behaviour': t.behaviour,
            'alert':     t.alert,
            'velocity':  round(t.mean_velocity(), 2),
            'age':       round(t.age_seconds, 1),
        } for t in self.tracks.values()]

    @property
    def active_tracks(self) -> dict:
        return self.tracks
