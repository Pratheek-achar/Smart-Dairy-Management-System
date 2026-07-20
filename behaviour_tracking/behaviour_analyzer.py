"""
behaviour_analyzer.py – Kinematic Behaviour Classification Engine
Smart Dairy Livestock Monitoring System

Classifies animal behaviour based on velocity, variance, and movement patterns
extracted from DeepSORT track histories.  No external dataset required.

Behaviour Classes:
  Resting    – very low velocity
  Feeding    – low velocity with slight oscillation
  Walking    – moderate, steady velocity
  Stress     – high velocity variance / erratic movement
  Aggressive – sudden velocity spike + direction changes
  Isolated   – centroid far from herd centre
"""

import numpy as np
import time
from collections import Counter

# ── Kinematic thresholds (tunable) ────────────────────────────────────────────
THRESHOLDS = {
    'resting_vel':    1.5,   # px/frame – almost stationary
    'feeding_vel':    4.0,   # px/frame – slow purposeful movement
    'walking_vel':   12.0,   # px/frame – normal walk
    'stress_var':    30.0,   # variance threshold for erratic movement
    'aggressive_vel': 20.0,  # px/frame – fast / sudden lunge
    'aggressive_dir':  5,    # direction changes in history window
    'isolated_px':   200,    # distance from herd centroid (pixels)
}

BEHAVIOUR_COLOURS = {
    'Resting':    (100, 180, 255),  # blue
    'Feeding':    (0,   200,  80),  # green
    'Walking':    (255, 200,  50),  # yellow
    'Stress':     (255, 140,   0),  # orange
    'Aggressive': (0,    50, 255),  # deep red-blue
    'Isolated':   (180,  50, 255),  # purple
    'Detecting…': (160, 160, 160),  # grey
}


class BehaviourAnalyzer:
    """
    Assigns behaviour labels to each tracked animal.
    Works by analysing the kinematic features of each Track object.
    """

    def __init__(self):
        self.history: dict[int, list] = {}   # tid → list of behaviour labels
        self.WINDOW = 20                      # smoothing window (frames)

    def analyse(self, tracks_dict: dict) -> dict[int, str]:
        """
        Classify behaviour for all active tracks.

        Args:
            tracks_dict: {track_id: Track} from Tracker.active_tracks

        Returns:
            {track_id: behaviour_label}
        """
        behaviours = {}

        # Compute herd centroid for isolation detection
        centroids = [t.centroid for t in tracks_dict.values()]
        herd_cx = int(np.mean([c[0] for c in centroids])) if centroids else 0
        herd_cy = int(np.mean([c[1] for c in centroids])) if centroids else 0

        for tid, track in tracks_dict.items():
            raw = self._classify_single(track, herd_cx, herd_cy)

            # Temporal smoothing — take majority vote over last N frames
            if tid not in self.history:
                self.history[tid] = []
            self.history[tid].append(raw)
            if len(self.history[tid]) > self.WINDOW:
                self.history[tid] = self.history[tid][-self.WINDOW:]

            # Majority vote
            smoothed = Counter(self.history[tid]).most_common(1)[0][0]
            behaviours[tid] = smoothed
            track.behaviour = smoothed   # update Track object in-place

        # Clean history for removed tracks
        self.history = {k: v for k, v in self.history.items() if k in tracks_dict}
        return behaviours

    def _classify_single(self, track, herd_cx: int, herd_cy: int) -> str:
        """Rule-based classifier for a single track."""
        T   = THRESHOLDS
        vel = track.mean_velocity()
        var = track.velocity_variance()
        dch = track.direction_changes()
        cx, cy = track.centroid

        # Isolation check (must have at least 2 animals for this to mean anything)
        dist_from_herd = np.hypot(cx - herd_cx, cy - herd_cy)
        if dist_from_herd > T['isolated_px'] and len(track.history) > 10:
            return 'Isolated'

        # Aggressive: very fast + many direction reversals
        if vel > T['aggressive_vel'] and dch > T['aggressive_dir']:
            return 'Aggressive'

        # Stress: high variance even without high average speed
        if var > T['stress_var']:
            return 'Stress'

        # Walking
        if T['feeding_vel'] < vel <= T['walking_vel']:
            return 'Walking'

        # Feeding: low vel but not stationary — slight oscillation
        if T['resting_vel'] < vel <= T['feeding_vel']:
            return 'Feeding'

        # Resting
        return 'Resting'

    @staticmethod
    def get_colour(behaviour: str) -> tuple:
        """Return BGR colour for drawing."""
        return BEHAVIOUR_COLOURS.get(behaviour, (160, 160, 160))

    @staticmethod
    def summary(tracks_dict: dict) -> dict:
        """Return behaviour distribution counts."""
        counts = Counter(t.behaviour for t in tracks_dict.values())
        return dict(counts)
