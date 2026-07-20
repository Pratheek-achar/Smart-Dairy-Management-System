"""
anomaly_detector.py – Real-Time Alert Generation Engine
Smart Dairy Livestock Monitoring System

Monitors behaviour patterns over time and triggers intelligent alerts
when abnormal conditions are detected.

Alert Levels:
  INFO     – Informational (green)
  WARNING  – Attention needed (yellow)
  CRITICAL – Immediate action required (red)
"""

import time
from collections import deque, defaultdict

# ── Alert thresholds ───────────────────────────────────────────────────────────
ALERT_RULES = {
    'inactivity': {
        'behaviour': 'Resting',
        'min_consecutive_sec': 120,   # 2 minutes of resting = info
        'level': 'WARNING',
        'message': 'Animal inactive for extended period – possible illness'
    },
    'stress_sustained': {
        'behaviour': 'Stress',
        'min_consecutive_sec': 15,
        'level': 'WARNING',
        'message': 'Sustained stress behaviour detected'
    },
    'aggressive': {
        'behaviour': 'Aggressive',
        'min_consecutive_sec': 5,
        'level': 'CRITICAL',
        'message': 'Aggressive movement detected – herd conflict risk'
    },
    'isolated': {
        'behaviour': 'Isolated',
        'min_consecutive_sec': 30,
        'level': 'WARNING',
        'message': 'Animal isolated from herd – monitor closely'
    },
}

MAX_ALERTS = 50   # keep at most this many in the buffer


class AnomalyDetector:
    """
    Stateful alert engine.
    Tracks how long each animal has been in a given behaviour and
    emits alerts when thresholds are crossed.
    """

    def __init__(self):
        # {track_id: {'behaviour': str, 'since': float}}
        self._state: dict[int, dict] = {}
        # Deque of emitted alert dicts (newest first)
        self.alerts: deque = deque(maxlen=MAX_ALERTS)
        # Suppress repeated alerts: {(tid, rule_key): last_emitted_ts}
        self._suppress: dict = defaultdict(float)
        self.SUPPRESS_SEC = 60   # don't repeat same alert within 60 s

    # ── Public API ─────────────────────────────────────────────────────────
    def update(self, tracks_dict: dict) -> list[dict]:
        """
        Check all active tracks for anomalous conditions.

        Args:
            tracks_dict: {track_id: Track}

        Returns:
            List of NEW alert dicts emitted this frame:
            {
                'level':      'INFO' | 'WARNING' | 'CRITICAL',
                'message':    str,
                'track_id':   int,
                'behaviour':  str,
                'timestamp':  float
            }
        """
        now = time.time()
        new_alerts = []

        for tid, track in tracks_dict.items():
            beh = track.behaviour

            # Update state
            if tid not in self._state or self._state[tid]['behaviour'] != beh:
                self._state[tid] = {'behaviour': beh, 'since': now}

            duration = now - self._state[tid]['since']

            for rule_key, rule in ALERT_RULES.items():
                if beh != rule['behaviour']:
                    continue
                if duration < rule['min_consecutive_sec']:
                    continue
                suppress_key = (tid, rule_key)
                if now - self._suppress[suppress_key] < self.SUPPRESS_SEC:
                    continue

                alert = {
                    'level':     rule['level'],
                    'message':   rule['message'],
                    'track_id':  tid,
                    'behaviour': beh,
                    'duration':  round(duration, 1),
                    'timestamp': now,
                }
                self.alerts.appendleft(alert)
                new_alerts.append(alert)
                self._suppress[suppress_key] = now
                # Update track object
                track.alert = rule['message']

        # Clear stale state for removed animals
        self._state = {k: v for k, v in self._state.items() if k in tracks_dict}
        return new_alerts

    def add_system_alert(self, level: str, message: str):
        """Add a system-level alert (e.g., camera disconnected)."""
        self.alerts.appendleft({
            'level':     level,
            'message':   message,
            'track_id':  None,
            'behaviour': 'System',
            'duration':  0,
            'timestamp': time.time(),
        })

    def recent_alerts(self, n: int = 10) -> list[dict]:
        """Return the n most recent alerts."""
        return list(self.alerts)[:n]

    def alert_counts(self) -> dict:
        """Return counts by level."""
        counts = {'INFO': 0, 'WARNING': 0, 'CRITICAL': 0}
        for a in self.alerts:
            lvl = a.get('level', 'INFO')
            if lvl in counts:
                counts[lvl] += 1
        return counts
