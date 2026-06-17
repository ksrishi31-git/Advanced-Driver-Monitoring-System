"""
incident_recorder.py
────────────────────
Drop-in module for the Driver Monitoring System.

Usage inside drowsiness_detection.py:
    from incident_recorder import IncidentRecorder
    recorder = IncidentRecorder()
    recorder.feed(frame)                       # call every frame
    recorder.trigger("L2", "Eye closure 3s")  # call when alarm fires
    recorder.close()                           # call on quit
"""

import cv2
import os
import threading
from collections import deque
from datetime import datetime


class IncidentRecorder:
    """
    Maintains a rolling ~5-second frame buffer.
    When trigger() is called it saves:
      • 5 s BEFORE the alarm  (from the rolling buffer)
      • 5 s AFTER  the alarm  (captured in real-time)
    Total clip = ~10 seconds, saved as MP4 in incidents/ folder.
    """

    PRE_SEC   = 5      # seconds of pre-alarm footage to keep
    POST_SEC  = 5      # seconds of post-alarm footage to record
    OUT_DIR   = "incidents"
    FPS_EST   = 25     # estimate; actual fps passed at trigger time

    def __init__(self):
        os.makedirs(self.OUT_DIR, exist_ok=True)
        self._buf          : deque        = deque()   # (frame_copy,) rolling
        self._buf_max      : int          = self.PRE_SEC * self.FPS_EST
        self._recording    : bool         = False
        self._post_frames  : list         = []
        self._post_needed  : int          = 0
        self._writer       : cv2.VideoWriter | None = None
        self._lock                        = threading.Lock()
        self._pending_meta : dict | None  = None
        self._incident_log : list[dict]   = []   # in-memory list for report

    # ── called every frame ───────────────────────────────────────────────────
    def feed(self, frame: "np.ndarray", fps: float = 25.0) -> None:
        """
        Add frame to the rolling pre-buffer.
        If a recording is active, also append to post-buffer.
        """
        f = frame.copy()
        with self._lock:
            # update rolling buffer max based on live fps
            self._buf_max = max(1, int(self.PRE_SEC * fps))
            self._buf.append(f)
            while len(self._buf) > self._buf_max:
                self._buf.popleft()

            if self._recording:
                self._post_frames.append(f)
                if len(self._post_frames) >= self._post_needed:
                    self._flush_clip(fps)

    # ── called when alarm fires ───────────────────────────────────────────────
    def trigger(self, level: str, reason: str, fps: float = 25.0) -> None:
        """
        Start saving an incident clip.
        Silently skipped if a recording is already in progress.
        """
        with self._lock:
            if self._recording:
                return   # don't nest recordings

            self._recording   = True
            self._post_needed = max(1, int(self.POST_SEC * fps))
            self._post_frames = []
            self._pending_meta = {
                "time"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level" : level,
                "reason": reason,
            }
            # snapshot the pre-buffer right now
            self._pre_snapshot = list(self._buf)

        print(f"[INCIDENT] Recording started — {level}: {reason}")

    # ── internal flush ───────────────────────────────────────────────────────
    def _flush_clip(self, fps: float) -> None:
        """Write pre + post frames to disk (called with lock held)."""
        meta  = self._pending_meta
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(
            self.OUT_DIR,
            f"incident_{ts}_{meta['level']}.mp4"
        )

        frames = self._pre_snapshot + self._post_frames
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(fname, fourcc, fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()
            size_kb = os.path.getsize(fname) // 1024
            print(f"[INCIDENT] Saved → {fname}  ({size_kb} KB, {len(frames)} frames)")

            # record for session report
            self._incident_log.append({
                **meta,
                "file"    : fname,
                "frames"  : len(frames),
                "duration": f"{len(frames)/max(fps,1):.1f}s",
            })

        # reset state
        self._recording    = False
        self._post_frames  = []
        self._pre_snapshot = []
        self._pending_meta = None

    def get_log(self) -> list:
        return list(self._incident_log)

    def close(self) -> None:
        """Force-flush any in-progress recording before exit."""
        with self._lock:
            if self._recording and self._post_frames:
                self._flush_clip(self.FPS_EST)