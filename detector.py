"""Vehicle detection, tracking, and traffic metrics.

The YOLO part (`iter_detections`) is kept separate from the maths
(`summarize`) so the metrics can be tested without a model or a GPU.
"""
from __future__ import annotations

import math
import os
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median
from typing import Callable, Iterable, Iterator, Optional

import cv2

# COCO class ids that count as vehicles.
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --- Tunable settings (all overridable with environment variables) ----------
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov8n.pt")
# Look at every Nth frame. 3 is ~3x faster than 1 with little loss for counting.
VID_STRIDE = max(1, int(os.environ.get("VID_STRIDE", "3")))
# Only analyze the first N seconds of a video (protects small servers).
MAX_VIDEO_SECONDS = int(os.environ.get("MAX_VIDEO_SECONDS", "120"))
# Congestion = average vehicles visible per analyzed frame. Tune these to your
# camera: a wide highway view and a narrow side street need different values.
CONGESTION_LIGHT = float(os.environ.get("CONGESTION_LIGHT", "3"))
CONGESTION_MODERATE = float(os.environ.get("CONGESTION_MODERATE", "8"))
# Image size YOLO works at. Smaller = less memory and faster, but small/far cars are missed.
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))
# A vehicle must be seen at least this many times before we trust its speed.
MIN_SPEED_HITS = 3


@dataclass
class Zone:
    """A rectangular lane/zone, in normalized 0-1 coordinates."""
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int


# One detection: (class_name, track_id, x1, y1, x2, y2) in pixels.
Detection = tuple[str, int, float, float, float, float]
DetectionSource = Callable[[str, VideoInfo], Iterable[list[Detection]]]


# --- Video helpers -----------------------------------------------------------
def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise ValueError("That file could not be opened as a video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise ValueError("That file does not contain readable video frames.")
    return VideoInfo(fps=fps, width=width, height=height, frame_count=max(frames, 0))


def extract_preview(path: str, max_width: int = 960) -> tuple[bytes, int, int]:
    """Return (jpeg_bytes, width, height) for a frame to draw lane zones on."""
    info = probe_video(path)
    cap = cv2.VideoCapture(path)
    try:
        # A frame about one second in is usually more representative than frame 0.
        target = min(int(info.fps), max(info.frame_count - 1, 0))
        if target > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok:  # seeking failed; fall back to the very first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise ValueError("Could not read a frame from that video.")
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("Could not encode the preview frame.")
    return buf.tobytes(), w, h


# --- YOLO ---------------------------------------------------------------------
_model = None
_model_lock = threading.Lock()


def load_model():
    """Load YOLO once and reuse it. The first call downloads ~6 MB of weights."""
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO  # imported lazily: torch is slow to import
            _model = YOLO(YOLO_MODEL)
    return _model


def iter_detections(path: str, info: VideoInfo) -> Iterator[list[Detection]]:
    """Yield the vehicle detections for each analyzed frame."""
    model = load_model()
    results = model.track(
        source=path,
        classes=list(VEHICLE_CLASSES),
        tracker="bytetrack.yaml",
        persist=False,  # one call = one video; don't leak tracks between uploads
        stream=True,
        verbose=False,
        vid_stride=VID_STRIDE,
        imgsz=YOLO_IMGSZ,
    )
    for r in results:
        frame: list[Detection] = []
        if r.boxes is not None and r.boxes.id is not None:
            for cls_id, track_id, xyxy in zip(
                r.boxes.cls.tolist(), r.boxes.id.tolist(), r.boxes.xyxy.tolist()
            ):
                name = VEHICLE_CLASSES.get(int(cls_id))
                if name:
                    x1, y1, x2, y2 = xyxy
                    frame.append((name, int(track_id), x1, y1, x2, y2))
        yield frame


# --- Metrics ------------------------------------------------------------------
def classify_congestion(avg_concurrent: float) -> str:
    if avg_concurrent < CONGESTION_LIGHT:
        return "light"
    if avg_concurrent < CONGESTION_MODERATE:
        return "moderate"
    return "heavy"


def summarize(
    frames: Iterable[list[Detection]],
    info: VideoInfo,
    zones: list[Zone],
    road_width_m: Optional[float],
    seconds_per_step: float,
    *,
    max_steps: Optional[int] = None,
    expected_steps: Optional[int] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> dict:
    """Turn per-frame detections into counts, congestion, zones and speed."""
    meters_per_pixel = (
        road_width_m / info.width if road_width_m and road_width_m > 0 else None
    )

    class_votes: dict[int, Counter] = defaultdict(Counter)   # track -> class counts
    zone_votes: dict[int, Counter] = defaultdict(Counter)    # track -> zone counts
    positions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    concurrent: list[int] = []

    it = iter(frames)
    step = 0
    truncated = False
    for dets in it:
        step += 1
        concurrent.append(len(dets))
        for name, track_id, x1, y1, x2, y2 in dets:
            class_votes[track_id][name] += 1
            # Bottom-centre of the box is where the vehicle meets the road,
            # which is more stable than the box centre under camera perspective.
            cx, cy = (x1 + x2) / 2, y2
            positions[track_id].append((step, cx, cy))
            nx, ny = cx / info.width, cy / info.height
            for z in zones:  # first matching zone wins if zones overlap
                if z.contains(nx, ny):
                    zone_votes[track_id][z.name] += 1
                    break
        if progress_cb and expected_steps:
            progress_cb(min(step / expected_steps, 1.0))
        if max_steps is not None and step >= max_steps:
            truncated = next(it, None) is not None
            break
    if hasattr(it, "close"):
        it.close()

    # A tracker can briefly flip a vehicle's class (truck <-> car). Give each
    # track exactly one class: its most frequent one. Otherwise it is counted twice.
    by_type: Counter = Counter()
    for votes in class_votes.values():
        by_type[votes.most_common(1)[0][0]] += 1
    total = sum(by_type.values())

    avg_concurrent = sum(concurrent) / len(concurrent) if concurrent else 0.0

    zone_counts = None
    if zones:
        zone_counts = {z.name: 0 for z in zones}
        for votes in zone_votes.values():
            zone_counts[votes.most_common(1)[0][0]] += 1

    avg_speed = None
    if meters_per_pixel:
        speeds = []
        for pts in positions.values():
            if len(pts) < MIN_SPEED_HITS:
                continue
            (s0, x0, y0), (s1, x1_, y1_) = pts[0], pts[-1]
            elapsed = (s1 - s0) * seconds_per_step
            if elapsed <= 0:
                continue
            pixels = math.hypot(x1_ - x0, y1_ - y0)
            speeds.append(pixels / elapsed * meters_per_pixel * 3.6)
        if speeds:
            # Median: one tracker ID-switch shouldn't skew the result.
            avg_speed = round(median(speeds), 1)

    if progress_cb:
        progress_cb(1.0)
    return {
        "total": total,
        "by_type": dict(by_type),
        "congestion": classify_congestion(avg_concurrent),
        "avg_concurrent": round(avg_concurrent, 1),
        "avg_speed_kmh": avg_speed,
        "zone_counts": zone_counts,
        "frames_processed": step,
        "seconds_analyzed": round(step * seconds_per_step, 1),
        "truncated": truncated,
    }


def analyze_video(
    path: str,
    zones: list[Zone],
    road_width_m: Optional[float] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
    detections: Optional[DetectionSource] = None,
) -> dict:
    """Analyze one video file. `detections` can be swapped out in tests."""
    info = probe_video(path)
    seconds_per_step = VID_STRIDE / info.fps
    max_steps = int(MAX_VIDEO_SECONDS * info.fps / VID_STRIDE) if MAX_VIDEO_SECONDS > 0 else None
    expected = math.ceil(info.frame_count / VID_STRIDE) if info.frame_count else None
    if expected and max_steps:
        expected = min(expected, max_steps)
    source = detections or iter_detections
    return summarize(
        source(path, info),
        info,
        zones,
        road_width_m,
        seconds_per_step,
        max_steps=max_steps,
        expected_steps=expected,
        progress_cb=progress_cb,
    )
