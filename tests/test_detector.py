import cv2
import numpy as np
import pytest

import detector
from detector import Zone, VideoInfo, summarize, analyze_video, classify_congestion

INFO = VideoInfo(fps=30.0, width=640, height=360, frame_count=90)


def make_video(path, seconds=3, fps=30, size=(640, 360)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for i in range(seconds * fps):
        frame = np.full((size[1], size[0], 3), 40, np.uint8)
        cv2.rectangle(frame, (i * 5 % 600, 150), (i * 5 % 600 + 40, 190), (200, 200, 200), -1)
        w.write(frame)
    w.release()
    return path


def car(track, x, y2=200, name="car", w=40):
    return (name, track, x, y2 - 20, x + w, y2)


def test_probe_and_preview(tmp_path):
    p = make_video(tmp_path / "v.mp4")
    info = detector.probe_video(str(p))
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(30, abs=1)
    jpg, w, h = detector.extract_preview(str(p), max_width=320)
    assert (w, h) == (320, 180)
    assert jpg[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_garbage_file_is_a_clear_error(tmp_path):
    p = tmp_path / "x.mp4"
    p.write_bytes(b"not a video at all")
    with pytest.raises(ValueError):
        detector.probe_video(str(p))


def test_class_flip_counts_vehicle_once():
    # Track 1 is mostly a truck but flickers to "car" once. It is ONE vehicle.
    frames = [[car(1, 10, name="truck")], [car(1, 20, name="truck")],
              [car(1, 30, name="car")], [car(1, 40, name="truck")]]
    r = summarize(frames, INFO, [], None, 0.1)
    assert r["total"] == 1
    assert r["by_type"] == {"truck": 1}


def test_unique_vehicles_not_per_frame_detections():
    frames = [[car(1, 10 + i), car(2, 300 + i, name="bus")] for i in range(20)]
    r = summarize(frames, INFO, [], None, 0.1)
    assert r["total"] == 2
    assert r["by_type"] == {"car": 1, "bus": 1}
    assert r["avg_concurrent"] == 2.0


def test_congestion_levels(monkeypatch):
    monkeypatch.setattr(detector, "CONGESTION_LIGHT", 3)
    monkeypatch.setattr(detector, "CONGESTION_MODERATE", 8)
    assert classify_congestion(0) == "light"
    assert classify_congestion(2.9) == "light"
    assert classify_congestion(3) == "moderate"
    assert classify_congestion(7.9) == "moderate"
    assert classify_congestion(8) == "heavy"


def test_zone_counts_use_majority_zone_and_keep_empty_zones():
    left, right, empty = (Zone("Left", 0, 0, .5, 1), Zone("Right", .5, 0, 1, 1),
                          Zone("Shoulder", .9, .9, 1, 1))
    # track 1 stays left; track 2 starts left, then spends most frames on the right
    frames = [[car(1, 50), car(2, 100)]] + [[car(1, 60), car(2, 400)] for _ in range(5)]
    r = summarize(frames, INFO, [left, right, empty], None, 0.1)
    assert r["zone_counts"] == {"Left": 1, "Right": 1, "Shoulder": 0}


def test_no_zones_gives_none():
    assert summarize([[car(1, 5)]], INFO, [], None, 0.1)["zone_counts"] is None


def test_speed_estimate_is_correct():
    # 100 px/s with 20 m across 640 px (0.03125 m/px) = 3.125 m/s = 11.25 km/h
    frames = [[car(1, 10 * i)] for i in range(11)]
    r = summarize(frames, INFO, [], 20.0, 0.1)
    assert r["avg_speed_kmh"] == pytest.approx(11.2, abs=0.1)


def test_speed_omitted_without_calibration_or_enough_hits():
    frames = [[car(1, 10 * i)] for i in range(11)]
    assert summarize(frames, INFO, [], None, 0.1)["avg_speed_kmh"] is None
    assert summarize([[car(1, 0)], [car(1, 50)]], INFO, [], 20.0, 0.1)["avg_speed_kmh"] is None


def test_speed_uses_median_so_one_outlier_does_not_skew():
    slow = [[car(1, 10 * i), car(2, 10 * i + 200), car(3, 10 * i + 400)] for i in range(11)]
    # track 4 'teleports' (e.g. a tracker ID switch): enormous apparent speed
    jump = [[car(4, 0)]] + [[car(4, 0)]] * 5 + [[car(4, 600)]]
    r = summarize([a + b for a, b in zip(slow, jump)], INFO, [], 20.0, 0.1)
    assert r["avg_speed_kmh"] == pytest.approx(11.2, abs=0.1)


def test_truncation_flag_and_progress():
    frames = [[car(1, i)] for i in range(10)]
    seen = []
    r = summarize(frames, INFO, [], None, 0.1, max_steps=4, expected_steps=4,
                  progress_cb=seen.append)
    assert r["frames_processed"] == 4 and r["truncated"] is True
    assert seen == sorted(seen) and seen[-1] == 1.0
    r2 = summarize([[car(1, i)] for i in range(4)], INFO, [], None, 0.1, max_steps=4)
    assert r2["truncated"] is False  # ended exactly at the limit: nothing was cut


def test_empty_video_is_all_zero():
    r = summarize([[], [], []], INFO, [], 20.0, 0.1)
    assert r["total"] == 0 and r["congestion"] == "light" and r["avg_speed_kmh"] is None


def test_analyze_video_end_to_end_with_fake_detector(tmp_path):
    p = make_video(tmp_path / "v.mp4", seconds=3, fps=30)

    def fake_source(path, info):  # 90 source frames / stride 3 = 30 analyzed frames
        for i in range(30):
            yield [car(1, i * 10), car(2, 300, name="truck")]

    progress = []
    r = analyze_video(str(p), [Zone("All", 0, 0, 1, 1)], 20.0, progress.append, fake_source)
    assert r["total"] == 2 and r["by_type"] == {"car": 1, "truck": 1}
    assert r["frames_processed"] == 30
    assert r["seconds_analyzed"] == pytest.approx(3.0, abs=0.2)
    assert r["zone_counts"] == {"All": 2}
    assert r["congestion"] == "light" and r["truncated"] is False
    assert progress[-1] == 1.0
