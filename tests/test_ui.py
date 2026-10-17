"""Drives the real page: upload -> (draw lane) -> analyze -> results -> history.

YOLO is replaced by a fake detection source; everything else is the real code.
"""
import cv2
import numpy as np
import pytest
from nicegui import ui
from nicegui.elements.upload_files import SmallFileUpload
from nicegui.testing import User

import database
import detector


@pytest.fixture
def clip_bytes(tmp_path):
    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for i in range(90):
        w.write(np.full((360, 640, 3), 50 + i % 50, np.uint8))
    w.release()
    return p.read_bytes()


@pytest.fixture(autouse=True)
def isolated_db_and_fake_yolo(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "data" / "history.db")
    database.init_db()

    def fake_source(path, info):  # 90 frames / stride 3 = 30 analyzed frames
        for i in range(30):
            # a car crossing the left half, a truck sitting in the right half
            yield [("car", 1, i * 5, 120, i * 5 + 40, 160), ("truck", 2, 500, 150, 560, 220)]

    monkeypatch.setattr(detector, "iter_detections", fake_source)


async def upload(user: User, data: bytes, name="junction.mp4", wait_ready=True):
    """Simulate choosing a file. The handler runs as a background task (just like in a
    browser), so wait until the page says READY before touching the buttons it enables."""
    el = user.find(ui.upload).elements.pop()
    await el.handle_uploads([SmallFileUpload(name=name, content_type="video/mp4", _data=data)])
    if wait_ready:
        await user.should_see("READY", retries=50)


async def test_page_starts_idle(user: User):
    await user.open("/")
    await user.should_see("NO FOOTAGE")
    await user.should_see("History")
    await user.should_see("No analyses yet")


async def test_upload_analyze_results_and_history(user: User, clip_bytes):
    await user.open("/")
    await upload(user, clip_bytes)
    await user.should_see("READY")
    await user.should_see("junction.mp4")

    user.find("Analyze footage").click()
    await user.should_see("Vehicles by type", retries=100)
    await user.should_see("Cars")
    await user.should_see("Trucks")
    await user.should_see("Light traffic")          # 2 vehicles in view on average
    await user.should_see("2 vehicles")             # the sign
    await user.should_not_see("Vehicles by lane")   # no lanes drawn

    rows = database.load_history()
    assert len(rows) == 1 and rows[0]["total"] == 2 and rows[0]["filename"] == "junction.mp4"
    await user.should_not_see("No analyses yet")


async def test_unreadable_file_gives_clear_message(user: User):
    await user.open("/")
    await upload(user, b"this is not a video", name="bad.mp4", wait_ready=False)
    await user.should_see("could not be opened")
    await user.should_see("NO FOOTAGE")  # nothing changed


async def test_draw_lane_then_count_by_lane(user: User, clip_bytes):
    await user.open("/")
    await upload(user, clip_bytes)
    user.find("Draw lanes on this video").click()
    await user.should_see("Drag a box over each lane", retries=50)

    img = user.find(ui.interactive_image)
    # preview is 640x360; drag a box over the LEFT half (where the car is)
    img.trigger("mouse", {"mouse_event_type": "mousedown", "image_x": 5, "image_y": 5})
    img.trigger("mouse", {"mouse_event_type": "mouseup", "image_x": 320, "image_y": 350})
    await user.should_see("Lane 1")

    user.find("Analyze footage").click()
    await user.should_see("Vehicles by lane", retries=100)
    # 1 vehicle (the car) sits in Lane 1; the truck is outside every lane
    assert database.load_history()[-1]["zone_counts"] == {"Lane 1": 1}


async def test_tap_two_corners_also_makes_a_lane(user: User, clip_bytes):
    await user.open("/")
    await upload(user, clip_bytes)
    user.find("Draw lanes on this video").click()
    await user.should_see("tap one corner", retries=50)
    img = user.find(ui.interactive_image)
    for x, y in ((10, 10), (300, 300)):  # two taps = down+up at the same point
        img.trigger("mouse", {"mouse_event_type": "mousedown", "image_x": x, "image_y": y})
        img.trigger("mouse", {"mouse_event_type": "mouseup", "image_x": x, "image_y": y})
    await user.should_see("Lane 1")


async def test_speed_shown_when_road_width_given(user: User, clip_bytes):
    await user.open("/")
    await upload(user, clip_bytes)
    user.find(ui.number).type("20")
    user.find("Analyze footage").click()
    await user.should_see("km/h", retries=100)
    assert database.load_history()[-1]["avg_speed_kmh"] is not None


async def test_analysis_failure_is_reported_not_silent(user: User, clip_bytes, monkeypatch, caplog):
    def boom(path, info):
        raise RuntimeError("model exploded")
        yield

    monkeypatch.setattr(detector, "iter_detections", boom)
    await user.open("/")
    await upload(user, clip_bytes)
    user.find("Analyze footage").click()
    await user.should_see("Analysis failed", retries=100)
    await user.should_see("ERROR")
    assert database.load_history() == []
    # the real cause must reach the server log, since the notification points people there
    assert any("model exploded" in (r.exc_text or r.getMessage()) or r.exc_info for r in caplog.records
               if r.levelname == "ERROR")
    caplog.clear()  # this error was expected; don't let the harness flag it
