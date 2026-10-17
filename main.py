"""Traffic monitor: a single-process Python web app (NiceGUI + YOLOv8).

Run locally:   python main.py        ->  http://localhost:8080
On Render:     start command `python main.py` (it reads $PORT).
"""
from __future__ import annotations

import asyncio
import html
import base64
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from nicegui import app, background_tasks, events, run, ui

import database
import detector
from detector import Zone, VideoInfo

logger = logging.getLogger("traffic-monitor")
logging.basicConfig(level=logging.INFO)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "100"))
LEVEL_COLOR = {"light": "#3DBA6E", "moderate": "#FFB81C", "heavy": "#E5484D"}
LEVEL_LABEL = {"light": "LIGHT", "moderate": "MODERATE", "heavy": "HEAVY"}
TYPE_LABEL = {"car": "Cars", "truck": "Trucks", "bus": "Buses", "motorcycle": "Motorcycles"}
ZONE_COLORS = ["#FFB81C", "#3DBA6E", "#5B8DEF", "#E5484D", "#B968E5", "#2EC4B6"]

# Only one analysis at a time: video + YOLO is memory-hungry on small servers.
ANALYSIS_LOCK = asyncio.Lock()

# --------------------------------------------------------------------------- style
ui.add_head_html(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600'
    '&family=Barlow+Semi+Condensed:wght@500;600;700&display=swap" rel="stylesheet">',
    shared=True,
)
ui.add_css(
    """
:root{--asphalt:#15171B;--panel:#1D2026;--edge:#2C313A;--paint:#ECE9E1;
      --muted:#98A0AD;--amber:#FFB81C;--go:#3DBA6E;--stop:#E5484D;}
body{background:var(--asphalt)!important;color:var(--paint);
     font-family:'Barlow',system-ui,-apple-system,'Segoe UI',sans-serif;}
.sign-font{font-family:'Barlow Semi Condensed','Barlow',system-ui,sans-serif;}
.muted{color:var(--muted);}
.panel{background:var(--panel)!important;border:1px solid var(--edge);
       border-radius:6px;box-shadow:none!important;}
/* a dashed lane marking, used as the divider */
.lane-rule{height:0;border:0;border-top:2px dashed #3A404B;width:100%;margin:4px 0;}

/* --- the message sign ------------------------------------------------- */
.vms{position:relative;background:#08090A;border:2px solid var(--edge);border-radius:6px;
     box-shadow:inset 0 0 0 4px #0F1114;padding:22px 26px;overflow:hidden;}
.vms .line{font-family:'Barlow Semi Condensed','Barlow',system-ui,sans-serif;font-weight:700;
     color:var(--amber);text-shadow:0 0 12px rgba(255,184,28,.55);
     letter-spacing:.06em;line-height:1.05;text-transform:uppercase;
     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.vms .big{font-size:clamp(34px,8.5vw,68px);}
.vms .small{font-size:clamp(16px,3.6vw,26px);opacity:.92;margin-top:6px;}
/* dot-matrix mask: dark gaps between 'LEDs' laid over the glowing text */
.vms::after{content:'';position:absolute;inset:0;pointer-events:none;
     background-image:radial-gradient(circle at center,transparent 0 38%,#08090A 44%);
     background-size:5px 5px;}
.signal{background:#0D0E10;border:2px solid var(--edge);border-radius:999px;
        padding:10px 8px;display:flex;flex-direction:column;gap:9px;align-items:center;}
.lamp{width:20px;height:20px;border-radius:50%;background:#262A30;transition:background .2s,box-shadow .2s;}
.lamp.go{background:var(--go);box-shadow:0 0 14px var(--go);}
.lamp.caution{background:var(--amber);box-shadow:0 0 14px var(--amber);}
.lamp.stop{background:var(--stop);box-shadow:0 0 14px var(--stop);}
.lamp.blink{animation:blink 1s steps(2,start) infinite;}
@keyframes blink{to{visibility:hidden;}}
@media (prefers-reduced-motion:reduce){.lamp.blink{animation:none;}}

/* --- Quasar tweaks to fit the palette ------------------------------------ */
.q-uploader{background:var(--panel)!important;color:var(--paint)!important;
            width:100%!important;max-width:none!important;box-shadow:none!important;
            border:1px dashed #3A404B;border-radius:6px;}
.q-uploader__header{background:#262A32!important;color:var(--paint)!important;}
.q-uploader__list{min-height:0!important;}
.q-field--outlined .q-field__control{border-radius:4px;}
:focus-visible{outline:2px solid var(--amber)!important;outline-offset:2px;}
.stat-num{font-family:'Barlow Semi Condensed','Barlow',sans-serif;font-weight:600;
          font-size:34px;line-height:1;}
""",
    shared=True,
)


# --------------------------------------------------------------------------- state
@dataclass
class Session:
    """Everything that belongs to one open browser tab."""
    workdir: Path
    path: Optional[Path] = None
    filename: str = ""
    info: Optional[VideoInfo] = None
    preview_size: Optional[tuple[int, int]] = None
    zones: list[Zone] = field(default_factory=list)
    drag_start: Optional[tuple[float, float]] = None
    corner: Optional[tuple[float, float]] = None
    busy: bool = False
    progress: float = 0.0


def zones_svg(zones: list[Zone], size: tuple[int, int], corner=None) -> str:
    w, h = size
    fs = max(14, round(w / 40))
    out = []
    for i, z in enumerate(zones):
        c = ZONE_COLORS[i % len(ZONE_COLORS)]
        x, y, rw, rh = z.x1 * w, z.y1 * h, (z.x2 - z.x1) * w, (z.y2 - z.y1) * h
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
            f'fill="{c}" fill-opacity="0.18" stroke="{c}" stroke-width="3"/>'
            f'<text x="{x + 6:.1f}" y="{y + fs + 4:.1f}" fill="#fff" font-size="{fs}" '
            f'font-weight="700" stroke="#000" stroke-width="3" paint-order="stroke">'
            f"{html.escape(z.name)}</text>"
        )
    if corner:
        out.append(f'<circle cx="{corner[0]:.1f}" cy="{corner[1]:.1f}" r="7" fill="#FFB81C" stroke="#000" stroke-width="2"/>')
    return "".join(out)


def unique_named(zones: list[Zone]) -> list[Zone]:
    """Blank or duplicate lane names would silently merge counts; fix them."""
    seen: dict[str, int] = {}
    fixed = []
    for i, z in enumerate(zones, 1):
        name = (z.name or "").strip() or f"Lane {i}"
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} ({seen[name]})"
        fixed.append(Zone(name, z.x1, z.y1, z.x2, z.y2))
    return fixed


# --------------------------------------------------------------------------- routes
@app.get("/healthz")
def healthz():
    return {"status": "ok"}


async def _warm_up():
    """Load the model in the background so the first analysis starts faster."""
    if os.environ.get("WARM_UP", "1") == "0":
        return
    try:
        await run.io_bound(detector.load_model)
        logger.info("YOLO model ready")
    except Exception:  # not fatal: it will be retried on the first analysis
        logger.exception("Model warm-up failed")


if not app.is_started:  # NiceGUI allows registering startup hooks only once per process
    app.on_startup(database.init_db)
    app.on_startup(lambda: background_tasks.create(_warm_up()))


# --------------------------------------------------------------------------- page
@ui.page("/")
def index():
    ui.dark_mode(True)
    ui.colors(primary="#FFB81C", positive="#3DBA6E", negative="#E5484D", warning="#FFB81C")

    s = Session(workdir=Path(tempfile.mkdtemp(prefix="trafficmon_")))
    ui.context.client.on_disconnect(lambda: shutil.rmtree(s.workdir, ignore_errors=True))

    with ui.column().classes("w-full max-w-3xl mx-auto gap-4 p-4 sm:p-6"):
        # ----- heading
        with ui.column().classes("gap-0"):
            ui.label("Traffic monitor").classes("sign-font text-3xl font-bold")
            ui.label("Count vehicles, read congestion and compare lanes from a video file.").classes("muted")

        # ----- the sign (status + result)
        with ui.row().classes("w-full items-stretch gap-3 no-wrap"):
            with ui.element("div").classes("vms grow").props('role=status aria-live=polite'):
                sign_1 = ui.label("NO FOOTAGE").classes("line big")
                sign_2 = ui.label("Choose a video below").classes("line small")
            with ui.element("div").classes("signal").props('aria-hidden=true'):
                lamps = {name: ui.element("div").classes("lamp") for name in ("stop", "caution", "go")}
        bar = ui.linear_progress(value=0, show_value=False, size="6px").classes("w-full")
        bar.visible = False

        def sign(l1: str, l2: str, lamp: Optional[str] = None, blink: bool = False):
            """Show two lines on the sign and light at most one lamp (stop / caution / go)."""
            sign_1.set_text(l1)
            sign_2.set_text(l2)
            for name, el in lamps.items():
                lit = lamp == name
                el.classes(replace="lamp" + (f" {name}" if lit else "") + (" blink" if lit and blink else ""))

        def lamp_for(level: str) -> str:
            return {"light": "go", "moderate": "caution", "heavy": "stop"}[level]

        # ----- footage card
        with ui.card().classes("panel w-full gap-3 p-4"):
            ui.label("Footage").classes("sign-font text-xl font-semibold")
            ui.label(
                f"MP4 works best, and short clips finish fastest. The first {detector.MAX_VIDEO_SECONDS} "
                f"seconds of a video are analyzed."
            ).classes("muted text-sm")

            async def on_upload(e: events.UploadEventArguments):
                name = e.file.name
                dest = s.workdir / f"video{Path(name).suffix.lower() or '.mp4'}"
                if s.path and s.path.exists():
                    s.path.unlink(missing_ok=True)
                await e.file.save(dest)
                upload.reset()
                try:
                    info = await run.io_bound(detector.probe_video, str(dest))
                except ValueError as err:
                    dest.unlink(missing_ok=True)
                    ui.notify(f"{err} Try an MP4.", type="negative")
                    return
                s.path, s.filename, s.info = dest, name, info
                s.zones, s.preview_size, s.drag_start, s.corner = [], None, None, None
                editor_box.clear()
                zone_list.refresh()
                results_box.clear()
                analyze_btn.enable()
                draw_btn.enable()
                secs = info.frame_count / info.fps if info.frame_count else 0
                file_info.set_text(f"{name}  ·  {info.width}×{info.height}" + (f"  ·  {secs:.0f} s" if secs else ""))
                sign("READY", name if len(name) <= 28 else name[:25] + "…")

            upload = ui.upload(
                label="Choose a video",
                auto_upload=True,
                max_file_size=MAX_UPLOAD_MB * 1024 * 1024,
                on_upload=on_upload,
                on_rejected=lambda: ui.notify(
                    f"That file is too large. The limit is {MAX_UPLOAD_MB} MB.", type="negative"
                ),
            ).props('accept="video/*" flat')
            file_info = ui.label("").classes("muted text-sm")

            with ui.expansion("Lanes and speed", caption="Optional", icon="route").classes("w-full").props("dense"):
                ui.label(
                    "Draw a box over each lane to count them separately, and enter the road width to estimate speed."
                ).classes("muted text-sm")

                async def load_preview():
                    if not s.path:
                        return
                    try:
                        jpg, w, h = await run.io_bound(detector.extract_preview, str(s.path))
                    except ValueError as err:
                        ui.notify(str(err), type="negative")
                        return
                    s.preview_size, s.zones, s.drag_start, s.corner = (w, h), [], None, None
                    editor_box.clear()
                    with editor_box:
                        ui.label(
                            "Drag a box over each lane. On a phone, tap one corner and then the opposite corner."
                        ).classes("muted text-sm")
                        editor = ui.interactive_image(
                            "data:image/jpeg;base64," + base64.b64encode(jpg).decode(),
                            on_mouse=on_mouse,
                            events=["mousedown", "mouseup"],
                            cross=True,
                        ).classes("w-full")
                        s_editor["img"] = editor
                    zone_list.refresh()

                draw_btn = ui.button("Draw lanes on this video", on_click=load_preview).props("outline no-caps")
                draw_btn.disable()
                editor_box = ui.column().classes("w-full gap-2")
                s_editor: dict = {}

                def redraw():
                    img = s_editor.get("img")
                    if img and s.preview_size:
                        img.set_content(zones_svg(s.zones, s.preview_size, s.corner))

                def add_zone(x0, y0, x1, y1):
                    w, h = s.preview_size
                    nx1, nx2 = sorted((min(max(x0 / w, 0), 1), min(max(x1 / w, 0), 1)))
                    ny1, ny2 = sorted((min(max(y0 / h, 0), 1), min(max(y1 / h, 0), 1)))
                    if nx2 - nx1 < 0.02 or ny2 - ny1 < 0.02:
                        return False
                    s.zones.append(Zone(f"Lane {len(s.zones) + 1}", nx1, ny1, nx2, ny2))
                    return True

                def on_mouse(e: events.MouseEventArguments):
                    if not s.preview_size:
                        return
                    if e.type == "mousedown":
                        s.drag_start = (e.image_x, e.image_y)
                    elif e.type == "mouseup" and s.drag_start:
                        x0, y0 = s.drag_start
                        s.drag_start = None
                        if add_zone(x0, y0, e.image_x, e.image_y):  # a real drag
                            s.corner = None
                            zone_list.refresh()
                        elif s.corner is None:  # a tap: remember the first corner
                            s.corner = (e.image_x, e.image_y)
                        else:  # a second tap: finish the box
                            cx, cy = s.corner
                            s.corner = None
                            if add_zone(cx, cy, e.image_x, e.image_y):
                                zone_list.refresh()
                        redraw()

                @ui.refreshable
                def zone_list():
                    redraw()
                    if not s.zones:
                        return
                    for i, z in enumerate(s.zones):
                        with ui.row().classes("w-full items-center gap-2 no-wrap"):
                            ui.element("div").classes("w-3 h-3 rounded-sm shrink-0").style(
                                f"background:{ZONE_COLORS[i % len(ZONE_COLORS)]}"
                            )
                            ui.input(
                                value=z.name,
                                on_change=lambda ev, z=z: (setattr(z, "name", ev.value or ""), redraw()),
                            ).props("dense outlined").classes("grow")
                            ui.button(
                                icon="delete",
                                on_click=lambda z=z: (s.zones.remove(z), zone_list.refresh()),
                            ).props("flat round dense").props('aria-label="Remove lane"')

                zone_list()

                road_width = ui.number(
                    "Road width across the frame (meters)", value=None, min=1, max=200, format="%.0f"
                ).props("outlined dense").classes("w-full")
                ui.label(
                    "Leave this empty to skip the speed estimate. The estimate is only as accurate as this number "
                    "and the camera angle."
                ).classes("muted text-sm")

            analyze_btn = ui.button("Analyze footage").props("unelevated no-caps color=primary text-color=black").classes(
                "w-full font-semibold"
            )
            analyze_btn.disable()

        # ----- results
        results_box = ui.column().classes("w-full gap-4")

        def bars(parent, title: str, data: dict[str, int], labels: Optional[dict[str, str]] = None):
            with parent:
                ui.label(title).classes("sign-font text-lg font-semibold")
                peak = max(data.values(), default=0) or 1
                for key, n in sorted(data.items(), key=lambda kv: -kv[1]):
                    with ui.row().classes("w-full justify-between items-baseline"):
                        ui.label((labels or {}).get(key, key)).classes("text-sm")
                        ui.label(str(n)).classes("sign-font font-semibold")
                    ui.linear_progress(value=n / peak, show_value=False, size="6px").classes("w-full")

        def render_results(r: dict):
            results_box.clear()
            level = r["congestion"]
            with results_box:
                with ui.card().classes("panel w-full gap-4 p-4"):
                    with ui.row().classes("w-full items-center gap-3"):
                        ui.label(f"{LEVEL_LABEL[level].capitalize()} traffic").classes(
                            "sign-font text-xl font-semibold px-3 py-1 rounded"
                        ).style(f"background:{LEVEL_COLOR[level]}22;color:{LEVEL_COLOR[level]}")
                        ui.label(
                            f"{r['avg_concurrent']} vehicles in view on average "
                            f"(moderate from {detector.CONGESTION_LIGHT:g}, heavy from {detector.CONGESTION_MODERATE:g})"
                        ).classes("muted text-sm")
                    with ui.element("div").classes("grid grid-cols-1 sm:grid-cols-3 gap-4 w-full"):
                        for caption, value, note in (
                            ("Vehicles counted", str(r["total"]), "each vehicle counted once"),
                            ("Average in view", f"{r['avg_concurrent']}", "per analyzed frame"),
                            (
                                "Estimated speed",
                                f"{r['avg_speed_kmh']} km/h" if r["avg_speed_kmh"] is not None else "–",
                                "median of tracked vehicles" if r["avg_speed_kmh"] is not None
                                else "enter the road width to estimate",
                            ),
                        ):
                            with ui.column().classes("gap-1"):
                                ui.label(value).classes("stat-num")
                                ui.label(caption).classes("text-sm font-medium")
                                ui.label(note).classes("muted text-xs")
                    ui.element("hr").classes("lane-rule")
                    with ui.element("div").classes("grid grid-cols-1 sm:grid-cols-2 gap-6 w-full"):
                        col = ui.column().classes("gap-2 w-full")
                        bars(col, "Vehicles by type", r["by_type"], TYPE_LABEL)
                        if not r["by_type"]:
                            with col:
                                ui.label("No vehicles were detected in this clip.").classes("muted text-sm")
                        if r["zone_counts"] is not None:
                            col2 = ui.column().classes("gap-2 w-full")
                            bars(col2, "Vehicles by lane", r["zone_counts"])
                            with col2:
                                ui.label("Vehicles outside every lane are not counted here.").classes("muted text-xs")
                    note = f"{r['frames_processed']} frames analyzed, covering {r['seconds_analyzed']} s."
                    if r["truncated"]:
                        note += f" The video is longer, so only the first {detector.MAX_VIDEO_SECONDS} s were used."
                    ui.label(note).classes("muted text-xs")

        async def show_done(r: dict):
            n, level = r["total"], r["congestion"]
            tail = f"{LEVEL_LABEL[level]} traffic"
            steps = min(n, 20)
            for i in range(1, steps + 1):  # short count-up in response to the click
                sign(f"{round(n * i / steps)} vehicles", tail, lamp_for(level))
                await asyncio.sleep(0.03)
            sign(f"{n} vehicle{'' if n == 1 else 's'}", tail, lamp_for(level))

        async def analyze():
            if not s.path or s.busy:
                return
            if ANALYSIS_LOCK.locked():
                ui.notify("Another video is being analyzed. Try again in a moment.", type="warning")
                return
            async with ANALYSIS_LOCK:
                s.busy, s.progress = True, 0.0
                analyze_btn.disable()
                results_box.clear()
                bar.value, bar.visible = 0, True
                zones = unique_named(s.zones)

                def on_progress(frac: float):  # runs in the worker thread
                    s.progress = frac

                try:
                    r = await run.io_bound(
                        detector.analyze_video, str(s.path), zones,
                        float(road_width.value) if road_width.value else None, on_progress,
                    )
                    await run.io_bound(database.save_run, r, s.filename)
                except Exception as err:
                    logger.exception("Analysis failed")
                    s.busy = False
                    sign("ERROR", "Analysis failed", "stop")
                    ui.notify(f"Analysis failed: {err}. The server log has the details.", type="negative", multi_line=True)
                else:
                    s.busy = False
                    render_results(r)
                    await show_done(r)
                    history.refresh()
                finally:
                    s.busy = False
                    bar.visible = False
                    analyze_btn.enable()

        analyze_btn.on_click(analyze)

        def tick():
            if s.busy:
                bar.value = s.progress
                sign("ANALYZING", "Loading the model…" if s.progress <= 0 else f"{int(s.progress * 100)}%",
                     "caution", blink=True)

        ui.timer(0.4, tick)

        # ----- history
        @ui.refreshable
        def history():
            rows = database.load_history()
            with ui.card().classes("panel w-full gap-3 p-4"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label("History").classes("sign-font text-xl font-semibold")
                    if rows:
                        ui.button("Clear", on_click=confirm_clear.open).props("flat no-caps dense")
                if not rows:
                    ui.label("No analyses yet. Each video you analyze is saved here so you can compare clips.").classes("muted text-sm")
                    return
                ui.echart(
                    {
                        "backgroundColor": "transparent",
                        "grid": {"left": 40, "right": 16, "top": 16, "bottom": 28},
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {"type": "time", "axisLabel": {"color": "#98A0AD"},
                                  "axisLine": {"lineStyle": {"color": "#2C313A"}}},
                        "yAxis": {"type": "value", "name": "vehicles", "nameTextStyle": {"color": "#98A0AD"},
                                  "axisLabel": {"color": "#98A0AD"},
                                  "splitLine": {"lineStyle": {"color": "#2C313A"}}},
                        "series": [{
                            "type": "line", "symbolSize": 11,
                            "lineStyle": {"color": "#98A0AD", "width": 1.5},
                            "data": [
                                {"value": [int(r["timestamp"] * 1000), r["total"]],
                                 "itemStyle": {"color": LEVEL_COLOR[r["congestion"]]}}
                                for r in rows
                            ],
                        }],
                    }
                ).classes("w-full h-56")
                ui.label("Dot colour shows congestion: green light, amber moderate, red heavy.").classes("muted text-xs")
                ui.table(
                    columns=[
                        {"name": "when", "label": "When (UTC)", "field": "when", "align": "left"},
                        {"name": "file", "label": "File", "field": "file", "align": "left"},
                        {"name": "n", "label": "Vehicles", "field": "n"},
                        {"name": "level", "label": "Traffic", "field": "level", "align": "left"},
                        {"name": "speed", "label": "Speed", "field": "speed"},
                    ],
                    rows=[
                        {
                            "id": r["id"],
                            "when": datetime.fromtimestamp(r["timestamp"], timezone.utc).strftime("%Y-%m-%d %H:%M"),
                            "file": r["filename"],
                            "n": r["total"],
                            "level": r["congestion"].capitalize(),
                            "speed": f"{r['avg_speed_kmh']} km/h" if r["avg_speed_kmh"] is not None else "–",
                        }
                        for r in reversed(rows[-8:])
                    ],
                    row_key="id",
                ).props("dense flat").classes("w-full bg-transparent")

        with ui.dialog() as confirm_clear, ui.card().classes("panel"):
            ui.label("Delete all saved results?").classes("sign-font text-lg")
            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=confirm_clear.close).props("flat no-caps")

                def do_clear():
                    database.clear_history()
                    confirm_clear.close()
                    history.refresh()

                ui.button("Delete", on_click=do_clear).props("unelevated no-caps color=negative")

        history()


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8080")),
        title="Traffic monitor",
        favicon="🚦",
        reload=False,
        show=False,
    )
