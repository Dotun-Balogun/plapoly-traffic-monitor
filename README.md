# Traffic monitor

Upload a video of a road and get:

- a **vehicle count** (cars, trucks, buses, motorcycles), where each vehicle is counted once
- a **congestion level**: light, moderate or heavy
- a **count per lane**, using boxes you draw on a frame of your own video
- an **estimated speed** in km/h
- a **history chart** so you can compare clips

Everything is Python. There is no separate website or API to host: one program serves
the page and does the detection, so there is **one thing to run and one thing to deploy**.

Built with [NiceGUI](https://nicegui.io) (the web page, written in Python) and
[YOLOv8](https://docs.ultralytics.com) (vehicle detection and tracking).

---

## What is in this folder

| File | What it does |
|---|---|
| `main.py` | The web page and the app. **Start this file.** |
| `detector.py` | Finds and tracks vehicles, then works out counts, congestion, lanes and speed. |
| `database.py` | Saves each result to a small SQLite file (`data/history.db`) for the history chart. |
| `requirements.txt` | The Python packages to install (PyTorch is installed separately, see Part 1). |
| `.python-version` | Tells tools which Python to use (3.11). |
| `render.yaml` | Optional: lets Render set itself up from this file. |
| `tests/`, `conftest.py`, `pytest.ini` | Automated tests (optional to run). |

---

# Part 1: Run it on your own computer

Do this **before** putting anything online. If it works here, any problem on Render is
about Render, not the code.

### Step 1. Check what you have

Open a terminal and run:

```bash
python3 --version     # needs 3.10, 3.11 or 3.12
df -h .               # look at "Avail": you want at least 3 GB free
```

If space is tight, free some first. Installing PyTorch is what fills disks, and the
steps below use the small CPU-only version to avoid that.

### Step 2. Open a terminal inside this folder

```bash
cd path/to/traffic-monitor
ls        # you should see main.py, detector.py, requirements.txt ...
```

### Step 3. Create a virtual environment and switch it on

A virtual environment is a private folder for this project's packages. It keeps them
off your system Python.

Linux / macOS:
```bash
python3 -m venv venv --upgrade-deps
source venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv venv --upgrade-deps
venv\Scripts\activate
```

**Check it worked.** Your prompt should now start with `(venv)`. On Linux/macOS also run:

```bash
which python
```

The path must end in `/venv/bin/python`. If it points to `/usr/bin/...` or
`/usr/lib/...`, the environment is **not** active and everything you install will go
into your system Python. Stop and repeat the `source venv/bin/activate` line.

You must repeat the activate line **every time you open a new terminal** for this project.

### Step 4. Install PyTorch (the CPU version) first

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

- Install `torch` and `torchvision` **in the same command**. Installing them separately
  can give you mismatched versions and the error
  `operator torchvision::nms does not exist`.
- This is the CPU-only build (roughly 200 MB). A plain `pip install torch` downloads a
  GPU build with several GB of CUDA libraries, which you do not need and which can fill
  your disk (`No space left on device`).

### Step 5. Install everything else

```bash
pip install -r requirements.txt
```

### Step 6. Fix OpenCV (important)

YOLO's package (`ultralytics`) always installs the *full* `opencv-python`, which needs
desktop graphics libraries. Servers don't have them, and the result is the error
`ImportError: libxcb.so.1: cannot open shared object file`. The fix is to remove it and
keep only the "headless" build, which does everything this app needs without a display:

```bash
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-python-headless
```

Now check everything is in place:

```bash
pip list | grep -i opencv                                  # only opencv-python-headless
python -c "import cv2, torch, ultralytics; print('all good')"
```

(On Windows replace `grep -i opencv` with `findstr /i opencv`.)

### Step 7. Start the app

```bash
python main.py
```

You should see `NiceGUI ready to go on http://localhost:8080`. Open
**http://localhost:8080** in your browser.

The first start downloads the YOLO weights (about 6 MB, once) and loads the model in
the background. Lines like `NNPACK ... Unsupported hardware` are harmless warnings.

To stop the app, press `Ctrl + C` in the terminal.

### Step 8. Test it properly

Use a short video (10 to 30 seconds) from a fixed camera looking along a road, ideally
from above, such as a bridge or a high window. Free clips from sites like Pexels or
Pixabay work. Go through this list:

| # | Do this | You should see |
|---|---|---|
| 1 | Open `http://localhost:8080/healthz` | `{"status":"ok"}` |
| 2 | Open `http://localhost:8080` | The sign says **NO FOOTAGE**, all three lamps dark, History says "No analyses yet". |
| 3 | Choose your video | The sign says **READY** and the file name; a line shows its size and length. |
| 4 | Click **Analyze footage** | Sign says **ANALYZING**, the amber lamp blinks, a progress bar fills. The first run says "Loading the model…" for a while. |
| 5 | Wait for it to finish | The sign shows **N vehicles** and **LIGHT / MODERATE / HEAVY traffic**, with one lamp lit (green, amber or red). A results card appears, and a dot is added to the History chart. |
| 6 | Open **Lanes and speed**, click **Draw lanes on this video**, drag a box over a lane, then **Analyze** again | The box appears named "Lane 1" (rename it if you like); results now include **Vehicles by lane**. |
| 7 | Type a road width such as `12`, then **Analyze** | **Estimated speed** shows a km/h value. |
| 8 | Refresh the page | History is still there (it is saved in `data/history.db`). |

**Check the count yourself.** Count the vehicles in a short clip by eye and compare. The
small YOLO model usually *under*-counts (it misses small, distant or hidden vehicles).
That is normal. See "Getting better results" below.

### Step 9 (optional). Run the automated tests

```bash
pip install pytest pytest-asyncio
pytest
```

You should see `20 passed`. These tests check the page and the counting maths using a
*fake* detector, so they run fast without YOLO. They prove the app logic works; they do
not measure YOLO's accuracy. Only your own clips (Step 8) can do that.

---

# Part 2: Put the code on GitHub

Render deploys from GitHub. In this folder:

```bash
git init
git add .
git commit -m "Traffic monitor"
git branch -M main
```

Create an empty repository on github.com (no README, no .gitignore), then:

```bash
git remote add origin https://github.com/YOUR-NAME/traffic-monitor.git
git push -u origin main
```

Open the repo in your browser and confirm `main.py` is at the **top level** (not inside
another folder). The `.gitignore` already keeps `venv/`, `data/` and the model file out.

---

# Part 3: Deploy on Render

## Read this first: memory

PyTorch plus video needs RAM. Render's instance sizes:

| Plan | Price | RAM | Notes |
|---|---|---|---|
| Free | $0 | **512 MB** | Sleeps after 15 min idle |
| Starter | $7/mo | **512 MB** | Same RAM as Free; only removes sleeping |
| Standard | $25/mo | **2 GB** | First plan with real headroom |

Starter does **not** give you more memory than Free. With 512 MB I expect the page to
load fine but the analysis to be killed part-way (Render logs say *out of memory* or
`exited with 137`). **Standard is the realistic choice for this app.** If you are not
ready to pay, run it locally (Part 1), which is always free and works the same.

To fit in less memory, set these environment variables (see Settings below):
`YOLO_IMGSZ=480`, `MAX_VIDEO_SECONDS=30`, and upload short clips. This helps but does not
guarantee success on 512 MB.

Prices change, so confirm them at render.com/pricing.

## Create the service

1. Go to **render.com** and sign in with GitHub.
2. Click **New** then **Web Service**, and pick your `traffic-monitor` repo.
3. Fill in the form:

| Field | Value |
|---|---|
| Name | anything, e.g. `traffic-monitor` |
| Branch | `main` |
| Root Directory | leave **empty** (the repo root is the app) |
| Runtime | `Python 3` |
| Build Command | the one-line command below |
| Start Command | `python main.py` |
| Instance Type | Standard (recommended), or Free to experiment |

**Build Command** (one line; copy it exactly):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && pip uninstall -y opencv-python && pip install --force-reinstall --no-deps opencv-python-headless
```

4. Open **Advanced** and add:
   - Environment variable `PYTHON_VERSION` = `3.11.9`
   - **Health Check Path** = `/healthz`
5. Click **Create Web Service**.

(Alternatively, **New** then **Blueprint** reads `render.yaml` and fills this in for
you. It defaults to the Free plan; change `plan:` there first if you want Standard.)

## Watch the first deploy

The build takes several minutes (PyTorch is large). In the **Logs** tab, a good deploy
ends with `Successfully installed ...` and then `NiceGUI ready to go on ...`. Render
then shows your public address, like `https://traffic-monitor-xxxx.onrender.com`.

Open it and repeat the Step 8 checklist from Part 1 on the live site.

## Good to know

- **No Vercel and no separate API service.** This one service is the whole app. If you
  created those earlier, you can delete them.
- **Free instances sleep** after 15 minutes idle; the next visit takes 30 to 60 seconds.
- **History resets on every redeploy**, because the server's disk is wiped. To keep it,
  attach a Render *persistent disk* (paid plans), mount it at `/var/data`, and set
  `DATA_DIR=/var/data`.
- **Updating:** change code, `git push`, and Render rebuilds automatically.

---

# Settings (environment variables)

All optional. Set them in your terminal (`DATA_DIR=/tmp python main.py`) or in Render's
Environment tab.

| Variable | Default | What it does |
|---|---|---|
| `PORT` | `8080` | Port to listen on. Render sets this itself. |
| `MAX_UPLOAD_MB` | `100` | Largest video you can upload. |
| `MAX_VIDEO_SECONDS` | `120` | Only the first N seconds are analyzed. Lower = less memory and time. |
| `VID_STRIDE` | `3` | Look at every Nth frame. `1` = every frame (slow); higher = faster, less precise. |
| `CONGESTION_LIGHT` | `3` | Under this many vehicles in view on average = light. |
| `CONGESTION_MODERATE` | `8` | Under this = moderate; at or above = heavy. |
| `YOLO_MODEL` | `yolov8n.pt` | Try `yolov8s.pt` for better accuracy (slower, more memory). |
| `YOLO_IMGSZ` | `640` | Detection image size. `480` uses less memory but misses small vehicles. |
| `DATA_DIR` | `./data` | Where the history database lives. |
| `WARM_UP` | `1` | Set `0` to skip loading the model at startup. |

---

# Getting better results

- **Camera:** a fixed camera, high up, looking along the road works best. Low angles
  make vehicles hide each other, so counts drop.
- **Congestion thresholds are a starting guess.** "Heavy" means 8 or more vehicles
  visible on average. A wide highway view and a narrow street need different numbers, so
  adjust `CONGESTION_LIGHT` and `CONGESTION_MODERATE` after you see a few real clips.
- **Lanes:** draw boxes over the road itself. A vehicle is assigned by where its wheels
  meet the road (the bottom-centre of its box). If boxes overlap, the first one drawn
  wins. Vehicles outside every box are not counted in the lane totals.
- **Speed:** it is only as good as the road width you type in and the camera angle. It
  assumes the width you entered spans the whole frame, and perspective makes far vehicles
  look slower. Treat it as a rough estimate (the app reports the median of all tracked
  vehicles), not a measurement.

---

# Troubleshooting

| What you see | Why | Fix |
|---|---|---|
| `No space left on device` while installing | Disk full, usually from the huge GPU PyTorch | `pip cache purge`, free space with `df -h .`, and install the CPU PyTorch as in Part 1 Step 4. |
| pip prints paths like `/usr/lib/python3.12/site-packages` | The virtual environment is not active | Run `source venv/bin/activate` and check `which python` (Step 3). |
| `venv/bin/pip: No such file or directory` | The venv was only half created (often a full disk) | `rm -rf venv`, free space, then `python3 -m venv venv --upgrade-deps`. |
| `operator torchvision::nms does not exist` | `torch` and `torchvision` versions do not match | `pip uninstall -y torch torchvision`, then reinstall both **together** (Step 4). |
| `ImportError: libxcb.so.1` or `libGL.so.1` | The full OpenCV got installed alongside the headless one | Step 6 (uninstall `opencv-python`, reinstall `opencv-python-headless`). |
| `NNPACK ... Unsupported hardware` | A harmless PyTorch warning | Ignore it. |
| `Address already in use` | Another copy is still running | Stop it, or use another port: `PORT=8081 python main.py`. |
| `Ctrl + C` does not stop it during an analysis | Heavy work delays the signal | In a second terminal: `pkill -f main.py` (Windows: close the terminal). |
| "That file could not be opened as a video" | Unreadable or unsupported format | Try an MP4 (H.264). |
| "That file is too large" | Over `MAX_UPLOAD_MB` | Use a shorter clip or raise `MAX_UPLOAD_MB`. |
| Analysis runs, then the page reloads or goes blank **on Render** | The process ran out of memory and was killed | Check Render's Logs for `out of memory` / `137`. Use Standard (2 GB), shorter clips, `YOLO_IMGSZ=480`, `MAX_VIDEO_SECONDS=30`. |
| Counts look too low | The small model misses small or hidden vehicles | Better camera angle, `YOLO_MODEL=yolov8s.pt`, or `VID_STRIDE=2`. |
| First visit to the Render site is very slow | Free instance waking up, plus model loading | Wait up to a minute; open `/healthz` first to wake it. |
| History vanished after a deploy | Render's disk is temporary | Attach a persistent disk and set `DATA_DIR` (see Part 3). |
| Fonts look plain | The page loads Barlow from Google Fonts and you are offline | Cosmetic only; it falls back to system fonts. |

---

# Honest limits

- Counts, congestion and speed are **estimates** from a small, fast model. They are good
  for comparing clips and spotting patterns, not for legal or safety-critical use.
- Only the first `MAX_VIDEO_SECONDS` of a video are analyzed, and only one video is
  analyzed at a time per server.
- Detection accuracy depends heavily on your footage. Test with your own clips.
