import os

os.environ["WARM_UP"] = "0"  # tests use a fake detector; don't try to load YOLO

pytest_plugins = ["nicegui.testing.user_plugin"]
