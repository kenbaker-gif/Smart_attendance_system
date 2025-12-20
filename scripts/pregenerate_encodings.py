#!/usr/bin/env python3
import os
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_PATH = ROOT / "streamlit" / "app.py"

if not APP_PATH.exists():
    print("❌ Could not find streamlit/app.py; are you running from the project root?")
    sys.exit(2)

spec = importlib.util.spec_from_file_location("project_streamlit_app", str(APP_PATH))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

# Optionally, allow overriding detection size via env var
det_size_env = os.getenv("PREGEN_DET_SIZE")
if det_size_env:
    try:
        parts = [int(x) for x in det_size_env.split("x")]
        det_size = tuple(parts)
    except Exception:
        print(f"❌ Invalid PREGEN_DET_SIZE={det_size_env}, expected format WIDTHxHEIGHT, e.g. 320x320")
        sys.exit(2)
else:
    det_size = (320, 320)

print(f"🔧 Starting pregenerate encodings (det_size={det_size})")

# If SUPABASE environment variables are available, the app's generate_encodings
# will attempt to download images from Supabase if configured. Otherwise it will
# use images present in streamlit/data/raw_faces.

try:
    app.get_insightface(det_size=det_size)
except Exception as e:
    print(f"❌ InsightFace initialization failed: {e}")
    sys.exit(3)

ok = app.generate_encodings(app.RAW_FACES_DIR, app.ENCODINGS_PATH)
if not ok:
    print("❌ Failed to generate encodings. Check that `streamlit/data/raw_faces` contains student folders or configure Supabase env vars.")
    sys.exit(4)

if app.ENCODINGS_PATH.exists():
    size = app.ENCODINGS_PATH.stat().st_size
    print(f"✅ Encodings generated: {app.ENCODINGS_PATH} ({size} bytes)")
    sys.exit(0)
else:
    print("❌ Encodings file not found after generation.")
    sys.exit(5)
