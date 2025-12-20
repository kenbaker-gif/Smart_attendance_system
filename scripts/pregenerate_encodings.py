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

# Diagnostic: print presence of Supabase env vars (not values) so CI logs show whether a download was attempted.
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
print(f"📡 USE_SUPABASE={USE_SUPABASE}, SUPABASE_URL_set={bool(SUPABASE_URL)}, SUPABASE_KEY_set={bool(SUPABASE_KEY)}, SUPABASE_BUCKET_set={bool(SUPABASE_BUCKET)}")

# If Supabase is enabled, attempt to list top objects for debugging; do not fail on errors.
if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY and SUPABASE_BUCKET:
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        storage = client.storage.from_(SUPABASE_BUCKET)
        all_files_raw = storage.list("", options={"limit": 1000, "deep": True})
        # Try to normalize response
        sample_names = []
        if isinstance(all_files_raw, dict):
            data = all_files_raw.get("data") or all_files_raw.get("files") or all_files_raw.get("list")
            if isinstance(data, list):
                for e in data[:20]:
                    name = e.get("name") or e.get("id") or str(e)
                    sample_names.append(name)
        elif isinstance(all_files_raw, list):
            for e in all_files_raw[:20]:
                name = e.get("name") or e.get("id") or str(e)
                sample_names.append(name)
        print(f"🔍 Supabase bucket sample (up to 20 objects): {sample_names}")
    except Exception as e:
        print(f"⚠ Could not list Supabase bucket for debug: {e}")

# If InsightFace fails to initialize, we can't proceed
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
