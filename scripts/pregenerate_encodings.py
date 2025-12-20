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

# ... (after your diagnostic print statements)

# 1. NEW STEP: Download the images if Supabase is enabled
if USE_SUPABASE:
    print("📥 Attempting to sync faces from Supabase...")
    try:
        # Check app.py for the specific download function name
        # It's likely called sync_faces(), download_faces(), or similar
        app.download_faces_from_supabase() 
        print("✅ Sync complete.")
    except AttributeError:
        print("❌ Error: Could not find a download function in app.py. Check function name.")
    except Exception as e:
        print(f"❌ Download failed: {e}")

# 2. Proceed with InsightFace initialization
try:
    app.get_insightface(det_size=det_size)
except Exception as e:
    print(f"❌ InsightFace initialization failed: {e}")
    sys.exit(3)

# 3. Generate encodings
ok = app.generate_encodings(app.RAW_FACES_DIR, app.ENCODINGS_PATH)
