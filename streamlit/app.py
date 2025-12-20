import os
import sys
import gc
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import List, Optional
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
import cv2
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Project root and paths
# -----------------------------
# Use resolve() to ensure absolute paths regardless of where the script is called from
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
TEMP_CAPTURE_PATH = DATA_DIR / "tmp_capture.jpg"

# Ensure directories exist immediately
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

# Remote artifact configuration
ENCODINGS_REMOTE_PATH = os.getenv("ENCODINGS_REMOTE_PATH", "encodings/encodings_insightface.pkl")
ENCODINGS_REMOTE_TYPE = os.getenv("ENCODINGS_REMOTE_TYPE", "supabase")
INSIGHTFACE_MODEL_NAME = "buffalo_s"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# -----------------------------
# Supabase Configuration
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "")

supabase = None
download_all_supabase_images = None

if USE_SUPABASE:
    try:
        from supabase import create_client
        if SUPABASE_URL and SUPABASE_KEY:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Import the recursive downloader we verified (handles StudentID/1.jpg)
        try:
            from app.utils.supabase_utils import download_all_supabase_images
        except ImportError:
            # Fallback for different folder structures
            import importlib.util
            utils_path = PROJECT_ROOT.parent / "app" / "utils" / "supabase_utils.py"
            if utils_path.exists():
                spec = importlib.util.spec_from_file_location("supabase_utils", str(utils_path))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                download_all_supabase_images = getattr(mod, "download_all_supabase_images", None)
    except Exception as e:
        print(f"⚠ Supabase initialization warning: {e}")

# -----------------------------
# InsightFace setup (lazy load)
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ImportError:
    st.error("❌ insightface not found. Install via: pip install insightface onnxruntime")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_insightface(det_size=(640, 640)):
    """Singleton for the FaceAnalysis model."""
    _app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=det_size)
    return _app

# -----------------------------
# Helpers
# -----------------------------
def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0: return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def _generate_face_encoding_from_image(path: Path) -> Optional[np.ndarray]:
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None: return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = get_insightface().get(img_rgb)
        if not faces: return None
        # Get largest face by bounding box area
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return np.array(face.embedding, dtype=np.float32)
    except Exception:
        return None

# -----------------------------
# Core Logic: Generate Encodings
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    """The function triggered by GitHub Actions or the Admin Panel."""
    
    # 1. Clear and Download fresh data if Supabase is enabled
    if USE_SUPABASE and download_all_supabase_images:
        print(f"📦 Syncing from Supabase: {SUPABASE_BUCKET} -> {images_dir}")
        ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=True)
        if not ok:
            print("❌ Download failed. Check Supabase connection.")
    
    # 2. Check directory structure (The Fix for Exit Code 4)
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    print(f"🔍 Found {len(student_dirs)} student folders in {images_dir}")

    if not student_dirs:
        print(f"❌ ERROR: No folders found in {images_dir}. Check if Supabase contains folders.")
        return False  # This triggers Exit Code 4 in a sys.exit wrapper

    # 3. Process Images
    encodings, ids = [], []
    detector = get_insightface(det_size=(320, 320)) # Smaller det_size for faster batch processing

    for s_dir in student_dirs:
        student_id = s_dir.name
        img_files = [p for p in s_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        for img_p in img_files:
            emb = _generate_face_encoding_from_image(img_p)
            if emb is not None:
                encodings.append(emb)
                ids.append(student_id)
                print(f"✅ Encoded: {student_id}/{img_p.name}")

    # 4. Save to Pickle
    if not encodings:
        print("❌ No faces detected in any of the downloaded images.")
        return False

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(output_path, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        print(f"✨ Successfully created {output_path} with {len(ids)} faces.")
        return True
    except Exception as e:
        print(f"❌ Failed to save pickle: {e}")
        return False

@st.cache_resource
def load_encodings():
    """Tries local load, then tries Supabase download if local is missing."""
    if not ENCODINGS_PATH.exists() and USE_SUPABASE and supabase:
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(ENCODINGS_REMOTE_PATH)
            # Handle both raw bytes and dictionary responses
            data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
            if data_bytes:
                with open(ENCODINGS_PATH, "wb") as f:
                    f.write(data_bytes)
        except Exception:
            pass

    if ENCODINGS_PATH.exists():
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        encs = np.array(data.get("encodings", []), dtype=np.float32)
        ids = [str(i) for i in data.get("ids", [])]
        return normalize_encodings(encs), ids
    return np.array([]), []

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸")
    st.title("📸 Attendance System")

    known_encs, known_ids = load_encodings()
    
    if known_encs.size == 0:
        st.warning("No student data loaded. Please run 'Generate' in the Admin Panel.")

    tab1, tab2 = st.tabs(["Verification", "Admin Panel"])

    with tab1:
        sid = st.text_input("Enter Student ID")
        img_file = st.camera_input("Take Photo")
        
        if sid and img_file and st.button("Verify"):
            img = Image.open(img_file).convert("RGB")
            img.save(TEMP_CAPTURE_PATH)
            captured_emb = _generate_face_encoding_from_image(TEMP_CAPTURE_PATH)
            
            if captured_emb is not None:
                captured_emb /= (np.linalg.norm(captured_emb) + 1e-10)
                dists = 1.0 - np.dot(known_encs, captured_emb)
                idx = np.argmin(dists)
                
                if dists[idx] < DEFAULT_THRESHOLD and known_ids[idx] == sid:
                    st.success(f"WELCOME: {sid} (Match: {100-dists[idx]*100:.1f}%)")
                    st.balloons()
                else:
                    st.error(f"Failed. Matched with {known_ids[idx]}? Dist: {dists[idx]:.3f}")

    with tab2:
        st.subheader("Data Management")
        st.write(f"Local Encodings: {'✅ Found' if ENCODINGS_PATH.exists() else '❌ Missing'}")
        
        if st.button("🔄 Sync & Regenerate Encodings"):
            with st.spinner("Processing... this may take a minute."):
                if generate_encodings():
                    st.success("Encodings updated!")
                    st.rerun()
                else:
                    st.error("Generation failed. Check logs.")

if __name__ == "__main__":
    # If called via GitHub Action using 'python app.py --generate'
    if "--generate" in sys.argv:
        success = generate_encodings()
        sys.exit(0 if success else 4)
    else:
        main()