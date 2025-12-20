import os
import sys
import gc
import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import cv2
from dotenv import load_dotenv

# --- Schema Integration ---
try:
    from schemas import AttendanceRecordIn
except ImportError:
    from pydantic import BaseModel, Field
    class AttendanceRecordIn(BaseModel):
        student_id: str
        confidence: float
        detection_method: str
        verified: str

# -----------------------------
# Configuration & Paths
# -----------------------------
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
TEMP_CAPTURE_PATH = DATA_DIR / "tmp_capture.jpg"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"

# Ensure directories exist
for d in [DATA_DIR, RAW_FACES_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Model & Thresholds
INSIGHTFACE_MODEL_NAME = "buffalo_s"
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))
ENCODINGS_REMOTE_PATH = os.getenv("ENCODINGS_REMOTE_PATH", "encodings/encodings_insightface.pkl")

# -----------------------------
# Logging Setup
# -----------------------------
logger = logging.getLogger("attendance_system")
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

# -----------------------------
# Supabase & State
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
        
        # Helper for recursive image sync
        try:
            from app.utils.supabase_utils import download_all_supabase_images
        except ImportError:
            download_all_supabase_images = None # Action fallback
    except Exception as e:
        logger.error(f"Supabase init warning: {e}")

if 'log_cache' not in st.session_state:
    st.session_state.log_cache = {}
LOG_COOLDOWN_SECONDS = 60

# -----------------------------
# InsightFace Engine
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ImportError:
    st.error("❌ insightface not found.")
    st.stop()

@st.cache_resource(show_spinner=False)
def get_insightface(det_size=(640, 640)):
    _app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=["CPUExecutionProvider"])
    _app.prepare(ctx_id=-1, det_size=det_size)
    return _app

# -----------------------------
# Core Utilities
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
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        return np.array(face.embedding, dtype=np.float32)
    except Exception as e:
        logger.error(f"Encoding generation error: {e}")
        return None

def add_attendance_record(student_id: str, confidence: float, status: str):
    """Logs to Supabase with Deduplication and Pydantic Validation."""
    current_time = datetime.now()
    if status == 'success':
        last_logged = st.session_state.log_cache.get(student_id)
        if last_logged and (current_time - last_logged) < timedelta(seconds=LOG_COOLDOWN_SECONDS):
            return

    logger.info(f"Log: {student_id} | Status: {status} | Conf: {confidence:.2f}")

    if USE_SUPABASE and supabase:
        try:
            record = AttendanceRecordIn(
                student_id=student_id,
                confidence=float(confidence),
                detection_method="insightface_buffalo_s",
                verified=status
            )
            data = record.dict() if hasattr(record, 'dict') else record.model_dump()
            supabase.table('attendance_records').insert(data).execute()
            if status == 'success':
                st.session_state.log_cache[student_id] = current_time
            st.toast(f"Synced to DB: {student_id}", icon="✅")
        except Exception as e:
            logger.error(f"DB Log Fail: {e}")

# -----------------------------
# Process Pipeline
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    if USE_SUPABASE and download_all_supabase_images:
        download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=True)
    
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    if not student_dirs: return False

    encodings, ids = [], []
    for s_dir in student_dirs:
        student_id = s_dir.name
        img_files = [p for p in s_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        for img_p in img_files:
            emb = _generate_face_encoding_from_image(img_p)
            if emb is not None:
                encodings.append(emb)
                ids.append(student_id)

    if not encodings: return False
    try:
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(output_path, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        return True
    except Exception:
        return False

@st.cache_resource
def load_encodings():
    if not ENCODINGS_PATH.exists() and USE_SUPABASE and supabase:
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(ENCODINGS_REMOTE_PATH)
            data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
            if data_bytes:
                with open(ENCODINGS_PATH, "wb") as f:
                    f.write(data_bytes)
        except Exception: pass

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
        st.warning("No student data loaded. Use Admin Panel to generate.")

    tab1, tab2 = st.tabs(["Verification", "Admin Panel"])

    with tab1:
        sid = st.text_input("Enter Student ID")
        img_file = st.camera_input("Take Photo")
        
        if sid and img_file and st.button("Verify Identity"):
            img = Image.open(img_file).convert("RGB")
            img.save(TEMP_CAPTURE_PATH)
            captured_emb = _generate_face_encoding_from_image(TEMP_CAPTURE_PATH)
            
            if captured_emb is not None:
                captured_emb /= (np.linalg.norm(captured_emb) + 1e-10)
                dists = 1.0 - np.dot(known_encs, captured_emb)
                idx = np.argmin(dists)
                conf = max(0.0, float(1.0 - dists[idx]))
                
                if dists[idx] < DEFAULT_THRESHOLD and known_ids[idx] == sid:
                    st.success(f"WELCOME: {sid} (Match: {conf*100:.1f}%)")
                    st.balloons()
                    add_attendance_record(sid, conf, "success")
                else:
                    st.error(f"Failed. Matched with {known_ids[idx]}? Dist: {dists[idx]:.3f}")
                    add_attendance_record(sid, conf, "failed")

    with tab2:
        st.subheader("📊 System Logs")
        if LOG_FILE.exists():
            with open(LOG_FILE, "r") as f:
                log_data = f.read()
            c1, c2 = st.columns(2)
            with c1:
                st.download_button("📥 Download Log", log_data, f"attendance_{datetime.now().date()}.txt")
            with c2:
                if st.button("🗑️ Clear Logs"):
                    with open(LOG_FILE, "w") as f: f.write("")
                    st.rerun()
        
        st.divider()
        if st.button("🔄 Sync & Regenerate Encodings"):
            with st.spinner("Processing..."):
                if generate_encodings():
                    st.success("Updated!")
                    st.rerun()

if __name__ == "__main__":
    if "--generate" in sys.argv:
        sys.exit(0 if generate_encodings() else 4)
    else:
        main()