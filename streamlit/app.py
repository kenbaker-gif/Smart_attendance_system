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
from pydantic import BaseModel

# --- Permanent Schema Definition ---
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
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"

for d in [DATA_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

supabase = None
if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Supabase init error: {e}")

if 'log_cache' not in st.session_state:
    st.session_state.log_cache = {}
LOG_COOLDOWN_SECONDS = 60

# -----------------------------
# InsightFace Engine
# -----------------------------
@st.cache_resource(show_spinner="Loading AI Engine...")
def get_insightface(det_size=(640, 640)):
    from insightface.app import FaceAnalysis
    # CPUExecutionProvider is standard for Railway
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

def add_attendance_record(student_id: str, confidence: float, status: str):
    current_time = datetime.now()
    if status == 'success':
        last_logged = st.session_state.log_cache.get(student_id)
        if last_logged and (current_time - last_logged) < timedelta(seconds=LOG_COOLDOWN_SECONDS):
            return

    if USE_SUPABASE and supabase:
        try:
            record = AttendanceRecordIn(
                student_id=student_id,
                confidence=float(confidence),
                detection_method="insightface_buffalo_s",
                verified=status
            )
            supabase.table('attendance_records').insert(record.model_dump()).execute()
            if status == 'success':
                st.session_state.log_cache[student_id] = current_time
            st.toast(f"Synced: {student_id}", icon="✅")
        except Exception as e:
            logger.error(f"DB Log Fail: {e}")

# -----------------------------
# Process Pipeline (Optimized for Speed)
# -----------------------------
def generate_encodings() -> bool:
    if not USE_SUPABASE or not supabase:
        st.error("Supabase not configured.")
        return False
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📂 Accessing Supabase Storage...")
        folders = supabase.storage.from_(SUPABASE_BUCKET).list()
        
        # Filter student folders
        valid_folders = [f for f in folders if not f['name'].startswith('.') and f['name'] != "encodings"]
        total_folders = len(valid_folders)
        
        if total_folders == 0:
            status_text.error("No student folders found in bucket.")
            return False

        encodings, ids = [], []
        engine = get_insightface()

        for i, folder in enumerate(valid_folders):
            student_id = folder['name']
            status_text.text(f"⚙️ Processing Student: {student_id} ({i+1}/{total_folders})")
            
            files = supabase.storage.from_(SUPABASE_BUCKET).list(student_id)
            for f_info in files:
                file_name = f_info['name']
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    remote_path = f"{student_id}/{file_name}"
                    
                    # Download to memory (skips slow disk I/O)
                    data = supabase.storage.from_(SUPABASE_BUCKET).download(remote_path)
                    nparr = np.frombuffer(data, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img_bgr is None: continue
                    
                    # Face detection & encoding
                    faces = engine.get(img_bgr)
                    if faces:
                        # Largest face only
                        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        encodings.append(face.embedding)
                        ids.append(student_id)
            
            progress_bar.progress((i + 1) / total_folders)

        if not encodings:
            status_text.error("No faces detected in images.")
            return False
        
        # Save to local cache & Upload
        status_text.text("☁️ Syncing Encodings to Cloud...")
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        
        with open(ENCODINGS_PATH, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=ENCODINGS_REMOTE_PATH.lstrip('/'),
                file=f,
                file_options={"upsert": "true"}
            )
        
        status_text.success("✅ Database Regenerated Successfully!")
        progress_bar.empty()
        return True
        
    except Exception as e:
        status_text.error(f"Critical Sync Error: {e}")
        return False
    finally:
        gc.collect()

@st.cache_resource
def load_encodings():
    """Tries local cache, then downloads from Supabase if on Railway."""
    if not ENCODINGS_PATH.exists() and USE_SUPABASE and supabase:
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(ENCODINGS_REMOTE_PATH.lstrip('/'))
            data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
            if data_bytes:
                with open(ENCODINGS_PATH, "wb") as f: f.write(data_bytes)
        except Exception: pass

    if ENCODINGS_PATH.exists():
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            return normalize_encodings(np.array(data["encodings"])), [str(i) for i in data["ids"]]
        except Exception: pass
    return np.array([]), []

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸", layout="centered")
    st.title("📸 AI Attendance System")

    known_encs, known_ids = load_encodings()
    tab1, tab2 = st.tabs(["Verification", "Admin Panel"])

    with tab1:
        if known_encs.size == 0:
            st.warning("Biometric database empty. Use Admin Panel to Sync.")
            
        sid = st.text_input("Enter Student ID (e.g., 2400102415)")
        img_file = st.camera_input("Capture Face")
        
        if sid and img_file:
            if st.button("Verify Identity"):
                img = Image.open(img_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                with st.spinner("Running AI Matching..."):
                    faces = get_insightface().get(img_bgr)
                
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
                    
                    if known_encs.size > 0:
                        dists = 1.0 - np.dot(known_encs, captured_emb)
                        idx = np.argmin(dists)
                        conf = float(1.0 - dists[idx])
                        
                        # Match Logic
                        if dists[idx] < DEFAULT_THRESHOLD and str(known_ids[idx]) == str(sid).strip():
                            st.success(f"Verified: {sid} ({conf:.2f})")
                            st.balloons()
                            add_attendance_record(sid, conf, "success")
                        else:
                            st.error("Access Denied: Identity mismatch.")
                            add_attendance_record(sid, conf, "failed")
                else:
                    st.warning("No face detected.")

    with tab2:
        st.subheader("🔐 System Management")
        
        # Check against Railway environment variable
        admin_pass = st.text_input("Admin Secret Key", type="password")
        
        if admin_pass:
            if admin_pass == ADMIN_SECRET:
                st.success("Access Granted")
                
                if st.button("🔄 Sync & Regenerate Face Database"):
                    # This is the heavy part - we now have progress tracking
                    if generate_encodings():
                        st.cache_resource.clear()
                        st.rerun()
            else:
                st.error("Invalid Secret Key")
        else:
            st.info("Enter the ADMIN_SECRET to unlock sync features.")

if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_encodings()
    else:
        main()