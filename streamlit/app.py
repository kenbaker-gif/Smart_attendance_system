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
RAW_FACES_DIR = DATA_DIR / "raw_faces"
ENCODINGS_PATH = DATA_DIR / "encodings_insightface.pkl"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "attendance.log"

for d in [DATA_DIR, RAW_FACES_DIR, LOG_DIR]:
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
# Process Pipeline
# -----------------------------
def generate_encodings() -> bool:
    if not USE_SUPABASE or not supabase:
        return False
    
    try:
        # 1. Download Images from Bucket
        folders = supabase.storage.from_(SUPABASE_BUCKET).list()
        for folder in folders:
            student_id = folder['name']
            if student_id.startswith('.') or student_id == "encodings": continue
            
            files = supabase.storage.from_(SUPABASE_BUCKET).list(student_id)
            for f_info in files:
                file_name = f_info['name']
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    remote_path = f"{student_id}/{file_name}"
                    local_path = RAW_FACES_DIR / student_id / file_name
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    data = supabase.storage.from_(SUPABASE_BUCKET).download(remote_path)
                    with open(local_path, "wb") as f: f.write(data)
        
        # 2. Generate Encodings
        student_dirs = sorted([p for p in RAW_FACES_DIR.iterdir() if p.is_dir()])
        if not student_dirs: return False

        encodings, ids = [], []
        engine = get_insightface()
        for s_dir in student_dirs:
            student_id = s_dir.name
            img_files = [p for p in s_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            for img_p in img_files:
                img_bgr = cv2.imread(str(img_p))
                if img_bgr is None: continue
                faces = engine.get(img_bgr)
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    encodings.append(face.embedding)
                    ids.append(student_id)

        if not encodings: return False
        
        # 3. Save & Upload .pkl
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        
        with open(ENCODINGS_PATH, "rb") as f:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                path=ENCODINGS_REMOTE_PATH.lstrip('/'),
                file=f,
                file_options={"upsert": "true"}
            )
        
        # Cleanup memory after processing
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"Generate Error: {e}")
        return False

@st.cache_resource
def load_encodings():
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
            st.warning("Face database is empty. Please sync in the Admin Panel.")
            
        sid = st.text_input("Enter Student ID (e.g., 2400102415)")
        img_file = st.camera_input("Capture Face for Verification")
        
        if sid and img_file:
            if st.button("Run Verification"):
                img = Image.open(img_file).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                with st.spinner("Analyzing biometric data..."):
                    faces = get_insightface().get(img_bgr)
                
                if faces:
                    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
                    
                    if known_encs.size > 0:
                        # Cosine Similarity check
                        dists = 1.0 - np.dot(known_encs, captured_emb)
                        idx = np.argmin(dists)
                        conf = float(1.0 - dists[idx])
                        
                        # Verify against user context: matching ID and threshold
                        if dists[idx] < DEFAULT_THRESHOLD and str(known_ids[idx]) == str(sid).strip():
                            st.success(f"Verified: {sid} (Confidence: {conf:.2f})")
                            st.balloons()
                            add_attendance_record(sid, conf, "success")
                        else:
                            st.error("Verification Failed: Identity mismatch or face not recognized.")
                            add_attendance_record(sid, conf, "failed")
                else:
                    st.warning("No face detected in the frame. Please try again.")

    with tab2:
        st.subheader("🔐 Database Management")
        
        # Using the ADMIN_SECRET synced from Railway
        admin_input = st.text_input("Admin Password", type="password")
        
        if admin_input:
            if admin_input == ADMIN_SECRET:
                st.success("Authenticated")
                st.info("Warning: Regenerating encodings will re-download all images and update the cloud database.")
                
                if st.button("🔄 Sync & Regenerate Encodings"):
                    with st.spinner("Processing... This may take a minute."):
                        if generate_encodings():
                            st.success("Cloud database synchronized!")
                            st.cache_resource.clear() # Force reload of encodings
                            st.rerun()
                        else:
                            st.error("Sync failed. Check logs.")
            else:
                st.error("Incorrect Password")
        else:
            st.write("Please authenticate to access system management tools.")

if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_encodings()
    else:
        main()