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
ENCODINGS_REMOTE_PATH = os.getenv("ENCODINGS_REMOTE_PATH", "raw_faces/encodings/encodings_insightface.pkl")

# -----------------------------
# Logging Setup
# -----------------------------
logger = logging.getLogger("attendance_system")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

# -----------------------------
# Integrated Supabase Logic
# -----------------------------
USE_SUPABASE = os.getenv("USE_SUPABASE", "").lower() == "true"
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "")

supabase = None
if USE_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Supabase init error: {e}")

def sync_images_from_supabase(target_dir: Path):
    if not supabase: return
    try:
        folders = supabase.storage.from_(SUPABASE_BUCKET).list()
        for folder in folders:
            student_id = folder['name']
            if student_id.startswith('.'): continue
            files = supabase.storage.from_(SUPABASE_BUCKET).list(student_id)
            for f_info in files:
                file_name = f_info['name']
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    remote_path = f"{student_id}/{file_name}"
                    local_path = target_dir / student_id / file_name
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_path, "wb") as f:
                        data = supabase.storage.from_(SUPABASE_BUCKET).download(remote_path)
                        f.write(data)
    except Exception as e:
        print(f"⚠️ Cloud Sync Error: {e}")

class AttendanceRecordIn(BaseModel):
    student_id: str
    confidence: float
    detection_method: str
    verified: str

# -----------------------------
# InsightFace Engine
# -----------------------------
@st.cache_resource(show_spinner="Loading AI Models...")
def get_insightface(det_size=(640, 640)):
    try:
        from insightface.app import FaceAnalysis
        _app = FaceAnalysis(name=INSIGHTFACE_MODEL_NAME, providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=det_size)
        return _app
    except Exception as e:
        if "--generate" not in sys.argv:
            st.error(f"Engine Error: {e}")
        return None

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
    if 'log_cache' not in st.session_state: st.session_state.log_cache = {}
    if status == 'success':
        last_logged = st.session_state.log_cache.get(student_id)
        if last_logged and (current_time - last_logged) < timedelta(seconds=60): return
    logger.info(f"Log: {student_id} | Status: {status} | Conf: {confidence:.2f}")
    if USE_SUPABASE and supabase:
        try:
            record = AttendanceRecordIn(student_id=student_id, confidence=float(confidence), 
                                      detection_method="insightface_buffalo_s", verified=status)
            supabase.table('attendance_records').insert(record.model_dump()).execute()
            if status == 'success': st.session_state.log_cache[student_id] = current_time
            st.toast(f"Synced: {student_id}", icon="✅")
        except Exception as e: logger.error(f"DB Sync Error: {e}")

# -----------------------------
# Process Pipeline
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    if USE_SUPABASE: sync_images_from_supabase(images_dir)
    image_paths = list(images_dir.glob("**/*.[jJ][pP][gG]")) + list(images_dir.glob("**/*.[pP][nN][gG]"))
    if not image_paths: return False
    encodings, ids = [], []
    engine = get_insightface()
    if not engine: return False
    for img_p in image_paths:
        student_id = img_p.parent.name
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None: continue
        faces = engine.get(img_bgr)
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            encodings.append(face.embedding)
            ids.append(student_id)
    if not encodings: return False
    try:
        arr = normalize_encodings(np.array(encodings, dtype=np.float32))
        with open(output_path, "wb") as f:
            pickle.dump({"encodings": arr, "ids": np.array(ids)}, f)
        return True
    except Exception: return False

@st.cache_resource
def load_encodings():
    """FIXED: Forces download on Railway if local file is missing"""
    # 1. Automatic Download from Supabase if not present
    if not ENCODINGS_PATH.exists() or ENCODINGS_PATH.stat().st_size < 100:
        if USE_SUPABASE and supabase:
            try:
                res = supabase.storage.from_(SUPABASE_BUCKET).download(ENCODINGS_REMOTE_PATH)
                data_bytes = res if isinstance(res, (bytes, bytearray)) else getattr(res, 'content', None)
                if data_bytes:
                    with open(ENCODINGS_PATH, "wb") as f: f.write(data_bytes)
            except Exception as e: print(f"Cloud Load Error: {e}")

    # 2. Load the file
    if ENCODINGS_PATH.exists():
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            return normalize_encodings(np.array(data["encodings"])), [str(i) for i in data["ids"]]
        except Exception: return np.array([]), []
    return np.array([]), []

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Smart Attendance", page_icon="📸", layout="wide")
    st.title("📸 Biometric Attendance System")
    engine = get_insightface()
    if not engine: st.stop()
    known_encs, known_ids = load_encodings()
    tab1, tab2 = st.tabs(["🎯 Verification", "⚙️ Admin Panel"])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            sid = st.text_input("Enter Student ID", placeholder="e.g. 2400102415")
            img_file = st.camera_input("Capture Face")
        with col2:
            if sid and img_file and st.button("Verify Identity", use_container_width=True):
                try:
                    img = Image.open(img_file).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    faces = engine.get(img_bgr)
                    if faces:
                        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                        captured_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
                        if known_encs.size > 0:
                            dists = 1.0 - np.dot(known_encs, captured_emb)
                            idx = np.argmin(dists)
                            conf = float(1.0 - dists[idx])
                            if dists[idx] < DEFAULT_THRESHOLD and known_ids[idx] == sid:
                                st.success(f"WELCOME {sid}!")
                                st.balloons()
                                add_attendance_record(sid, conf, "success")
                            else:
                                st.error("Verification Failed.")
                                add_attendance_record(sid, conf, "failed")
                        else: st.error("Database Empty. Sync in Admin Panel.")
                    else: st.warning("No face detected.")
                except Exception as e: st.error(f"Error: {e}")

    with tab2:
        if st.button("🔄 Sync Cloud Data"):
            with st.spinner("Syncing..."):
                if generate_encodings():
                    st.success("Synced!")
                    st.rerun()

if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_encodings()
    else:
        main()