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
ENCODINGS_REMOTE_PATH = os.getenv(
    "ENCODINGS_REMOTE_PATH",
    "raw_faces/encodings/encodings_insightface.pkl"
)

# -----------------------------
# Logging Setup
# -----------------------------
logger = logging.getLogger("attendance_system")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3
    )
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

# -----------------------------
# Supabase Configuration
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

# -----------------------------
# Supabase Helpers
# -----------------------------
def sync_images_from_supabase(target_dir: Path):
    if not supabase:
        return
    try:
        folders = supabase.storage.from_(SUPABASE_BUCKET).list()
        for folder in folders:
            student_id = folder["name"]
            if student_id.startswith("."):
                continue
            files = supabase.storage.from_(SUPABASE_BUCKET).list(student_id)
            for f_info in files:
                name = f_info["name"]
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    remote = f"{student_id}/{name}"
                    local = target_dir / student_id / name
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with open(local, "wb") as f:
                        f.write(
                            supabase.storage
                            .from_(SUPABASE_BUCKET)
                            .download(remote)
                        )
    except Exception as e:
        logger.error(f"Cloud sync error: {e}")

# -----------------------------
# Data Models
# -----------------------------
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
        app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=det_size)
        return app
    except Exception as e:
        st.error(f"InsightFace error: {e}")
        return None

# -----------------------------
# Utilities
# -----------------------------
def normalize_encodings(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms

def add_attendance_record(student_id: str, confidence: float, status: str):
    now = datetime.now()
    if "log_cache" not in st.session_state:
        st.session_state.log_cache = {}

    if status == "success":
        last = st.session_state.log_cache.get(student_id)
        if last and (now - last) < timedelta(seconds=60):
            return

    logger.info(f"{student_id} | {status} | {confidence:.2f}")

    if USE_SUPABASE and supabase:
        try:
            record = AttendanceRecordIn(
                student_id=student_id,
                confidence=confidence,
                detection_method="insightface_buffalo_s",
                verified=status
            )
            supabase.table("attendance_records") \
                .insert(record.model_dump()) \
                .execute()

            if status == "success":
                st.session_state.log_cache[student_id] = now

        except Exception as e:
            logger.error(f"DB insert error: {e}")

# -----------------------------
# Encoding Pipeline
# -----------------------------
def generate_encodings(
    images_dir: Path = RAW_FACES_DIR,
    output_path: Path = ENCODINGS_PATH
) -> bool:

    if USE_SUPABASE:
        sync_images_from_supabase(images_dir)

    image_paths = (
        list(images_dir.glob("**/*.[jJ][pP][gG]")) +
        list(images_dir.glob("**/*.[pP][nN][gG]"))
    )

    if not image_paths:
        return False

    engine = get_insightface()
    if not engine:
        return False

    encodings, ids = [], []

    for img_p in image_paths:
        student_id = img_p.parent.name
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        faces = engine.get(img)
        if faces:
            face = max(
                faces,
                key=lambda f:
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
            encodings.append(face.embedding)
            ids.append(student_id)

    if not encodings:
        return False

    try:
        arr = normalize_encodings(
            np.array(encodings, dtype=np.float32)
        )

        with open(output_path, "wb") as f:
            pickle.dump(
                {"encodings": arr, "ids": np.array(ids)},
                f
            )

        # ✅ Upload to Supabase (critical for Railway)
        if USE_SUPABASE and supabase:
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                "raw_faces/encodings/encodings_insightface.pkl",
                ENCODINGS_PATH.read_bytes(),
                upsert=True
            )
            logger.info("Encodings uploaded to Supabase")

        return True

    except Exception as e:
        logger.error(f"Encoding generation failed: {e}")
        return False

# -----------------------------
# Load Encodings
# -----------------------------
def load_encodings():
    if not ENCODINGS_PATH.exists() or ENCODINGS_PATH.stat().st_size < 100:
        if USE_SUPABASE and supabase:
            try:
                data = supabase.storage \
                    .from_(SUPABASE_BUCKET) \
                    .download(ENCODINGS_REMOTE_PATH)
                with open(ENCODINGS_PATH, "wb") as f:
                    f.write(data)
            except Exception as e:
                logger.error(f"Encoding download failed: {e}")

    if ENCODINGS_PATH.exists():
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                data = pickle.load(f)
            return (
                normalize_encodings(np.array(data["encodings"])),
                [str(i) for i in data["ids"]]
            )
        except Exception:
            pass

    return np.array([]), []

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Smart Attendance",
        page_icon="📸",
        layout="wide"
    )

    st.title("📸 Biometric Attendance System")

    engine = get_insightface()
    if not engine:
        st.stop()

    known_encs, known_ids = load_encodings()

    tab1, tab2 = st.tabs(["🎯 Verification", "⚙️ Admin Panel"])

    with tab1:
        sid = st.text_input("Student ID")
        img_file = st.camera_input("Capture Face")

        if sid and img_file and st.button("Verify Identity"):
            img = Image.open(img_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            faces = engine.get(img_bgr)

            if faces and known_encs.size > 0:
                face = faces[0]
                emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-10)
                dists = 1 - np.dot(known_encs, emb)
                idx = np.argmin(dists)

                if dists[idx] < DEFAULT_THRESHOLD and known_ids[idx] == sid:
                    st.success(f"WELCOME {sid}")
                    add_attendance_record(
                        sid,
                        1 - dists[idx],
                        "success"
                    )
                else:
                    st.error("Verification Failed")
            else:
                st.error("Database Empty. Sync in Admin Panel.")

    with tab2:
        if st.button("🔄 Sync Cloud Data"):
            with st.spinner("Syncing..."):
                if generate_encodings():
                    st.success("Synced successfully")
                    st.rerun()
                else:
                    st.error("Sync failed")

# -----------------------------
# Entry
# -----------------------------
if __name__ == "__main__":
    if "--generate" in sys.argv:
        generate_encodings()
    else:
        main()
