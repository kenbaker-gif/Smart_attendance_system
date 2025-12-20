import os
import sys
import gc
import pickle
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Project root
# -----------------------------
ABSOLUTE_PROJECT_ROOT = "/home/kenbaker-gif/smart_attendance_system"
if sys.path[0] != ABSOLUTE_PROJECT_ROOT:
    sys.path.insert(0, ABSOLUTE_PROJECT_ROOT)

# -----------------------------
# Supabase configuration
# -----------------------------
USE_SUPABASE: bool = os.getenv("USE_SUPABASE", "false").lower() == "true"
SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET", "")

download_all_supabase_images = None
try:
    from app.utils.supabase_utils import download_all_supabase_images
except ImportError:
    print("⚠ WARNING: Could not import supabase_utils. Supabase downloads disabled.")

# -----------------------------
# Memory tuning
# -----------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# -----------------------------
# InsightFace
# -----------------------------
try:
    from insightface.app import FaceAnalysis
except ModuleNotFoundError:
    print("❌ ERROR: insightface not found. Install: pip install insightface[onnx]")
    raise SystemExit(1)

print("🔍 Initializing InsightFace (buffalo_s, CPU)...")
# Use smaller detection size to save RAM
app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(320, 320))
print("✅ InsightFace ready.")

# -----------------------------
# Paths
# -----------------------------
RAW_FACES_DIR = Path(ABSOLUTE_PROJECT_ROOT) / "streamlit" / "data" / "raw_faces"
ENCODINGS_PATH = Path(ABSOLUTE_PROJECT_ROOT) / "streamlit" / "data" / "encodings_insightface.pkl"
RAW_FACES_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def _get_image_paths_for_student(student_dir: Path) -> List[Path]:
    return sorted([p for p in student_dir.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")])

def _largest_face(faces) -> Optional[any]:
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def _generate_face_encoding_from_image(path: Path) -> Optional[np.ndarray]:
    try:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            print(f"❌ Failed to read {path}")
            return None
        # Resize to save memory
        img_bgr = cv2.resize(img_bgr, (320, 320))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if not faces:
            return None
        face = _largest_face(faces)
        if not face or getattr(face, "embedding", None) is None:
            return None
        embedding = np.array(face.embedding, dtype=np.float32)
        del img_bgr, img_rgb, faces, face
        gc.collect()
        return embedding
    except Exception as e:
        print(f"❌ InsightFace failed: {e}")
        gc.collect()
        return None

# -----------------------------
# Generate encodings
# -----------------------------
def generate_encodings(images_dir: Path = RAW_FACES_DIR, output_path: Path = ENCODINGS_PATH) -> bool:
    images_dir.mkdir(parents=True, exist_ok=True)
    
    if USE_SUPABASE and download_all_supabase_images:
        print("📦 Downloading images from Supabase...")
        ok = download_all_supabase_images(SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET, str(images_dir), clear_local=False)
        print("✅ Supabase download complete." if ok else "⚠️ Supabase download skipped or failed.")
    
    encodings, ids = [], []
    student_dirs = sorted([p for p in images_dir.iterdir() if p.is_dir()])

    for student_dir in student_dirs:
        student_id = student_dir.name
        for img_path in _get_image_paths_for_student(student_dir):
            enc = _generate_face_encoding_from_image(img_path)
            if enc is not None:
                encodings.append(enc)
                ids.append(student_id)

    if not encodings:
        print("❌ No encodings generated.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump({"encodings": np.array(encodings, dtype=np.float32), "ids": np.array(ids)}, fh)
    print(f"✅ Saved {len(encodings)} encodings for {len(set(ids))} students → {output_path}")
    return True

# -----------------------------
# CLI entry
# -----------------------------
if __name__ == "__main__":
    generate_encodings()
