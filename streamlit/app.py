import sys
from pathlib import Path
import os
import streamlit as st
import pickle
import face_recognition
import numpy as np
from PIL import Image

# -----------------------------
# --- Project Path Setup ------
# -----------------------------
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))  # Add project root so `app` imports work

# -----------------------------
# --- App Imports -------------
# -----------------------------
from app.utils.encoding_utils import generate_encodings, USE_SUPABASE, download_all_supabase_images
from app.database import add_attendance_record

# -----------------------------
# --- Supabase Setup ----------
# -----------------------------
supabase = None
SUPABASE_BUCKET = None

if USE_SUPABASE:
    from supabase import create_client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")

    if not SUPABASE_URL or not SUPABASE_KEY or not SUPABASE_BUCKET:
        st.error("❌ Supabase configuration missing in environment variables.")
    else:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# --- Streamlit Page Setup ----
# -----------------------------
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📸 Smart Attendance - Camera Verification")

# -----------------------------
# --- Paths -------------------
# -----------------------------
ENCODINGS_PATH = ROOT_DIR / "data" / "encodings_facenet.pkl"
IMAGES_DIR = ROOT_DIR / "data" / "raw_faces"

# -----------------------------
# --- Helper Functions --------
# -----------------------------
def _safe_get(data_dict, *keys):
    for k in keys:
        if k in data_dict and data_dict[k] is not None:
            return data_dict[k]
    return None

def _to_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return [value]

def compare_128dim_encodings(known_encodings, uploaded_encoding, threshold=0.6):
    distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
    min_distance = np.min(distances)
    matched_index = np.argmin(distances)
    return min_distance, matched_index, min_distance < threshold

def compare_512dim_encodings(known_encodings, uploaded_encoding, threshold=0.5):
    distances = np.linalg.norm(np.array(known_encodings) - uploaded_encoding, axis=1)
    min_distance = np.min(distances)
    matched_index = np.argmin(distances)
    return min_distance, matched_index, min_distance < threshold

# -----------------------------
# --- Load Encodings ----------
# -----------------------------
@st.cache_resource
def load_encodings():
    if not ENCODINGS_PATH.exists():
        st.info("📂 Encodings file not found. Generating from student images...")
        success = generate_encodings(images_dir=str(IMAGES_DIR), output_path=str(ENCODINGS_PATH))
        if not success:
            st.error("❌ Failed to generate encodings. Check data/raw_faces folder.")
            return [], [], None

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
        encs_raw = _safe_get(data, "encodings", "encodings_facenet")
        ids_raw = _safe_get(data, "ids", "labels")
        known_encodings = _to_list(encs_raw) if encs_raw is not None else []
        known_ids = [str(i) for i in _to_list(ids_raw)] if ids_raw is not None else []
        encoding_dim = len(known_encodings[0]) if known_encodings else None
        return known_encodings, known_ids, encoding_dim
    except Exception as e:
        st.error(f"❌ Failed to load encodings: {e}")
        return [], [], None

known_encodings, known_ids, encoding_dim = load_encodings()
threshold = 0.6 if encoding_dim == 128 else 0.5

if known_encodings:
    st.success(f"✅ Loaded {len(known_ids)} encodings ({encoding_dim}-dim)")
else:
    st.warning("⚠️ No encodings loaded. Check your setup.")

# -----------------------------
# --- Camera Verification -----
# -----------------------------
st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📷 Capture Image")
    camera_input = st.camera_input("Point camera at your face")

with col2:
    st.subheader("ℹ️ Instructions")
    st.info("""
    1. Look directly at camera
    2. Ensure good lighting
    3. Face must be clearly visible
    4. Click "Verify" after capture
    """)

if camera_input:
    st.divider()
    image = Image.open(camera_input).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="Captured Image", width='stretch')

    boxes = face_recognition.face_locations(img_array, model="hog")
    used_model = "HOG"

    if len(boxes) == 0:
        with st.spinner("🔄 HOG failed, trying CNN model (slower)..."):
            try:
                boxes = face_recognition.face_locations(img_array, model="cnn")
                used_model = "CNN"
            except Exception as e:
                st.error(f"❌ CNN detection failed: {e}")
                boxes = []

    if len(boxes) == 0:
        st.error("❌ No face detected in image")
        st.info("💡 Try: better lighting, closer to camera, face centered")
    else:
        st.info(f"✅ Face detected using {used_model} model")
        try:
            uploaded_encodings = face_recognition.face_encodings(img_array, boxes)
        except Exception as e:
            st.error(f"❌ Failed to generate encoding: {e}")
            uploaded_encodings = []

        if uploaded_encodings:
            st.divider()
            st.subheader("🔍 Verification Results")
            student_id_input = st.text_input("Enter your Student ID to verify", placeholder="e.g., 2400102415")
            
            if st.button("✅ Verify Identity", width='stretch', type="primary"):
                if not student_id_input:
                    st.error("❌ Please enter your Student ID")
                else:
                    recognized = False
                    for enc in uploaded_encodings:
                        if len(enc) != encoding_dim:
                            st.error(f"❌ Encoding dimension mismatch: live={len(enc)} stored={encoding_dim}")
                            break
                        if encoding_dim == 128:
                            min_distance, matched_index, is_match = compare_128dim_encodings(known_encodings, enc, threshold)
                        else:
                            min_distance, matched_index, is_match = compare_512dim_encodings(known_encodings, enc, threshold)
                        matched_id = known_ids[matched_index]
                        confidence = max(0, min(1, 1 - (min_distance / threshold)))
                        st.info(f"**Best match:** {matched_id} ({confidence:.1%} confidence)")

                        if is_match and matched_id == student_id_input:
                            st.success(f"✅ **VERIFICATION SUCCESSFUL**\n\nStudent ID: {student_id_input}\nConfidence: {confidence:.1%}\nDistance: {min_distance:.3f}")
                            st.balloons()
                            recognized = True
                            add_attendance_record(student_id=student_id_input, confidence=float(confidence), detection_method=used_model, verified="success")
                            break

                    if not recognized:
                        final_confidence = confidence if 'confidence' in locals() else 0.0
                        add_attendance_record(student_id=student_id_input, confidence=float(final_confidence), detection_method=used_model, verified="failed")
                        st.error("❌ **VERIFICATION FAILED**\n\nID mismatch or low confidence. Please try again with better lighting or different angle.")

# -----------------------------
# --- Admin Panel -------------
# -----------------------------
st.divider()
with st.expander("🔧 Admin Panel"):
    # 1. Metrics Display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Known Students", len(known_ids))
        st.metric("Encoding Dim", encoding_dim)
    with col2:
        st.metric("Threshold", f"{threshold:.2f}")
    
    # 2. Retrain Button Logic (using the corrected call)
    # The 'type="secondary"' is used to distinguish the button visually.
    if st.button("🔄 Retrain Encodings", key="retrain_button_final", type="secondary"):
        st.info("⏳ Retraining encodings...")

        # FIX: Call generate_encodings without supabase_client or bucket_name 
        # as those are handled by global variables inside encoding_utils.py
        success = generate_encodings(
            images_dir=str(IMAGES_DIR), 
            output_path=str(ENCODINGS_PATH)
        )
        
        if success:
            st.success("✅ Encodings retrained successfully!")
            # Clear resource cache to force 'load_encodings' to re-read the new file
            st.cache_resource.clear() 
            # Rerun the script to reload the new encodings immediately
            st.experimental_rerun()
        else:
            st.error("❌ Retrain failed. Check data/raw_faces folder or Supabase configuration.")