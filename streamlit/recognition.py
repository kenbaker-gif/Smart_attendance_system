import os
import pickle
from pathlib import Path
import face_recognition
from app.database import add_attendance_record, register_student
from app.models import Student
from supabase import create_client

# --- CONFIG ---
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"

if USE_SUPABASE:
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_supabase_images(student_id: str, local_dir="data/raw_faces"):
    """Download images from Supabase bucket to local folder for processing."""
    local_path = Path(local_dir) / student_id
    local_path.mkdir(parents=True, exist_ok=True)

    files = supabase.storage.from_(SUPABASE_BUCKET).list(f"{student_id}/")
    for file_info in files:
        file_name = file_info["name"].split("/")[-1]
        local_file_path = local_path / file_name
        data = supabase.storage.from_(SUPABASE_BUCKET).download(file_info["name"])
        with open(local_file_path, "wb") as f:
            f.write(data)

def generate_encodings(images_dir="data/raw_faces", output_path="data/encodings_facenet.pkl"):
    """
    Generate face encodings from Supabase bucket or local folder.

    Args:
        images_dir (str): Local folder to store images temporarily
        output_path (str): Path to save encodings pickle file

    Returns:
        bool: True if successful, False otherwise
    """
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    encodings = []
    ids = []

    # --- Always fetch from Supabase if enabled ---
    if USE_SUPABASE:
        print("📦 Fetching images from Supabase...")
        # Fetch all student folders dynamically
        student_folders = [f["name"].split("/")[0] for f in supabase.storage.from_(SUPABASE_BUCKET).list("", {"limit":1000})]
        student_folders = list(set(student_folders))  # unique
        for student_id in student_folders:
            download_supabase_images(student_id, images_dir)

    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return False

    print(f"📂 Scanning {images_dir} for images...")

    for student_id_folder in sorted(images_dir.iterdir()):
        if not student_id_folder.is_dir():
            continue

        student_id = student_id_folder.name
        image_files = list(student_id_folder.glob("*.jpg")) + list(student_id_folder.glob("*.png"))

        if not image_files:
            print(f"⚠️  No images found for {student_id}")
            continue

        print(f"📸 Processing student {student_id} ({len(image_files)} images)...")

        for img_path in image_files:
            try:
                image = face_recognition.load_image_file(str(img_path))
                face_locations = face_recognition.face_locations(image, model="hog")

                if not face_locations:
                    print(f"  ⚠️  No face detected in {img_path.name}")
                    continue

                face_encodings = face_recognition.face_encodings(image, face_locations)

                if face_encodings:
                    encodings.append(face_encodings[0])  # Use first face
                    ids.append(student_id)
                    print(f"  ✅ {img_path.name}")
            except Exception as e:
                print(f"  ❌ Error processing {img_path.name}: {e}")

    if encodings:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, "wb") as f:
                pickle.dump({"encodings": encodings, "ids": ids}, f)
            print(f"\n✅ Saved {len(encodings)} encodings to {output_path}")
            return True
        except Exception as e:
            print(f"\n❌ Failed to save encodings: {e}")
            return False
    else:
        print("\n❌ No encodings generated. Check your image folder structure.")
        return False

# Allow direct run
if __name__ == "__main__":
    generate_encodings()
