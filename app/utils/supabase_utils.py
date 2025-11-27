import os
from pathlib import Path
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def download_all_images(local_dir="data/raw_faces"):
    """Download all images from Supabase bucket into local folder for retraining."""
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    # List all student folders
    student_folders = supabase.storage.from_(SUPABASE_BUCKET).list()
    for student in student_folders:
        student_path = Path(local_dir) / student["name"]
        student_path.mkdir(parents=True, exist_ok=True)
        files = supabase.storage.from_(SUPABASE_BUCKET).list(student["name"])
        for f in files:
            file_path = student_path / f["name"]
            data = supabase.storage.from_(SUPABASE_BUCKET).download(f"{student['name']}/{f['name']}")
            with open(file_path, "wb") as out:
                out.write(data)
    print(f"[INFO] Downloaded all images to {local_dir}")
