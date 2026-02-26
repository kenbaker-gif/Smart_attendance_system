import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Attendance — Upload Service")

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Supabase ───────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "raw_faces"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Smart Attendance: OTA ACTIVE"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Upload student face ────────────────────────────────────────────────────
@app.post("/upload-student-face")
async def upload_student(
    student_id: str = Form(...),
    name: str = Form(...),
    file: UploadFile = File(...),
):
    # 1. Validate inputs
    if not student_id.strip():
        raise HTTPException(status_code=400, detail="student_id cannot be empty.")
    if not name.strip():
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Allowed: jpeg, png, webp.",
        )

    file_content = await file.read()

    # 2. Validate file size
    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_path = f"{student_id.strip()}/{file.filename}"

    try:
        # 3. Remove existing file if present (prevents storage conflicts)
        try:
            supabase.storage.from_(BUCKET).remove([file_path])
        except Exception:
            pass  # file may not exist yet — that's fine

        # 4. Upload to Supabase Storage
        supabase.storage.from_(BUCKET).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type},
        )

        # 5. Get public URL
        image_url = supabase.storage.from_(BUCKET).get_public_url(file_path)

        # 6. Upsert student record in DB
        supabase.table("students").upsert({
            "id": student_id.strip(),
            "name": name.strip(),
        }).execute()

        return {
            "success": True,
            "student_id": student_id.strip(),
            "name": name.strip(),
            "image_url": image_url,
            "file_path": file_path,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ── List enrolled students ─────────────────────────────────────────────────
@app.get("/students")
def list_students():
    """Return all enrolled students from the DB."""
    try:
        response = supabase.table("students").select("*").order("name").execute()
        return {"students": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Delete a student ───────────────────────────────────────────────────────
@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    """Remove a student record and their images from storage."""
    try:
        # List and remove all images for this student
        files = supabase.storage.from_(BUCKET).list(student_id)
        if files:
            paths = [f"{student_id}/{f['name']}" for f in files]
            supabase.storage.from_(BUCKET).remove(paths)

        # Remove from DB
        supabase.table("students").delete().eq("id", student_id).execute()

        return {"success": True, "deleted_student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))