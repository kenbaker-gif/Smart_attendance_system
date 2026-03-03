import os
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Attendance — Upload Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET        = "raw_faces"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

MVP_URL      = os.getenv("MVP_URL", "https://smartattendancemvp-production.up.railway.app")
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")


async def trigger_sync_in_background():
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{MVP_URL}/admin/sync-encodings",
                headers={"Authorization": f"Bearer {ADMIN_SECRET}"}
            )
            print(f"✅ Auto-sync triggered: {resp.status_code}")
    except Exception as e:
        print(f"❌ Auto-sync failed: {e}")


@app.get("/")
def home():
    return {"status": "Smart Attendance: OTA ACTIVE"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-student-face")
async def upload_student(
    background_tasks: BackgroundTasks,
    student_id:     str        = Form(...),
    name:           str        = Form(...),
    institution_id: str        = Form(default="NKU"),
    file:           UploadFile = File(...),
):
    if not student_id.strip():
        raise HTTPException(status_code=400, detail="student_id cannot be empty.")
    if not name.strip():
        raise HTTPException(status_code=400, detail="name cannot be empty.")
    if not institution_id.strip():
        raise HTTPException(status_code=400, detail="institution_id cannot be empty.")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file.content_type}'.")

    file_content = await file.read()

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ✅ New path: institution_id/student_id/filename
    # e.g. NKU/2400102415/1.jpg
    file_path = f"{institution_id.strip()}/{student_id.strip()}/{file.filename}"

    try:
        try:
            supabase.storage.from_(BUCKET).remove([file_path])
        except Exception:
            pass

        supabase.storage.from_(BUCKET).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type},
        )

        image_url = supabase.storage.from_(BUCKET).get_public_url(file_path)

        supabase.table("students").upsert({
            "id":             student_id.strip(),
            "name":           name.strip(),
            "institution_id": institution_id.strip(),
        }).execute()

        if file.filename == "4.jpg":
            print(f"📸 4th photo uploaded for {student_id} — triggering auto-sync...")
            background_tasks.add_task(trigger_sync_in_background)

        return {
            "success":        True,
            "student_id":     student_id.strip(),
            "name":           name.strip(),
            "institution_id": institution_id.strip(),
            "image_url":      image_url,
            "file_path":      file_path,
            "sync_triggered": file.filename == "4.jpg",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/students")
def list_students(institution_id: str = None):
    try:
        query = supabase.table("students").select("*").order("name")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        response = query.execute()
        return {"students": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/students/{student_id}")
def delete_student(student_id: str):
    """Remove a student and their images from storage."""
    try:
        # Get institution_id to build correct path
        resp = supabase.table("students").select("institution_id") \
            .eq("id", student_id).maybe_single().execute()
        institution_id = resp.data.get("institution_id", "NKU") if resp.data else "NKU"

        # List and delete files from new path
        folder = f"{institution_id}/{student_id}"
        files = supabase.storage.from_(BUCKET).list(folder)
        if files:
            paths = [f"{folder}/{f['name']}" for f in files]
            supabase.storage.from_(BUCKET).remove(paths)

        # Delete attendance records first, then student
        supabase.table("attendance_records").delete().eq("student_id", student_id).execute()
        supabase.table("students").delete().eq("id", student_id).execute()

        return {"success": True, "deleted_student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))