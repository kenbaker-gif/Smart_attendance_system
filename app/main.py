import os
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Attendance System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Supabase + Admin Setup ---
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
ADMIN_SECRET  = os.getenv("ADMIN_SECRET", "")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
security      = HTTPBearer()

# --- InsightFace backend for auto-sync ---
MVP_URL = os.getenv("MVP_URL", "https://smartattendancemvp-production.up.railway.app")

BUCKET        = "raw_faces"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# --- Auth ---
def verify_admin(token=Depends(security)):
    if token.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")


# --- Auto-sync after 4th photo ---
async def trigger_sync_in_background():
    """Calls InsightFace backend to rebuild encodings after new student registered."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{MVP_URL}/admin/sync-encodings",
                headers={"Authorization": f"Bearer {ADMIN_SECRET}"}
            )
            print(f"✅ Auto-sync triggered: {resp.status_code}")
    except Exception as e:
        print(f"❌ Auto-sync failed: {e}")


# --- Endpoints ---
@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))


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
    import logging
    logging.warning(f"UPLOAD DEBUG: student_id={student_id!r} name={name!r} institution_id={institution_id!r} content_type={file.content_type!r} filename={file.filename!r}")

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

    file_path = f"{student_id.strip()}/{file.filename}"

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

        # ✅ Auto-trigger sync after 4th photo
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
def list_students(institution_id: str = None, admin=Depends(verify_admin)):
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
    try:
        files = supabase.storage.from_(BUCKET).list(student_id)
        if files:
            paths = [f"{student_id}/{f['name']}" for f in files]
            supabase.storage.from_(BUCKET).remove(paths)
        supabase.table("students").delete().eq("id", student_id).execute()
        return {"success": True, "deleted_student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Admin endpoints for dashboard ---
@app.get("/admin/attendance-records")
def get_attendance(institution_id: str = None, admin=Depends(verify_admin)):
    try:
        query = supabase.table("attendance_records") \
            .select("*, students(name)") \
            .order("timestamp", desc=True)
        if institution_id:
            query = query.eq("institution_id", institution_id)
        return query.execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/attendance_summary")
def get_summary(institution_id: str = None, admin=Depends(verify_admin)):
    try:
        query = supabase.table("attendance_records").select("*")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        rows = query.execute().data
        total_present = sum(1 for r in rows if r.get("verified") == "success")
        total_absent  = sum(1 for r in rows if r.get("verified") == "failed")
        by_student = {}
        for r in rows:
            sid = r.get("student_id") or "Unknown"
            if r.get("verified") == "success":
                by_student[sid] = by_student.get(sid, 0) + 1
        return {
            "total_present": total_present,
            "total_absent":  total_absent,
            "by_student":    by_student,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))