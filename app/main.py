from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Smart Attendance System")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

# --- Supabase + Admin Setup ---
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")
ADMIN_SECRET  = os.getenv("ADMIN_SECRET")
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)
security      = HTTPBearer()

def verify_admin(token = Depends(security)):
    if token.credentials != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

# ── Attendance records — fixed endpoint name ───────────────────────────────
@app.get("/admin/attendance-records")
def get_attendance(admin=Depends(verify_admin)):
    response = supabase.table("attendance_records") \
        .select("*, students(name)") \
        .order("timestamp", desc=True) \
        .execute()
    return response.data

# ── Summary — fixed field names to match actual DB schema ─────────────────
@app.get("/admin/attendance_summary")
def get_summary(admin=Depends(verify_admin)):
    response = supabase.table("attendance_records").select("*").execute()
    rows = response.data

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

# ── Students list ──────────────────────────────────────────────────────────
@app.get("/students")
def get_students(admin=Depends(verify_admin)):
    response = supabase.table("students").select("*").order("name").execute()
    return {"students": response.data, "count": len(response.data)}