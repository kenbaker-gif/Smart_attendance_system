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
            .eq("id", student_id).limit(1).execute()
        institution_id = resp.data[0].get("institution_id", "NKU") if resp.data else "NKU"

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


# ── Institution self-service registration ──────────────────────────────────
import re
import unicodedata

def generate_institution_id(name: str) -> str:
    """Generate a short unique ID from university name e.g. Kampala University → KU"""
    # Normalize and extract uppercase letters
    words = name.strip().upper().split()
    # Filter out common words
    stop = {"OF", "THE", "AND", "FOR", "A", "AN", "IN", "AT", "TO"}
    words = [w for w in words if w not in stop]
    if len(words) >= 3:
        code = "".join(w[0] for w in words[:3])
    elif len(words) == 2:
        code = words[0][:2] + words[1][0]
    else:
        code = words[0][:3]
    return re.sub(r'[^A-Z0-9]', '', code)[:5]


@app.post("/register-institution")
async def register_institution(
    university_name: str = Form(...),
    admin_full_name: str = Form(...),
    admin_email:     str = Form(...),
    phone:           str = Form(...),
    logo:            UploadFile = File(None),  # optional
):
    """Self-service institution registration with email verification."""

    # ── Validate inputs ────────────────────────────────────────────────
    if not all([university_name.strip(), admin_full_name.strip(),
                admin_email.strip(), phone.strip()]):
        raise HTTPException(status_code=400, detail="All fields are required.")

    email = admin_email.strip().lower()
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        raise HTTPException(status_code=400, detail="Invalid email address.")

    # ── Generate institution ID ────────────────────────────────────────
    base_id = generate_institution_id(university_name)
    inst_id = base_id

    # Check for conflicts and append number if needed
    existing = supabase.table("institutions").select("id").execute().data
    existing_ids = {r["id"] for r in existing}
    counter = 1
    while inst_id in existing_ids:
        inst_id = f"{base_id}{counter}"
        counter += 1

    # ── Upload logo if provided ────────────────────────────────────────
    logo_url = None
    if logo and logo.filename:
        logo_bytes = await logo.read()
        if len(logo_bytes) > 0:
            logo_path = f"logos/{inst_id}.png"
            try:
                supabase.storage.from_("raw_faces").remove([logo_path])
            except:
                pass
            supabase.storage.from_("raw_faces").upload(
                logo_path, logo_bytes,
                file_options={"content-type": logo.content_type or "image/png"}
            )
            logo_url = supabase.storage.from_("raw_faces").get_public_url(logo_path)

    # ── Create Supabase Auth user ──────────────────────────────────────
    try:
        auth_response = supabase.auth.admin.create_user({
            "email":            email,
            "email_confirm":    False,  # requires email verification
            "user_metadata":    {
                "full_name":       admin_full_name.strip(),
                "institution_id":  inst_id,
            }
        })
        user_id = auth_response.user.id
    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already exists" in error_msg.lower():
            raise HTTPException(status_code=409, detail="Email already registered.")
        raise HTTPException(status_code=500, detail=f"Auth error: {error_msg}")

    # ── Insert institution ─────────────────────────────────────────────
    try:
        supabase.table("institutions").insert({
            "id":          inst_id,
            "name":        university_name.strip(),
            "plan":        "trial",
            "is_active":   True,
            "admin_email": email,
            "phone":       phone.strip(),
            "logo_url":    logo_url,
        }).execute()
    except Exception as e:
        # Rollback auth user if institution insert fails
        try: supabase.auth.admin.delete_user(user_id)
        except: pass
        raise HTTPException(status_code=500, detail=f"Institution creation failed: {str(e)}")

    # ── Create profile ─────────────────────────────────────────────────
    try:
        supabase.table("profiles").insert({
            "id":             user_id,
            "full_name":      admin_full_name.strip(),
            "is_admin":       True,
            "institution_id": inst_id,
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile creation failed: {str(e)}")

    # ── Send verification email ────────────────────────────────────────
    try:
        supabase.auth.admin.invite_user_by_email(email)
    except:
        pass  # Verification email sent automatically on create_user

    return {
        "success":          True,
        "institution_id":   inst_id,
        "institution_name": university_name.strip(),
        "message":          f"Registration successful. Check {email} to verify your account and start your 30-day trial.",
        "trial_days":       30,
    }


@app.get("/check-trial/{institution_id}")
def check_trial(institution_id: str):
    """Check if an institution's trial is still active."""
    try:
        resp = supabase.table("institutions") \
            .select("plan, is_active, trial_ends_at, name") \
            .eq("id", institution_id) \
            .limit(1).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Institution not found.")
        inst = resp.data[0]
        from datetime import datetime, timezone
        trial_ends = inst.get("trial_ends_at")
        is_active  = inst.get("is_active", False)
        plan       = inst.get("plan", "trial")

        if not is_active:
            return {"active": False, "reason": "Account suspended."}
        if plan == "paid":
            return {"active": True, "plan": "paid"}
        if trial_ends:
            ends_at = datetime.fromisoformat(trial_ends.replace("Z", "+00:00"))
            days_left = (ends_at - datetime.now(timezone.utc)).days
            if days_left <= 0:
                return {"active": False, "reason": "Trial expired.", "days_left": 0}
            return {"active": True, "plan": "trial", "days_left": days_left}
        return {"active": True, "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))