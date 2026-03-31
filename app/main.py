import os
import re
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

load_dotenv()

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "icloud.com", "aol.com", "protonmail.com",
    "zoho.com", "ymail.com", "mail.com", "googlemail.com"
}

app = FastAPI(title="Smart Attendance — Upload Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://faceattend.app",
        "https://www.faceattend.app",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_KEY         = os.getenv("SUPABASE_KEY")         # anon key — for token verification
SUPABASE_SERVICE_KEY = os.getenv("SERVICE_KEY")          # service role — for DB/storage ops

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")
if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SERVICE_KEY in environment.")

# ✅ Two clients — anon for auth verification, service role for data ops
supabase: Client       = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BUCKET        = "raw_faces"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

APP_URL = os.getenv("APP_URL", "https://faceattend.app")
MVP_URL = os.getenv("MVP_URL", "https://smartattendancemvp-production.up.railway.app")


# ── Auth dependencies ──────────────────────────────────────────────────────

async def verify_supabase_token(authorization: str = Header(None)):
    """Verify that the request comes from a valid authenticated Supabase user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_response.user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")

async def check_admin(authorization: str = Header(None)):
    """Verify that the user is authenticated, is an admin, and their institution is active."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization.replace("Bearer ", "").strip()
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        user_id = user_response.user.id
        resp = supabase_admin.table("profiles").select("is_admin, is_super_admin, institution_id") \
            .eq("id", user_id).limit(1).execute()

        def _bool_flag(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)

        profile_data = resp.data[0] if resp.data else None
        is_admin = _bool_flag(profile_data.get("is_admin") if profile_data else None)
        is_super_admin = _bool_flag(profile_data.get("is_super_admin") if profile_data else None)

        if not profile_data or not (is_admin or is_super_admin):
            raise HTTPException(status_code=403, detail="Admin access required")

        # ✅ Check institution status
        institution_id = resp.data[0].get("institution_id")
        if institution_id:
            inst_resp = supabase_admin.table("institutions").select("status") \
                .eq("id", institution_id).limit(1).execute()
            if inst_resp.data:
                status = inst_resp.data[0].get("status", "active")
                if status == "pending":
                    raise HTTPException(
                        status_code=403,
                        detail="Your institution is pending approval. You will be notified once approved."
                    )
                elif status == "suspended":
                    raise HTTPException(
                        status_code=403,
                        detail="Your institution account has been suspended. Contact support."
                    )

        return user_response.user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")


# ── Background sync ────────────────────────────────────────────────────────

async def trigger_sync_in_background(token: str):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{MVP_URL}/admin/sync-encodings",
                headers={"Authorization": f"Bearer {token}"}
            )
            print(f"✅ Auto-sync triggered: {resp.status_code}")
    except Exception as e:
        print(f"❌ Auto-sync failed: {e}")


# ── Static files ───────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/set-password")
def set_password_page():
    return FileResponse(os.path.join("static", "set-password.html"))

@app.get("/dashboard")
def dashboard_page():
    return FileResponse(os.path.join("static", "dashboard.html"))

@app.get("/privacy")
def privacy_page():
    return FileResponse(os.path.join("static", "privacy.html"))

@app.get("/terms")
def terms_page():
    return FileResponse(os.path.join("static", "terms.html"))

# ── Upload student face ────────────────────────────────────────────────────

@app.post("/upload-student-face")
async def upload_student(
    background_tasks: BackgroundTasks,
    student_id:     str        = Form(...),
    name:           str        = Form(...),
    institution_id: str        = Form(default="NKU"),
    file:           UploadFile = File(...),
    authorization:  str        = Header(None),
    user=Depends(verify_supabase_token),
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

    file_path = f"{institution_id.strip()}/{student_id.strip()}/{file.filename}"

    try:
        try:
            supabase_admin.storage.from_(BUCKET).remove([file_path])
        except Exception:
            pass

        supabase_admin.storage.from_(BUCKET).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": file.content_type},
        )

        image_url = supabase_admin.storage.from_(BUCKET).get_public_url(file_path)

        supabase_admin.table("students").upsert({
            "id":             student_id.strip(),
            "name":           name.strip(),
            "institution_id": institution_id.strip(),
        }).execute()

        if file.filename == "4.jpg":
            print(f"📸 4th photo uploaded for {student_id} — triggering auto-sync...")
            background_tasks.add_task(trigger_sync_in_background, authorization)

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


# ── Students ───────────────────────────────────────────────────────────────

@app.get("/students")
async def list_students(
    institution_id: str = None,
    user=Depends(check_admin),
):
    try:
        query = supabase_admin.table("students").select("*").order("name")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        response = query.execute()
        return {"students": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/students/{student_id}")
async def delete_student(
    student_id: str,
    user=Depends(check_admin),
):
    try:
        resp = supabase_admin.table("students").select("institution_id") \
            .eq("id", student_id).limit(1).execute()
        institution_id = resp.data[0].get("institution_id", "NKU") if resp.data else "NKU"

        folder = f"{institution_id}/{student_id}"
        files = supabase_admin.storage.from_(BUCKET).list(folder)
        if files:
            paths = [f"{folder}/{f['name']}" for f in files]
            supabase_admin.storage.from_(BUCKET).remove(paths)

        supabase_admin.table("attendance_records").delete().eq("student_id", student_id).execute()
        supabase_admin.table("students").delete().eq("id", student_id).execute()

        return {"success": True, "deleted_student_id": student_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Attendance records ─────────────────────────────────────────────────────

@app.get("/admin/attendance-records")
async def get_attendance_records(
    institution_id: str = None,
    limit: int = 500,
    user=Depends(check_admin),
):
    try:
        query = supabase_admin.table("attendance_records") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(limit)
        if institution_id:
            query = query.eq("institution_id", institution_id)
        return query.execute().data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/attendance_summary")
async def get_summary(
    institution_id: str = None,
    user=Depends(check_admin),
):
    try:
        query = supabase_admin.table("attendance_records").select("*")
        if institution_id:
            query = query.eq("institution_id", institution_id)
        rows          = query.execute().data
        total_present = sum(1 for r in rows if r.get("verified") == "success")
        total_absent  = sum(1 for r in rows if r.get("verified") == "failed")
        by_student    = {}
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


# ── Institution registration (intentionally public) ────────────────────────

def generate_institution_id(name: str) -> str:
    words = name.strip().upper().split()
    stop  = {"OF", "THE", "AND", "FOR", "A", "AN", "IN", "AT", "TO"}
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
    university_name: str        = Form(...),
    admin_full_name: str        = Form(...),
    admin_email:     str        = Form(...),
    phone:           str        = Form(...),
    logo:            UploadFile = File(None),
):
    if not all([university_name.strip(), admin_full_name.strip(),
                admin_email.strip(), phone.strip()]):
        raise HTTPException(status_code=400, detail="All fields are required.")

    email = admin_email.strip().lower()
    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        raise HTTPException(status_code=400, detail="Invalid email address.")

    # ✅ Block free email domains
    domain = email.split("@")[-1]
    if domain in FREE_EMAIL_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail="Please use your official institutional email address. Personal emails (Gmail, Yahoo, etc.) are not accepted."
        )

    base_id = generate_institution_id(university_name)
    inst_id = base_id
    existing     = supabase_admin.table("institutions").select("id").execute().data
    existing_ids = {r["id"] for r in existing}
    counter = 1
    while inst_id in existing_ids:
        inst_id = f"{base_id}{counter}"
        counter += 1

    logo_url = None
    if logo and logo.filename:
        logo_bytes = await logo.read()
        if len(logo_bytes) > 0:
            logo_path = f"logos/{inst_id}.png"
            try:
                supabase_admin.storage.from_("raw_faces").remove([logo_path])
            except:
                pass
            supabase_admin.storage.from_("raw_faces").upload(
                logo_path, logo_bytes,
                file_options={"content-type": logo.content_type or "image/png"}
            )
            logo_url = supabase_admin.storage.from_("raw_faces").get_public_url(logo_path)

    try:
        supabase_admin.table("institutions").insert({
            "id":          inst_id,
            "name":        university_name.strip(),
            "plan":        "trial",
            "is_active":   True,
            "status":      "pending",           # ✅ All new signups start as pending
            "admin_email": email,
            "phone":       phone.strip(),
            "logo_url":    logo_url,
            "trial_ends_at": (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Institution creation failed: {str(e)}")

    try:
        auth_response = supabase_admin.auth.admin.invite_user_by_email(
            email,
            options={
                "data": {
                    "full_name":      admin_full_name.strip(),
                    "institution_id": inst_id,
                },
                "redirect_to": f"{APP_URL}/set-password",
            }
        )
        user_id = auth_response.user.id
    except Exception as e:
        try:
            supabase_admin.table("institutions").delete().eq("id", inst_id).execute()
        except:
            pass
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already exists" in error_msg.lower():
            raise HTTPException(status_code=409, detail="Email already registered.")
        raise HTTPException(status_code=500, detail=f"Auth error: {error_msg}")

    try:
        supabase_admin.table("profiles").insert({
            "id":             user_id,
            "is_admin":       True,
            "institution_id": inst_id,
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Profile creation failed: {str(e)}")

    return {
        "success":          True,
        "institution_id":   inst_id,
        "institution_name": university_name.strip(),
        "message":          f"Registration received. Check {email} to set your password. Your account will be activated after review.",
        "trial_days":       30,
    }


# ── Institution approval (super-admin only) ────────────────────────────────

@app.patch("/admin/institutions/{institution_id}/status")
async def update_institution_status(
    institution_id: str,
    status: str = Form(...),
    user=Depends(check_admin),
):
    """Approve, suspend, or reactivate an institution. status: pending | active | suspended"""
    if status not in ("pending", "active", "suspended"):
        raise HTTPException(status_code=400, detail="status must be one of: pending, active, suspended")
    try:
        resp = supabase_admin.table("institutions") \
            .update({"status": status}) \
            .eq("id", institution_id) \
            .execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Institution not found.")
        return {
            "success":        True,
            "institution_id": institution_id,
            "status":         status,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/institutions")
async def list_institutions(
    status: str = None,
    user=Depends(check_admin),
):
    """List all institutions, optionally filtered by status."""
    try:
        query = supabase_admin.table("institutions") \
            .select("id, name, admin_email, phone, plan, status, is_active, logo_url") \
            .order("name")
        if status:
            query = query.eq("status", status)
        resp = query.execute()
        return {"institutions": resp.data, "count": len(resp.data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Plans ──────────────────────────────────────────────────────────────────

@app.get("/plans")
def get_plans():
    """Public endpoint — returns active pricing plans for the landing page."""
    try:
        resp = supabase_admin.table("plans") \
            .select("*") \
            .eq("is_active", True) \
            .order("price_usd", desc=False, nullsfirst=False) \
            .execute()
        return {"plans": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Trial check ────────────────────────────────────────────────────────────

@app.get("/check-trial/{institution_id}")
def check_trial(institution_id: str):
    try:
        resp = supabase_admin.table("institutions") \
            .select("plan, is_active, trial_ends_at, name, status") \
            .eq("id", institution_id) \
            .limit(1).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Institution not found.")
        inst = resp.data[0]
        from datetime import datetime, timezone

        # ✅ Check approval status first
        status = inst.get("status", "active")
        if status == "pending":
            return {"active": False, "reason": "Institution pending approval."}
        if status == "suspended":
            return {"active": False, "reason": "Account suspended."}

        trial_ends = inst.get("trial_ends_at")
        is_active  = inst.get("is_active", False)
        plan       = inst.get("plan", "trial")

        if not is_active:
            return {"active": False, "reason": "Account suspended."}
        if plan == "paid":
            return {"active": True, "plan": "paid"}
        if trial_ends:
            ends_at   = datetime.fromisoformat(trial_ends.replace("Z", "+00:00"))
            days_left = (ends_at - datetime.now(timezone.utc)).days
            if days_left <= 0:
                return {"active": False, "reason": "Trial expired.", "days_left": 0}
            return {"active": True, "plan": "trial", "days_left": days_left}
        return {"active": True, "plan": plan}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))