import os
import re
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import Form
from typing import Optional
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# ── Shared dependencies (clients, limiter, auth) ───────────────────────────
from .dep import (
    supabase,
    supabase_admin,
    limiter,
    _bool_flag,
    verify_supabase_token,
    check_admin,
    check_super_admin,
)

# ── Enterprise API v1 router ───────────────────────────────────────────────
from .v1_api import router as v1_router
from .pesapal_router import router as pesapal_router

load_dotenv()

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "live.com", "icloud.com", "aol.com", "protonmail.com",
    "zoho.com", "ymail.com", "mail.com", "googlemail.com"
}

app = FastAPI(title="Smart Attendance — Upload Service")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

BUCKET        = "raw_faces"
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

APP_URL = os.getenv("APP_URL", "https://faceattend.app")
MVP_URL = os.getenv("MVP_URL", "https://smartattendancemvp-production.up.railway.app/").rstrip("/")

if not MVP_URL:
    print("⚠️  WARNING: MVP_URL is not set. Auto-sync after photo upload will be disabled.")
else:
    print(f"✅ MVP_URL = {MVP_URL}")

# ── Mount Enterprise API v1 ────────────────────────────────────────────────
app.include_router(v1_router)
app.include_router(pesapal_router)


# ── Background sync ────────────────────────────────────────────────────────

async def trigger_sync_in_background(token: str):
    if not MVP_URL:
        print("⚠️  Auto-sync skipped: MVP_URL is not configured.")
        return

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{MVP_URL}/admin/sync-encodings",
                headers={"Authorization": f"Bearer {token}"}
            )
            if resp.status_code == 200:
                print(f"✅ Auto-sync complete: {resp.status_code}")
            else:
                print(f"⚠️  Auto-sync rejected: HTTP {resp.status_code} — {resp.text[:300]}")
    except httpx.UnsupportedProtocol as e:
        print(f"❌ Auto-sync failed: MVP_URL is invalid — {e!r} (MVP_URL={repr(MVP_URL)})")
    except httpx.ConnectError as e:
        print(f"❌ Auto-sync failed: cannot reach MVP server — {e!r}")
    except httpx.TimeoutException:
        print(f"❌ Auto-sync timed out after 60s (MVP_URL={repr(MVP_URL)})")
    except Exception as e:
        print(f"❌ Auto-sync failed ({type(e).__name__}): {e!r}")


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

        sync_triggered = False
        if file.filename == "4.jpg":
            token = authorization.replace("Bearer ", "").strip() if authorization else None

            if token and MVP_URL:
                print(f"📸 4th photo uploaded for {student_id} — triggering auto-sync...")
                background_tasks.add_task(trigger_sync_in_background, token)
                sync_triggered = True
            else:
                if not token:
                    print(f"⚠️  Skipping auto-sync for {student_id}: no auth token available.")
                if not MVP_URL:
                    print(f"⚠️  Skipping auto-sync for {student_id}: MVP_URL not configured.")

        return {
            "success":        True,
            "student_id":     student_id.strip(),
            "name":           name.strip(),
            "institution_id": institution_id.strip(),
            "image_url":      image_url,
            "file_path":      file_path,
            "sync_triggered": sync_triggered,
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
        profile_resp = supabase_admin.table("profiles") \
            .select("institution_id, is_super_admin, role") \
            .eq("id", user.id).single().execute()

        is_super_admin = _bool_flag(profile_resp.data.get("is_super_admin") if profile_resp.data else None)
        role           = profile_resp.data.get("role", "") if profile_resp.data else ""
        user_institution_id = profile_resp.data.get("institution_id") if profile_resp.data else None

        is_super = is_super_admin or role == "super_admin"

        if not is_super and not user_institution_id:
            raise HTTPException(status_code=403, detail="Institution admin requires institution_id in profile")

        query = supabase_admin.table("students").select("*").order("name")

        if is_super:
            if institution_id:
                query = query.eq("institution_id", institution_id)
        else:
            query = query.eq("institution_id", institution_id or user_institution_id)

        response = query.execute()
        students = response.data or []
        return {"students": students, "count": len(students)}
    except HTTPException:
        raise
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
        profile_resp = supabase_admin.table("profiles") \
            .select("institution_id, is_super_admin, role") \
            .eq("id", user.id).single().execute()

        is_super_admin = _bool_flag(profile_resp.data.get("is_super_admin") if profile_resp.data else None)
        role           = profile_resp.data.get("role", "") if profile_resp.data else ""
        user_institution_id = profile_resp.data.get("institution_id") if profile_resp.data else None

        is_super = is_super_admin or role == "super_admin"

        if not is_super and not user_institution_id:
            raise HTTPException(status_code=403, detail="Institution admin requires institution_id in profile")

        query = supabase_admin.table("attendance_records") \
            .select("*") \
            .order("timestamp", desc=True) \
            .limit(limit)

        if is_super:
            if institution_id:
                query = query.eq("institution_id", institution_id)
        else:
            query = query.eq("institution_id", institution_id or user_institution_id)

        rows = query.execute().data or []
        return rows
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/attendance_summary")
async def get_summary(
    institution_id: str = None,
    user=Depends(check_admin),
):
    try:
        profile_resp = supabase_admin.table("profiles") \
            .select("institution_id, is_super_admin, role") \
            .eq("id", user.id).single().execute()

        is_super_admin = _bool_flag(profile_resp.data.get("is_super_admin") if profile_resp.data else None)
        role           = profile_resp.data.get("role", "") if profile_resp.data else ""
        user_institution_id = profile_resp.data.get("institution_id") if profile_resp.data else None

        is_super = is_super_admin or role == "super_admin"

        if not is_super and not user_institution_id:
            raise HTTPException(status_code=403, detail="Institution admin requires institution_id in profile")

        query = supabase_admin.table("attendance_records").select("*")
        if is_super:
            if institution_id:
                query = query.eq("institution_id", institution_id)
        else:
            query = query.eq("institution_id", institution_id or user_institution_id)

        rows = query.execute().data or []
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Coordinator invite ─────────────────────────────────────────────────────

@app.post("/invite-coordinator")
async def invite_coordinator(
    full_name:       str           = Form(...),
    email:           str           = Form(...),
    institution_id:  str           = Form(default=None),
    course_unit_id:  Optional[str] = Form(None),          # ← NEW
    user=Depends(check_admin),
):
    profile_resp = supabase_admin.table("profiles") \
        .select("institution_id, is_super_admin, role") \
        .eq("id", user.id).single().execute()

    profile = profile_resp.data
    if not profile:
        raise HTTPException(status_code=403, detail="Admin profile not found.")

    is_super_admin   = _bool_flag(profile.get("is_super_admin"))
    role             = profile.get("role", "")
    user_institution = profile.get("institution_id")
    is_super         = is_super_admin or role == "super_admin"

    if is_super:
        if not institution_id or not institution_id.strip():
            raise HTTPException(
                status_code=400,
                detail="Super admins must provide institution_id when inviting a coordinator."
            )
        target_institution = institution_id.strip()
    else:
        if not user_institution:
            raise HTTPException(status_code=400, detail="Your admin account is not linked to an institution.")
        target_institution = user_institution

    full_name = full_name.strip()
    email     = email.strip().lower()

    if not full_name:
        raise HTTPException(status_code=400, detail="full_name cannot be empty.")
    if not email or not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
        raise HTTPException(status_code=400, detail="Invalid email address.")

    # Normalise course_unit_id — treat blank string as None
    course_unit_id = course_unit_id.strip() if course_unit_id and course_unit_id.strip() else None

    try:
        auth_response = supabase_admin.auth.admin.invite_user_by_email(
            email,
            options={
                "data": {
                    "full_name":      full_name,
                    "institution_id": target_institution,
                    "role":           "coordinator",
                    "course_unit_id": course_unit_id,     # ← NEW (None is fine here)
                },
                "redirect_to": f"{APP_URL}/set-password",
            }
        )
        coordinator_user_id = auth_response.user.id
    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already exists" in error_msg.lower():
            raise HTTPException(status_code=409, detail="This email is already registered.")
        raise HTTPException(status_code=500, detail=f"Invite failed: {error_msg}")

    try:
        profile_data = {
            "id":             coordinator_user_id,
            "full_name":      full_name,
            "institution_id": target_institution,
            "is_admin":       False,
            "is_super_admin": False,
            "role":           "coordinator",
        }
        if course_unit_id:
            profile_data["course_unit_id"] = course_unit_id   # ← NEW

        supabase_admin.table("profiles").insert(profile_data).execute()
    except Exception as e:
        try:
            supabase_admin.auth.admin.delete_user(coordinator_user_id)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Profile creation failed: {str(e)}")

    return {
        "success":        True,
        "coordinator_id": coordinator_user_id,
        "full_name":      full_name,
        "email":          email,
        "institution_id": target_institution,
        "course_unit_id": course_unit_id,                     # ← NEW
        "message":        f"Invite sent to {email}. They will receive an email to set their password.",
    }


# ── List coordinators ──────────────────────────────────────────────────────

@app.get("/coordinators")
async def list_coordinators(
    institution_id: str = None,
    user=Depends(check_admin),
):
    profile_resp = supabase_admin.table("profiles") \
        .select("institution_id, is_super_admin, role") \
        .eq("id", user.id).single().execute()

    profile = profile_resp.data
    if not profile:
        raise HTTPException(status_code=403, detail="Admin profile not found.")

    is_super_admin   = _bool_flag(profile.get("is_super_admin"))
    role             = profile.get("role", "")
    user_institution = profile.get("institution_id")
    is_super         = is_super_admin or role == "super_admin"

    try:
        query = supabase_admin.table("profiles") \
            .select("id, full_name, role, institution_id, course_unit_id, created_at") \
            .eq("role", "coordinator") \
            .order("created_at", desc=True)

        if is_super:
            if institution_id:
                query = query.eq("institution_id", institution_id)
        else:
            if not user_institution:
                raise HTTPException(status_code=400, detail="Admin not linked to an institution.")
            query = query.eq("institution_id", user_institution)

        resp = query.execute()
        coordinators = resp.data or []
        return {"coordinators": coordinators, "count": len(coordinators)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/admin/coordinators/{coordinator_id}/course-unit")
async def update_coordinator_course_unit(
    coordinator_id: str,
    course_unit_id: Optional[str] = Form(None),
    user=Depends(check_admin),
):
    profile_resp = supabase_admin.table("profiles") \
        .select("institution_id, is_super_admin, role") \
        .eq("id", user.id).single().execute()

    profile = profile_resp.data
    if not profile:
        raise HTTPException(status_code=403, detail="Admin profile not found.")

    is_super_admin   = _bool_flag(profile.get("is_super_admin"))
    role             = profile.get("role", "")
    user_institution = profile.get("institution_id")
    is_super         = is_super_admin or role == "super_admin"

    coord_resp = supabase_admin.table("profiles") \
        .select("institution_id, role") \
        .eq("id", coordinator_id).single().execute()
    coord = coord_resp.data
    if not coord:
        raise HTTPException(status_code=404, detail="Coordinator not found.")
    if coord.get("role") != "coordinator":
        raise HTTPException(status_code=400, detail="Target user is not a coordinator.")
    if not is_super and coord.get("institution_id") != user_institution:
        raise HTTPException(status_code=403, detail="Coordinator does not belong to your institution.")

    course_unit_id = course_unit_id.strip() if course_unit_id and course_unit_id.strip() else None
    if course_unit_id:
        unit_resp = supabase_admin.table("course_units") \
            .select("id, institution_id") \
            .eq("id", course_unit_id).single().execute()
        unit = unit_resp.data
        if not unit:
            raise HTTPException(status_code=404, detail="Course unit not found.")
        if not is_super and unit.get("institution_id") != user_institution:
            raise HTTPException(status_code=403, detail="Course unit does not belong to your institution.")

    update_resp = supabase_admin.table("profiles") \
        .update({"course_unit_id": course_unit_id}) \
        .eq("id", coordinator_id).execute()
    if not hasattr(update_resp, 'status_code') or update_resp.status_code >= 400:
        detail = getattr(update_resp, 'data', None)
        raise HTTPException(status_code=500, detail=(str(detail) if detail else 'Failed to update coordinator course unit.'))

    return {
        "success": True,
        "coordinator_id": coordinator_id,
        "course_unit_id": course_unit_id,
    }


@app.delete("/coordinators/{coordinator_id}")
async def remove_coordinator(
    coordinator_id: str,
    user=Depends(check_admin),
):
    profile_resp = supabase_admin.table("profiles") \
        .select("institution_id, is_super_admin, role") \
        .eq("id", user.id).single().execute()

    profile = profile_resp.data
    if not profile:
        raise HTTPException(status_code=403, detail="Admin profile not found.")

    is_super_admin   = _bool_flag(profile.get("is_super_admin"))
    role             = profile.get("role", "")
    user_institution = profile.get("institution_id")
    is_super         = is_super_admin or role == "super_admin"

    coord_resp = supabase_admin.table("profiles") \
        .select("institution_id, role") \
        .eq("id", coordinator_id).limit(1).execute()

    coord = coord_resp.data[0] if coord_resp.data else None
    if not coord:
        raise HTTPException(status_code=404, detail="Coordinator not found.")
    if coord.get("role") != "coordinator":
        raise HTTPException(status_code=400, detail="Target user is not a coordinator.")

    if not is_super and coord.get("institution_id") != user_institution:
        raise HTTPException(status_code=403, detail="Coordinator does not belong to your institution.")

    try:
        supabase_admin.table("profiles").delete().eq("id", coordinator_id).execute()
        supabase_admin.auth.admin.delete_user(coordinator_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Removal failed: {str(e)}")

    return {
        "success":    True,
        "removed_id": coordinator_id,
    }


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
            "plans":       "trial",
            "is_active":   True,
            "status":      "pending",
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
            "role":           "admin",
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
    user=Depends(check_super_admin),
):
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
    try:
        profile_resp = supabase_admin.table("profiles") \
            .select("institution_id, is_super_admin, role") \
            .eq("id", user.id).single().execute()

        if not profile_resp.data:
            raise HTTPException(status_code=403, detail="Admin profile not found")

        is_super_admin = _bool_flag(profile_resp.data.get("is_super_admin"))
        role           = profile_resp.data.get("role", "")
        institution_id = profile_resp.data.get("institution_id")
        is_super       = is_super_admin or role == "super_admin"

        if is_super:
            query = supabase_admin.table("institutions") \
                .select("id, name, admin_email, phone, plans, status, is_active, logo_url") \
                .order("name")
            if status:
                query = query.eq("status", status)
        else:
            if not institution_id:
                raise HTTPException(status_code=403, detail="Institution admin requires institution_id in profile")
            query = supabase_admin.table("institutions") \
                .select("id, name, admin_email, phone, plans, status, is_active, logo_url") \
                .eq("id", institution_id)

        resp = query.execute()
        institutions = resp.data or []
        return {"institutions": institutions, "count": len(institutions)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] /admin/institutions failed: {repr(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ── Plans ──────────────────────────────────────────────────────────────────

@app.get("/plans")
def get_plans():
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
            .select("plans, is_active, trial_ends_at, name, status") \
            .eq("id", institution_id) \
            .limit(1).execute()
        if not resp.data:
            raise HTTPException(status_code=404, detail="Institution not found.")
        inst = resp.data[0]

        status = inst.get("status", "active")
        if status == "pending":
            return {"active": False, "reason": "Institution pending approval."}
        if status == "suspended":
            return {"active": False, "reason": "Account suspended."}

        trial_ends = inst.get("trial_ends_at")
        is_active  = inst.get("is_active", False)
        plans      = inst.get("plans", "trial")

        if not is_active:
            return {"active": False, "reason": "Account suspended."}
        if plans == "paid":
            return {"active": True, "plans": "paid"}
        if trial_ends:
            ends_at   = datetime.fromisoformat(trial_ends.replace("Z", "+00:00"))
            days_left = (ends_at - datetime.now(timezone.utc)).days
            if days_left <= 0:
                return {"active": False, "reason": "Trial expired.", "days_left": 0}
            return {"active": True, "plans": "trial", "days_left": days_left}
        return {"active": True, "plans": plans}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))