"""
FaceAttend Enterprise API — /v1/ routes
Stack: FastAPI + Supabase + slowapi

API Key format: fa_live_<32-char-hex>
Header:         X-API-Key: fa_live_...
"""

import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel


from .dep import limiter, supabase, supabase_admin, check_admin, _bool_flag, require_enterprise

# ── Router ───────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1", tags=["Enterprise API v1"])


# ── Helper: generate a new API key ───────────────────────────────────────────

def generate_api_key() -> tuple[str, str]:
    """
    Returns (raw_key, key_hash).
    Store only the hash in Supabase. Give the raw key to the user ONCE.
    """
    raw = "fa_live_" + secrets.token_hex(32)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, key_hash


# ── Helper: resolve org_id from JWT user ─────────────────────────────────────

def _get_org_id(user, requested_org_id: Optional[str] = None) -> Optional[str]:
    """
    Super admin: returns requested_org_id if provided, else None (= no filter).
    Regular admin: always returns their own institution_id.
    """
    resp = supabase_admin.table("profiles") \
        .select("institution_id, is_super_admin") \
        .eq("id", user.id) \
        .limit(1).execute()

    profile = resp.data[0] if resp.data else None
    is_super = _bool_flag(profile.get("is_super_admin") if profile else None)

    if is_super:
        return requested_org_id or None  # None = super admin, no org filter

    org_id = profile.get("institution_id") if profile else None
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin account is not linked to an institution.",
        )
    return org_id


# ── Dependency: validate X-API-Key header ────────────────────────────────────

async def validate_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
) -> dict:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header.")

    if not x_api_key.startswith("fa_live_"):
        raise HTTPException(status_code=401, detail="Invalid API key format.")

    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    result = (
        supabase_admin.table("api_keys")
        .select("id, org_id, plan, is_active, name")
        .eq("key_hash", key_hash)
        .limit(1)
        .execute()
    )

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    key_row = result.data[0]

    if not key_row["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This API key has been deactivated.",
        )

    if key_row.get("plan") != "enterprise":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API access requires an Enterprise plan.",
        )

    request.state.org_id = key_row["org_id"]

    supabase_admin.table("api_keys").update(
        {"last_used_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", key_row["id"]).execute()

    return key_row


# ── Schemas ──────────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    name: str = "Default Key"


class MarkAttendanceRequest(BaseModel):
    session_id: str
    student_id: str
    confidence_score: float
    anti_spoof_passed: bool
    marked_at: Optional[str] = None


class AttendanceRecord(BaseModel):
    id: str
    session_id: str
    student_id: str
    confidence_score: float
    anti_spoof_passed: bool
    marked_at: str
    status: str


# ── Key Management Routes ────────────────────────────────────────────────────

@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: CreateKeyRequest,
    org_id: Optional[str] = None,  # super admin passes ?org_id=XXX
    user=Depends(check_admin),
):
    """
    Generate a new API key for the authenticated institution.
    The raw key is returned ONCE — store it securely. Only the hash is saved.
    Super admin must pass ?org_id=<institution_id>.
    Institution must be on the Enterprise plan.

    Auth: JWT (institution admin). Not protected by X-API-Key.
    """
    resolved = _get_org_id(user, org_id)
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Super admin must provide ?org_id=<institution_id>.",
        )

    require_enterprise(resolved)

    raw_key, key_hash = generate_api_key()

    result = supabase_admin.table("api_keys").insert({
        "key_hash":   key_hash,
        "key_suffix": raw_key[-4:],
        "org_id":     resolved,
        "name":       body.name,
        "plan":       "enterprise",
        "is_active":  True,
    }).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key.",
        )

    return {
        "api_key": raw_key,
        "key_id":  result.data[0]["id"],
        "name":    body.name,
        "warning": "Save this key now. It will not be shown again.",
    }


@router.get("/keys")
async def list_api_keys(
    org_id: Optional[str] = None,  # super admin filter; omit = all keys
    user=Depends(check_admin),
):
    """
    List all API keys for the authenticated institution.
    Super admin: returns all keys, or filtered by ?org_id=XXX.
    key_hash is never returned — only metadata.
    Institution must be on the Enterprise plan (skipped for super admin listing all).

    Auth: JWT (institution admin).
    """
    resolved = _get_org_id(user, org_id)

    if resolved:  # None = super admin with no filter — skip plan check
        require_enterprise(resolved)

    query = supabase_admin.table("api_keys") \
        .select("id, name, plan, is_active, created_at, last_used_at, key_suffix")

    if resolved:
        query = query.eq("org_id", resolved)

    result = query.order("created_at", desc=True).execute()
    return {"keys": result.data}


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    user=Depends(check_admin),
):
    """
    Revoke (deactivate) an API key by ID.
    Regular admin: scoped to their institution — cannot revoke another org's key.
    Super admin: can revoke any key.
    Institution must be on the Enterprise plan (skipped for super admin).

    Auth: JWT (institution admin).
    """
    resolved = _get_org_id(user)

    if resolved:  # None = super admin — skip plan check
        require_enterprise(resolved)

    query = supabase_admin.table("api_keys") \
        .update({"is_active": False}) \
        .eq("id", key_id)

    if resolved:  # regular admin: scope to their org
        query = query.eq("org_id", resolved)
    # super admin: no org filter, can revoke any key

    result = query.execute()
    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Key not found or does not belong to your organisation.",
        )


# ── API Routes (protected by X-API-Key) ─────────────────────────────────────

@router.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request, api_key: dict = Depends(validate_api_key)):
    """
    Verify your API key is active and check your current plan.
    """
    return {
        "status":    "ok",
        "org_id":    api_key["org_id"],
        "plan":      api_key["plan"],
        "key_name":  api_key["name"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/attendance/mark", status_code=status.HTTP_201_CREATED)
@limiter.limit("120/minute")
async def mark_attendance(
    request: Request,
    body: MarkAttendanceRequest,
    api_key: dict = Depends(validate_api_key),
):
    """
    Mark a student as present for a session.
    Requires anti_spoof_passed=true and confidence_score >= 0.75.
    """
    if not body.anti_spoof_passed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Anti-spoofing check failed. Attendance not recorded.",
        )

    if body.confidence_score < 0.75:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Confidence score {body.confidence_score:.2f} is below minimum threshold (0.75).",
        )

    payload = {
        "session_id":        body.session_id,
        "student_id":        body.student_id,
        "org_id":            api_key["org_id"],
        "confidence_score":  body.confidence_score,
        "anti_spoof_passed": body.anti_spoof_passed,
        "status":            "present",
    }
    if body.marked_at:
        payload["marked_at"] = body.marked_at

    result = supabase_admin.table("attendance_records").insert(payload).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record attendance.",
        )

    return {"success": True, "record": result.data[0]}


@router.get("/attendance/session/{session_id}")
@limiter.limit("60/minute")
async def get_session_attendance(
    request: Request,
    session_id: str,
    api_key: dict = Depends(validate_api_key),
):
    """
    Fetch all attendance records for a session.
    Scoped to the org that owns the session.
    """
    result = (
        supabase_admin.table("attendance_records")
        .select("*")
        .eq("session_id", session_id)
        .eq("org_id", api_key["org_id"])
        .order("marked_at", desc=False)
        .execute()
    )

    return {
        "session_id": session_id,
        "total":      len(result.data),
        "records":    result.data,
    }


@router.get("/attendance/student/{student_id}")
@limiter.limit("60/minute")
async def get_student_attendance(
    request: Request,
    student_id: str,
    limit: int = 50,
    api_key: dict = Depends(validate_api_key),
):
    """
    Fetch attendance history for a specific student.
    Scoped to the requesting org. Hard cap at 100 records per call.
    """
    limit = min(limit, 100)

    result = (
        supabase_admin.table("attendance_records")
        .select("*")
        .eq("student_id", student_id)
        .eq("org_id", api_key["org_id"])
        .order("marked_at", desc=True)
        .limit(limit)
        .execute()
    )

    return {
        "student_id": student_id,
        "total":      len(result.data),
        "records":    result.data,
    }