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


# ── Shared limiter ───────────────────────────────────────────────────────────
# FIX 3: Import the single Limiter instance from main.py instead of creating
# a second one here.  Two separate Limiter objects — even with different
# key_funcs — both need to be registered on the FastAPI app via
# app.state.limiter.  Only the one in main.py is registered, so any
# @limiter.limit() decorators referencing a local instance were silently
# no-ops (or raised errors on some slowapi versions).
#
# The shared limiter in main.py uses get_remote_address as its key_func.
# For enterprise routes we want to rate-limit by org_id where available,
# so we set request.state.org_id in validate_api_key and the key management
# routes set request.state.org_id from the JWT profile — the limiter will
# pick it up automatically via rate_limit_key in main.py if you switch the
# key_func there.  For now, IP-based limiting is active and correct.
from dep import limiter  # ✅ single shared instance (see deps.py)

# ── Auth dependency from main app ────────────────────────────────────────────
# FIX 2: Import check_admin and supabase_admin so key-management routes
# can authenticate callers and resolve their institution (org_id).
# Previously these routes read org_id from request.state which was never
# populated, always returning 400.
from dep import supabase, supabase_admin, check_admin

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

def _get_org_id(user) -> str:
    """Fetch institution_id from profiles for the given authenticated user."""
    resp = supabase_admin.table("profiles") \
        .select("institution_id") \
        .eq("id", user.id) \
        .limit(1).execute()
    org_id = resp.data[0].get("institution_id") if resp.data else None
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
    """
    Validates the incoming API key against Supabase api_keys table.
    Returns the api_key row (includes org_id, plan, etc.) if valid.
    Also sets request.state.org_id for the rate limiter.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
        )

    if not x_api_key.startswith("fa_live_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format.",
        )

    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    result = (
        supabase.table("api_keys")
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

    # Expose org_id on request state so the rate limiter can use it
    request.state.org_id = key_row["org_id"]

    # Fire-and-forget: update last_used_at
    supabase.table("api_keys").update(
        {"last_used_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", key_row["id"]).execute()

    return key_row


# ── Schemas ──────────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    name: str = "Default Key"  # e.g. "Production", "Staging"


class MarkAttendanceRequest(BaseModel):
    session_id: str
    student_id: str
    confidence_score: float          # face recognition confidence 0.0–1.0
    anti_spoof_passed: bool
    marked_at: Optional[str] = None  # ISO timestamp; defaults to now() in DB


class AttendanceRecord(BaseModel):
    id: str
    session_id: str
    student_id: str
    confidence_score: float
    anti_spoof_passed: bool
    marked_at: str
    status: str  # present | late | absent


# ── Key Management Routes ────────────────────────────────────────────────────
# FIX 2: All three routes now use Depends(check_admin) for JWT auth and call
# _get_org_id() to resolve the institution.  Previously org_id was read from
# request.state which was never set, so every call returned 400.

@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: CreateKeyRequest,
    user=Depends(check_admin),  # ✅ JWT auth wired in
):
    """
    Generate a new API key for the authenticated institution.
    The raw key is returned ONCE — store it securely. Only the hash is saved.

    Auth: JWT (institution admin). Not protected by X-API-Key.
    """
    org_id = _get_org_id(user)  # ✅ resolved from JWT profile

    raw_key, key_hash = generate_api_key()
    key_suffix = raw_key[-4:]

    result = supabase_admin.table("api_keys").insert({
        "key_hash":   key_hash,
        "key_suffix": key_suffix,
        "org_id":     org_id,
        "name":       body.name,
        "plans":       "enterprise",
        "is_active":  True,
    }).execute()

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key.",
        )

    return {
        "api_key": raw_key,          # shown ONCE — not stored
        "key_id":  result.data[0]["id"],
        "name":    body.name,
        "warning": "Save this key now. It will not be shown again.",
    }


@router.get("/keys")
async def list_api_keys(
    user=Depends(check_admin),  # ✅ JWT auth wired in
):
    """
    List all API keys for the authenticated institution.
    key_hash is never returned — only metadata.

    Auth: JWT (institution admin).
    """
    org_id = _get_org_id(user)  # ✅ resolved from JWT profile

    result = (
        supabase_admin.table("api_keys")
        .select("id, name, plan, is_active, created_at, last_used_at, key_suffix")
        .eq("org_id", org_id)
        .order("created_at", desc=True)
        .execute()
    )

    return {"keys": result.data}


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: str,
    user=Depends(check_admin),  # ✅ JWT auth wired in
):
    """
    Revoke (deactivate) an API key by ID.
    Scoped to the requesting institution — cannot revoke another org's key.

    Auth: JWT (institution admin).
    """
    org_id = _get_org_id(user)  # ✅ resolved from JWT profile

    result = (
        supabase_admin.table("api_keys")
        .update({"is_active": False})
        .eq("id", key_id)
        .eq("org_id", org_id)   # prevent cross-org revocation
        .execute()
    )

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

    result = supabase.table("attendance_records").insert(payload).execute()

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
        supabase.table("attendance_records")
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
        supabase.table("attendance_records")
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