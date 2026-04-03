"""
deps.py — Shared dependencies for FaceAttend API
Centralises: Supabase clients, rate limiter, and auth dependencies.

Both main.py and v1_api.py import from here to avoid circular imports
and to ensure there is exactly ONE Limiter instance registered on the app.
"""

import os
from fastapi import Header, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Supabase clients ─────────────────────────────────────────────────────────

SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_KEY         = os.getenv("SUPABASE_KEY")          # anon key — auth verification
SUPABASE_SERVICE_KEY = os.getenv("SERVICE_KEY")            # service role — DB/storage ops

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")
if not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SERVICE_KEY in environment.")

supabase: Client       = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Rate limiter ─────────────────────────────────────────────────────────────
# Single shared instance. main.py attaches this to app.state.limiter so
# slowapi can enforce limits on all routes — including /v1/ routes — through
# the one registered exception handler.
#
# Key function: use org_id from request.state when available (set by
# validate_api_key in v1_api.py), otherwise fall back to remote IP.
# This means enterprise clients behind NAT/proxies are bucketed by org,
# not by the shared egress IP.

def rate_limit_key(request: Request) -> str:
    return getattr(request.state, "org_id", None) or get_remote_address(request)

limiter = Limiter(key_func=rate_limit_key)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)

# ── Auth dependencies ────────────────────────────────────────────────────────

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
        resp = supabase_admin.table("profiles") \
            .select("is_admin, is_super_admin, role, institution_id") \
            .eq("id", user_id).limit(1).execute()

        profile_data = resp.data[0] if resp.data else None
        if not profile_data:
            raise HTTPException(status_code=403, detail="Admin profile not found or incomplete")

        is_admin       = _bool_flag(profile_data.get("is_admin"))
        is_super_admin = _bool_flag(profile_data.get("is_super_admin"))
        role           = profile_data.get("role", "")

        if not (is_admin or is_super_admin or role in ("admin", "super_admin")):
            raise HTTPException(status_code=403, detail="Admin access required")

        institution_id = profile_data.get("institution_id")
        if institution_id and not (is_super_admin or role == "super_admin"):
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
    except Exception as e:
        print(f"[check_admin] error: {e!r}")
        raise HTTPException(status_code=401, detail="Token verification failed")


async def check_super_admin(authorization: str = Header(None)):
    """Verify that the user is a super admin."""
    user = await check_admin(authorization)
    user_id = user.id
    resp = supabase_admin.table("profiles") \
        .select("is_super_admin, role") \
        .eq("id", user_id).limit(1).execute()
    profile_data   = resp.data[0] if resp.data else None
    is_super_admin = _bool_flag(profile_data.get("is_super_admin") if profile_data else None)
    role           = profile_data.get("role", "") if profile_data else ""

    if not (is_super_admin or role == "super_admin"):
        raise HTTPException(status_code=403, detail="Super admin access required")

    return user