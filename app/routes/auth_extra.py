"""
routes/auth_extra.py
Additional auth routes for FaceAttend:
  - POST /auth/forgot-password       → trigger Supabase password reset email
  - POST /auth/log-login             → Flutter fallback to log a login event
  - POST /webhooks/supabase-auth     → Supabase Auth webhook (primary login capture)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from pydantic import BaseModel, EmailStr

from app.dep import supabase, supabase_admin, verify_supabase_token, limiter
from app.utils.audit import AuditAction, log_event

logger = logging.getLogger(__name__)
router = APIRouter()

SUPABASE_WEBHOOK_SECRET = os.getenv("SUPABASE_WEBHOOK_SECRET", "")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class LogLoginRequest(BaseModel):
    device_model: Optional[str] = None
    os_version:   Optional[str] = None
    app_version:  Optional[str] = None
    source:       Optional[str] = None

    class Config:
        extra = "allow"  # ← ignore unknown fields


# ---------------------------------------------------------------------------
# POST /auth/forgot-password  (public — no JWT required)
# ---------------------------------------------------------------------------

@router.post("/auth/forgot-password")
@limiter.limit("5/hour")
async def forgot_password(request: Request):
    # parse body manually — limiter interferes with automatic binding
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid or missing JSON body")
    
    body = ForgotPasswordRequest(**payload)
    
    try:
        supabase.auth.reset_password_for_email(
            body.email,
            options={"redirect_to": "https://faceattend.app/reset-password"},
        )
    except Exception as exc:
        logger.error("[forgot-password] Supabase error: %s", exc)

    await log_event(
        AuditAction.AUTH_PASSWORD_RESET,
        actor_email=body.email,
        actor_id=None,
        institution_id=None,
        metadata={"note": "Password reset requested"},
        request=request,
    )

    return {"message": "If that email is registered, a reset link has been sent."}


# ---------------------------------------------------------------------------
# POST /auth/log-login  (JWT required — called by Flutter after signIn)
# ---------------------------------------------------------------------------

@router.post("/auth/log-login")
@limiter.limit("20/hour")
async def log_login(
    request: Request,
    current_user = Depends(verify_supabase_token),
):
    """
    Flutter/dashboard calls this after successful signIn.
    Fallback in case Supabase webhook missed the event.
    """
    try:
        raw = await request.json()
        body = LogLoginRequest(**raw)
    except Exception:
        body = LogLoginRequest()

    logger.info(f"[log-login] User: {current_user.id} | Source: {body.source}")

    actor_id    = current_user.id
    actor_email = current_user.email

    try:
        prof = supabase_admin.table("profiles").select("institution_id") \
            .eq("id", actor_id).limit(1).execute()
        institution_id = prof.data[0].get("institution_id") if prof.data else None
    except Exception as e:
        logger.error(f"[log-login] Profile lookup failed: {e}")
        institution_id = None

    try:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        recent = (
            supabase.table("audit_logs")
            .select("id")
            .eq("actor_id", actor_id)
            .eq("action", AuditAction.AUTH_LOGIN)
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
        if recent.data:
            logger.info("[log-login] Skipping duplicate for %s", actor_id)
            return {"message": "already logged"}
    except Exception as exc:
        logger.error("[log-login] Dedup check failed: %s", exc)

    await log_event(
        AuditAction.AUTH_LOGIN,
        actor_id=actor_id,
        actor_email=actor_email,
        institution_id=institution_id,
        metadata={
            "device_model": body.device_model,
            "os_version":   body.os_version,
            "app_version":  body.app_version,
            "source":       body.source,
        },
        request=request,
    )

    return {"message": "login logged"}

@router.post("/auth/log-logout")
async def log_logout(
    request: Request,
    current_user = Depends(verify_supabase_token),
):
    try:
        prof = supabase_admin.table("profiles").select("institution_id") \
            .eq("id", current_user.id).limit(1).execute()
        institution_id = prof.data[0].get("institution_id") if prof.data else None
    except Exception:
        institution_id = None

    await log_event(
        AuditAction.AUTH_LOGOUT,
        actor_id=current_user.id,
        actor_email=current_user.email,
        institution_id=institution_id,
        metadata={"source": request.headers.get("X-Source", "dashboard")},
        request=request,
    )
    return {"message": "logout logged"}


# ---------------------------------------------------------------------------
# POST /webhooks/supabase-auth  (called by Supabase Auth webhook)
# ---------------------------------------------------------------------------

def _verify_webhook_signature(payload_bytes: bytes, signature: Optional[str]) -> bool:
    """Verify Supabase webhook HMAC-SHA256 signature."""
    if not SUPABASE_WEBHOOK_SECRET:
        logger.warning("[webhook] SUPABASE_WEBHOOK_SECRET not set — skipping verification")
        return True  # allow in dev; set secret in prod
    if not signature:
        return False
    expected = hmac.new(
        SUPABASE_WEBHOOK_SECRET.encode(),
        payload_bytes,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature.removeprefix("sha256="))


@router.post("/webhooks/supabase-auth")
async def supabase_auth_webhook(
    request: Request,
    x_webhook_signature: Optional[str] = Header(None),
):
    """
    Supabase Auth webhook — receives events when users log in, sign up, etc.
    Configure in Supabase Dashboard → Auth → Hooks.

    Supported event types: LOGIN, SIGNUP, PASSWORD_RECOVERY, TOKEN_REFRESHED
    """
    raw_body = await request.body()

    if not _verify_webhook_signature(raw_body, x_webhook_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        import json
        payload = json.loads(raw_body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event_type = payload.get("event", payload.get("type", "")).upper()
    user_data  = payload.get("user", payload.get("record", {})) or {}

    user_id    = user_data.get("id")
    user_email = user_data.get("email")

    if not user_id or not user_email:
        logger.warning("[webhook] Missing user_id or email in payload")
        return {"ok": True}

    logger.info("[webhook] Auth event: %s for %s", event_type, user_email)

    if event_type not in ("LOGIN", "SIGNUP", "TOKEN_REFRESHED"):
        return {"ok": True}  # ignore other events

    # Look up profile for institution_id
    try:
        profile = (
            supabase.table("profiles")
            .select("id, institution_id")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        profile_data = profile.data[0] if profile.data else {}
        institution_id = profile_data.get("institution_id")
    except Exception as exc:
        logger.error("[webhook] Profile lookup failed: %s", exc)
        institution_id = None

    # Deduplicate — same 60s window as Flutter fallback
    try:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        recent = (
            supabase.table("audit_logs")
            .select("id")
            .eq("actor_id", user_id)
            .eq("action", AuditAction.AUTH_LOGIN)
            .gte("created_at", cutoff)
            .limit(1)
            .execute()
        )
        if recent.data:
            logger.info("[webhook] Skipping duplicate login log for %s", user_id)
            return {"ok": True}
    except Exception as exc:
        logger.error("[webhook] Webhook dedup check failed: %s", exc)

    await log_event(
        AuditAction.AUTH_LOGIN,
        actor_id=user_id,
        actor_email=user_email,
        institution_id=institution_id,
        metadata={"source": "supabase_webhook", "event_type": event_type},
        request=request,
    )

    return {"ok": True}
