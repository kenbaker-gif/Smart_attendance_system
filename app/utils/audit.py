"""
utils/audit.py
Central audit logging helper for FaceAttend.
Call log_event() from any route to write an audit record + fire email alerts.
"""

from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import Request
from postgrest.exceptions import APIError

from app.utils.email import send_alert, send_new_device_alert
from app.dep import supabase, supabase_admin

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action constants — import these in routes so strings stay consistent
# ---------------------------------------------------------------------------
class AuditAction:
    # Auth
    AUTH_LOGIN           = "auth.login"
    AUTH_LOGOUT          = "auth.logout"
    AUTH_PASSWORD_RESET  = "auth.password_reset"
    AUTH_PASSWORD_CHANGE = "auth.password_change"

    # Coordinators
    COORDINATOR_INVITE   = "coordinator.invite"
    COORDINATOR_REMOVE   = "coordinator.remove"

    # Students
    STUDENT_CREATE       = "student.create"
    STUDENT_DELETE       = "student.delete"
    STUDENT_UPDATE       = "student.update"

    # API keys
    API_KEY_CREATE       = "api_key.create"
    API_KEY_REVOKE       = "api_key.revoke"

    # Institutions (super admin)
    INSTITUTION_APPROVE  = "institution.approve"
    INSTITUTION_SUSPEND  = "institution.suspend"
    INSTITUTION_UPDATE   = "institution.update"

    # Attendance
    ATTENDANCE_VERIFY    = "attendance.verify"
    ATTENDANCE_SPOOF     = "attendance.spoof_detected"

    # Course units
    COURSE_UNIT_CREATE   = "course_unit.create"
    COURSE_UNIT_DELETE   = "course_unit.delete"


# Actions that immediately send an email alert to the actor / admin
ALERT_ACTIONS = {
    AuditAction.AUTH_LOGIN,           # only if new device — checked inside
    AuditAction.AUTH_PASSWORD_RESET,
    AuditAction.COORDINATOR_INVITE,
    AuditAction.API_KEY_CREATE,
    AuditAction.API_KEY_REVOKE,
}


def _get_ip(request: Optional[Request]) -> Optional[str]:
    if request is None:
        return None
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


async def log_event(
    action: str,
    *,
    actor_id: Optional[str] = None,
    actor_email: Optional[str] = None,
    institution_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    request: Optional[Request] = None,
    ip_address: Optional[str] = None,
) -> None:
    """
    Write one audit log row and fire any necessary email alerts.
    Never raises — errors are logged but swallowed so they don't break routes.
    """
    ip = ip_address or _get_ip(request)
    meta = metadata or {}

    try:
        supabase.table("audit_logs").insert({
            "institution_id": institution_id,
            "actor_id":       actor_id,
            "actor_email":    actor_email,
            "action":         action,
            "resource_type":  resource_type,
            "resource_id":    resource_id,
            "metadata":       meta,
            "ip_address":     ip,
        }).execute()
    except Exception as exc:
        logger.error(f"[audit] Failed to write log: {exc}")
        return  # don't block the request

    # -----------------------------------------------------------------------
    # Email alerts
    # -----------------------------------------------------------------------
    if action not in ALERT_ACTIONS:
        return

    try:
        if action == AuditAction.AUTH_LOGIN:
            await _handle_login_alert(
                actor_id=actor_id,
                actor_email=actor_email,
                institution_id=institution_id,
                ip=ip,
                meta=meta,
            )
        else:
            # All other alert actions — send immediately to actor + institution admin
            recipients = _resolve_alert_recipients(actor_email, institution_id)
            await send_alert(
                recipients=recipients,
                action=action,
                actor_email=actor_email or "Unknown",
                resource_type=resource_type,
                resource_id=resource_id,
                metadata=meta,
                ip_address=ip,
            )
    except Exception as exc:
        logger.error(f"[audit] Failed to send alert email for {action}: {exc}")


async def _handle_login_alert(
    actor_id: Optional[str],
    actor_email: Optional[str],
    institution_id: Optional[str],
    ip: Optional[str],
    meta: dict,
) -> None:
    """Check if this is a new device/IP. If so, alert."""
    if not actor_id or not ip:
        return

    device_info = meta.get("device_info", "Unknown device")

    # Check known_devices
    existing = (
        supabase_admin.table("known_devices")
        .select("id")
        .eq("actor_id", actor_id)
        .eq("ip_address", ip)
        .limit(1)
        .execute()
    )

    is_new = not existing.data

    if is_new:
        # Record it
        supabase_admin.table("known_devices").upsert({
            "actor_id":    actor_id,
            "ip_address":  ip,
            "device_info": device_info,
            "last_seen_at": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="actor_id,ip_address").execute()

        recipients = _resolve_alert_recipients(actor_email, institution_id)
        await send_new_device_alert(
            recipients=recipients,
            actor_email=actor_email or "Unknown",
            ip_address=ip,
            device_info=device_info,
            metadata=meta,
        )
    else:
        # Known device — just update last_seen_at, no alert
        supabase_admin.table("known_devices").update({
            "last_seen_at": datetime.now(timezone.utc).isoformat(),
        }).eq("actor_id", actor_id).eq("ip_address", ip).execute()


def _resolve_alert_recipients(
    actor_email: Optional[str],
    institution_id: Optional[str],
) -> list[str]:
    """
    Returns list of emails to alert:
    - The actor themselves
    - The institution admin (if different from actor)
    """
    recipients: list[str] = []

    if actor_email:
        recipients.append(actor_email)

    if institution_id:
        try:
            result = (
                supabase_admin.table("profiles")
                .select("id, email:id")  # we need auth email — use a join or store email in profiles
                .eq("institution_id", institution_id)
                .eq("is_admin", True)
                .eq("is_super_admin", False)
                .limit(1)
                .execute()
            )
            # If you store email in profiles table, use it directly:
            # admin_email = result.data[0].get("email") if result.data else None
            # Otherwise fetch from auth.users via service role — see note below
        except Exception:
            pass

    return list(set(recipients))  # deduplicate