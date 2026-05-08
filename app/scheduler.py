"""
scheduler.py
APScheduler daily digest job for FaceAttend.
Wire into FastAPI lifespan in main.py.

Sends:
  - Super admin (abubaker@faceattend.app): global digest of all institutions
  - Each institution admin: their institution's events from last 24h
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.dep import supabase, supabase_admin
import app.utils.email as email_util

logger = logging.getLogger(__name__)

SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL", "abubaker@faceattend.app")

# EAT = UTC+3  →  8AM EAT = 5AM UTC
DIGEST_HOUR_UTC   = 5
DIGEST_MINUTE_UTC = 0


async def _fetch_events_since(hours: int = 24, institution_id: str | None = None) -> list[dict]:
    """Fetch audit log events from the last N hours, optionally scoped to an institution."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        q = (
            supabase.table("audit_logs")
            .select("*")
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
            .limit(500)
        )
        if institution_id:
            q = q.eq("institution_id", institution_id)
        res = q.execute()
        return res.data or []
    except Exception as exc:
        logger.error("[digest] Failed to fetch events: %s", exc)
        return []


async def _fetch_institution_admins() -> list[dict]:
    """
    Fetch all institution admins (non-super-admin) who have email stored.
    Returns list of {institution_id, email, institution_name}.
    """
    try:
        res = (
            supabase.table("profiles")
            .select("id, institution_id, email, institutions(name)")
            .eq("is_admin", True)
            .eq("is_super_admin", False)
            .not_.is_("institution_id", "null")
            .execute()
        )
        admins = []
        for row in res.data or []:
            email = row.get("email")
            if not email:
                continue
            inst = row.get("institutions") or {}
            admins.append({
                "email":            email,
                "institution_id":   row.get("institution_id"),
                "institution_name": inst.get("name", "Your Institution"),
            })
        return admins
    except Exception as exc:
        logger.error("[digest] Failed to fetch admins: %s", exc)
        return []


async def send_daily_digests() -> None:
    """Main digest job — runs once daily at 8AM EAT."""
    logger.info("[digest] Starting daily digest job")
    date_label = datetime.now(timezone.utc).strftime("%B %d, %Y")

    # 1. Global digest → super admin
    all_events = await _fetch_events_since(hours=24)
    if all_events:
        await email_util.send_digest(
            recipients=[SUPER_ADMIN_EMAIL],
            events=all_events,
            institution_name="All Institutions",
            date_label=date_label,
        )
        logger.info("[digest] Global digest sent (%d events)", len(all_events))
    else:
        logger.info("[digest] No events in last 24h — skipping global digest")

    # 2. Per-institution digest → each admin
    admins = await _fetch_institution_admins()
    for admin in admins:
        events = await _fetch_events_since(
            hours=24,
            institution_id=admin["institution_id"],
        )
        if not events:
            continue
        await email_util.send_digest(
            recipients=[admin["email"]],
            events=events,
            institution_name=admin["institution_name"],
            date_label=date_label,
        )
        logger.info(
            "[digest] Sent to %s for %s (%d events)",
            admin["email"], admin["institution_name"], len(events),
        )

    logger.info("[digest] Daily digest job complete")


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the APScheduler instance."""
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        send_daily_digests,
        trigger=CronTrigger(hour=DIGEST_HOUR_UTC, minute=DIGEST_MINUTE_UTC, timezone="UTC"),
        id="daily_digest",
        name="Daily Audit Digest",
        replace_existing=True,
        misfire_grace_time=3600,  # allow up to 1hr late if server was down
    )
    return scheduler