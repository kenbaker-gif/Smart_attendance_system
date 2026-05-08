"""
routes/audit_logs.py
GET /audit-logs  — paginated audit log retrieval.
  - Coordinators/admins see their institution only (enforced by RLS + query filter)
  - Super admins see all, can filter by institution_id
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from db import supabase
from deps import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/audit-logs")
async def get_audit_logs(
    page:           int            = Query(1, ge=1),
    limit:          int            = Query(50, ge=1, le=200),
    action:         Optional[str]  = Query(None),          # filter by action type
    institution_id: Optional[str]  = Query(None),          # super admin only
    start_date:     Optional[str]  = Query(None),          # ISO date e.g. 2025-05-01
    end_date:       Optional[str]  = Query(None),          # ISO date e.g. 2025-05-08
    current_user:   dict           = Depends(get_current_user),
):
    is_super_admin   = current_user.get("is_super_admin") and current_user.get("is_admin")
    user_institution = current_user.get("institution_id")

    if not is_super_admin and not user_institution:
        raise HTTPException(status_code=403, detail="No institution associated with account")

    offset = (page - 1) * limit

    # Build query
    q = (
        supabase.table("audit_logs")
        .select("*", count="exact")
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    # Scope to institution
    if is_super_admin and institution_id:
        q = q.eq("institution_id", institution_id)
    elif not is_super_admin:
        q = q.eq("institution_id", user_institution)

    # Optional filters
    if action:
        q = q.eq("action", action)
    if start_date:
        q = q.gte("created_at", f"{start_date}T00:00:00Z")
    if end_date:
        q = q.lte("created_at", f"{end_date}T23:59:59Z")

    try:
        res = q.execute()
    except Exception as exc:
        logger.error("[audit-logs] Query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to fetch audit logs")

    return {
        "data":  res.data,
        "total": res.count,
        "page":  page,
        "limit": limit,
        "pages": -(-res.count // limit) if res.count else 0,  # ceiling division
    }


@router.get("/audit-logs/actions")
async def get_audit_actions(current_user: dict = Depends(get_current_user)):
    """Return distinct action types for dashboard filter dropdown."""
    try:
        res = supabase.table("audit_logs").select("action").execute()
        actions = sorted(set(r["action"] for r in res.data if r.get("action")))
        return {"actions": actions}
    except Exception as exc:
        logger.error("[audit-logs/actions] %s", exc)
        return {"actions": []}