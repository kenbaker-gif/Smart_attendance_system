"""
utils/email.py
Zoho Mail API email sender for FaceAttend audit alerts and daily digest.
"""

from __future__ import annotations

import os
import logging
import httpx
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FROM_NAME          = "FaceAttend Security"
FROM_EMAIL         = os.getenv("ZOHO_SMTP_USER", "abubaker@faceattend.app")

ZOHO_API_BASE      = "https://mail.zoho.com/api"
ZOHO_OAUTH_BASE    = "https://accounts.zoho.com/oauth/v2"
ZOHO_ACCOUNT_ID    = os.getenv("ZOHO_ACCOUNT_ID")
ZOHO_CLIENT_ID     = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")


# ---------------------------------------------------------------------------
# Zoho OAuth + core sender
# ---------------------------------------------------------------------------
async def _get_access_token() -> str:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            f"{ZOHO_OAUTH_BASE}/token",
            params={
                "refresh_token": ZOHO_REFRESH_TOKEN,
                "client_id":     ZOHO_CLIENT_ID,
                "client_secret": ZOHO_CLIENT_SECRET,
                "grant_type":    "refresh_token",
            },
        )
        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Zoho token refresh failed: {data}")
        return token


async def _send_email(to: list[str], subject: str, html_body: str) -> None:
    if not to:
        return
    if not all([ZOHO_ACCOUNT_ID, ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN]):
        logger.warning("[email] Zoho API credentials not fully set — skipping email")
        return

    try:
        token = await _get_access_token()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{ZOHO_API_BASE}/accounts/{ZOHO_ACCOUNT_ID}/messages",
                headers={
                    "Authorization": f"Zoho-oauthtoken {token}",
                    "Content-Type":  "application/json",
                },
                json={
                    "fromAddress": FROM_EMAIL,
                    "toAddress":   ", ".join(to),
                    "subject":     subject,
                    "content":     html_body,
                    "mailFormat":  "html",
                },
            )
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Zoho send failed ({resp.status_code}): {resp.text}")
        logger.info(f"[email] Sent '{subject}' to {to}")
    except Exception as exc:
        logger.error(f"[email] Zoho API error: {exc}")
        raise


# ---------------------------------------------------------------------------
# HTML template helpers
# ---------------------------------------------------------------------------
ACTION_LABELS = {
    "auth.login":           ("🔐", "New Login Detected",         "#3B82F6"),
    "auth.password_reset":  ("🔑", "Password Reset Requested",   "#F59E0B"),
    "auth.password_change": ("🔑", "Password Changed",           "#F59E0B"),
    "coordinator.invite":   ("👤", "Coordinator Invited",         "#8B5CF6"),
    "coordinator.remove":   ("👤", "Coordinator Removed",         "#EF4444"),
    "api_key.create":       ("🗝️",  "API Key Created",            "#10B981"),
    "api_key.revoke":       ("🗝️",  "API Key Revoked",            "#EF4444"),
    "institution.approve":  ("🏛️",  "Institution Approved",       "#10B981"),
    "institution.suspend":  ("🏛️",  "Institution Suspended",      "#EF4444"),
    "student.create":       ("🎓", "Student Registered",          "#10B981"),
    "student.delete":       ("🎓", "Student Deleted",             "#EF4444"),
    "attendance.verify":    ("✅", "Attendance Verified",          "#10B981"),
    "attendance.spoof_detected": ("⚠️", "Spoof Attempt Detected", "#EF4444"),
}

def _base_template(title: str, content: str, color: str = "#6366F1") -> str:
    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background:#F3F4F6;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#F3F4F6;padding:32px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <tr>
          <td style="background:{color};padding:24px 32px;">
            <table width="100%"><tr>
              <td>
                <span style="color:#ffffff;font-size:20px;font-weight:700;letter-spacing:-0.5px;">FaceAttend</span>
                <span style="color:rgba(255,255,255,0.7);font-size:14px;margin-left:8px;">Security Alert</span>
              </td>
              <td align="right">
                <span style="color:rgba(255,255,255,0.9);font-size:13px;">
                  {datetime.now(timezone.utc).strftime("%b %d, %Y %H:%M UTC")}
                </span>
              </td>
            </tr></table>
          </td>
        </tr>
        <tr>
          <td style="padding:28px 32px 0;">
            <h1 style="margin:0;font-size:22px;font-weight:700;color:#111827;">{title}</h1>
          </td>
        </tr>
        <tr>
          <td style="padding:20px 32px 32px;">
            {content}
          </td>
        </tr>
        <tr>
          <td style="background:#F9FAFB;border-top:1px solid #E5E7EB;padding:20px 32px;">
            <p style="margin:0;font-size:12px;color:#9CA3AF;">
              This is an automated security notification from FaceAttend.<br>
              If you did not perform this action, contact <a href="mailto:abubaker@faceattend.app" style="color:#6366F1;">abubaker@faceattend.app</a> immediately.
            </p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>
"""

def _detail_row(label: str, value: str) -> str:
    return f"""
    <tr>
      <td style="padding:8px 0;color:#6B7280;font-size:14px;width:140px;vertical-align:top;">{label}</td>
      <td style="padding:8px 0;color:#111827;font-size:14px;font-weight:500;">{value or "—"}</td>
    </tr>"""

def _detail_table(rows: list[tuple[str, str]]) -> str:
    inner = "".join(_detail_row(l, v) for l, v in rows)
    return f'<table width="100%" cellpadding="0" cellspacing="0" style="background:#F9FAFB;border-radius:8px;padding:16px 20px;margin-top:16px;">{inner}</table>'


# ---------------------------------------------------------------------------
# Public alert senders
# ---------------------------------------------------------------------------
async def send_alert(
    recipients: list[str],
    action: str,
    actor_email: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    ip_address: Optional[str] = None,
) -> None:
    meta = metadata or {}
    icon, label, color = ACTION_LABELS.get(action, ("📋", action.replace(".", " ").title(), "#6366F1"))

    rows = [
        ("Action",       f"{icon} {label}"),
        ("Performed by", actor_email),
        ("IP Address",   ip_address or "Unknown"),
        ("Time",         datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
    ]
    if resource_type:
        rows.append(("Resource", f"{resource_type}: {resource_id or ''}"))
    if meta:
        for k, v in meta.items():
            if k not in ("device_info",):
                rows.append((k.replace("_", " ").title(), str(v)))

    content = f"""
    <p style="color:#374151;font-size:15px;line-height:1.6;margin:0 0 8px;">
      A security-relevant action was performed on your FaceAttend account.
    </p>
    {_detail_table(rows)}
    <p style="margin-top:20px;color:#6B7280;font-size:13px;">
      If this was you, no action is needed. Otherwise, secure your account immediately.
    </p>
    """

    await _send_email(recipients, f"[FaceAttend] {icon} {label}", _base_template(label, content, color))


async def send_new_device_alert(
    recipients: list[str],
    actor_email: str,
    ip_address: str,
    device_info: str,
    metadata: Optional[dict] = None,
) -> None:
    meta = metadata or {}

    rows = [
        ("Account",    actor_email),
        ("IP Address", ip_address),
        ("Device",     device_info),
        ("Time",       datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("Location",   meta.get("location", "Unknown")),
    ]

    content = f"""
    <p style="color:#374151;font-size:15px;line-height:1.6;margin:0 0 8px;">
      A login was detected from a <strong>new device or IP address</strong> that we haven't seen before.
    </p>
    {_detail_table(rows)}
    <div style="margin-top:20px;padding:14px 18px;background:#FEF3C7;border-left:4px solid #F59E0B;border-radius:4px;">
      <p style="margin:0;font-size:14px;color:#92400E;">
        <strong>Not you?</strong> Change your password immediately and contact support.
      </p>
    </div>
    """

    await _send_email(
        recipients,
        "[FaceAttend] 🔐 New device login detected",
        _base_template("New Login from Unknown Device", content, "#F59E0B"),
    )


async def send_digest(
    recipients: list[str],
    events: list[dict],
    institution_name: str = "All Institutions",
    date_label: str = "",
) -> None:
    if not events:
        return

    date_label = date_label or datetime.now(timezone.utc).strftime("%B %d, %Y")

    rows_html = ""
    for ev in events[:100]:
        icon, label, color = ACTION_LABELS.get(ev.get("action", ""), ("📋", ev.get("action", ""), "#6B7280"))
        ts = ev.get("created_at", "")
        if ts:
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%H:%M UTC")
            except Exception:
                pass

        rows_html += f"""
        <tr>
          <td style="padding:10px 12px;font-size:13px;color:#374151;border-bottom:1px solid #F3F4F6;">
            <span style="background:{color}20;color:{color};padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;">{icon} {label}</span>
          </td>
          <td style="padding:10px 12px;font-size:13px;color:#6B7280;border-bottom:1px solid #F3F4F6;">{ev.get("actor_email") or "—"}</td>
          <td style="padding:10px 12px;font-size:13px;color:#6B7280;border-bottom:1px solid #F3F4F6;">{ev.get("ip_address") or "—"}</td>
          <td style="padding:10px 12px;font-size:13px;color:#9CA3AF;border-bottom:1px solid #F3F4F6;">{ts}</td>
        </tr>"""

    total = len(events)
    summary_counts: dict[str, int] = {}
    for ev in events:
        a = ev.get("action", "other")
        summary_counts[a] = summary_counts.get(a, 0) + 1

    summary_rows = "".join(
        f'<tr><td style="padding:4px 0;font-size:13px;color:#374151;">{ACTION_LABELS.get(a, ("","",))[1] or a}</td>'
        f'<td style="padding:4px 0;font-size:13px;color:#6B7280;text-align:right;font-weight:600;">{c}</td></tr>'
        for a, c in sorted(summary_counts.items(), key=lambda x: -x[1])
    )

    content = f"""
    <p style="color:#374151;font-size:15px;margin:0 0 20px;">
      Here is a summary of <strong>{total} audit event{"s" if total != 1 else ""}</strong>
      for <strong>{institution_name}</strong> on {date_label}.
    </p>
    <table width="100%" cellpadding="0" cellspacing="0" style="background:#F9FAFB;border-radius:8px;padding:16px 20px;margin-bottom:24px;">
      <tr>
        <td style="font-size:12px;font-weight:700;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;padding-bottom:8px;">Event Type</td>
        <td style="font-size:12px;font-weight:700;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.05em;padding-bottom:8px;text-align:right;">Count</td>
      </tr>
      {summary_rows}
    </table>
    <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #E5E7EB;border-radius:8px;overflow:hidden;">
      <thead>
        <tr style="background:#F9FAFB;">
          <th style="padding:10px 12px;font-size:11px;font-weight:700;color:#9CA3AF;text-align:left;text-transform:uppercase;">Action</th>
          <th style="padding:10px 12px;font-size:11px;font-weight:700;color:#9CA3AF;text-align:left;text-transform:uppercase;">Actor</th>
          <th style="padding:10px 12px;font-size:11px;font-weight:700;color:#9CA3AF;text-align:left;text-transform:uppercase;">IP</th>
          <th style="padding:10px 12px;font-size:11px;font-weight:700;color:#9CA3AF;text-align:left;text-transform:uppercase;">Time</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    {"<p style='margin-top:12px;font-size:12px;color:#9CA3AF;'>Showing first 100 events. Log into your dashboard for the full audit log.</p>" if total > 100 else ""}
    """

    await _send_email(
        recipients,
        f"[FaceAttend] Daily Audit Digest — {date_label}",
        _base_template(f"Daily Audit Digest: {institution_name}", content, "#6366F1"),
    )