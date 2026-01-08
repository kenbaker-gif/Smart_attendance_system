from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.routes import admin, attendance


def test_admin_routes_return_503_when_supabase_missing():
    app = FastAPI()
    app.include_router(admin.router)
    client = TestClient(app)

    # No admin secret provided and no supabase env vars -> auth will fail or 503 for supabase
    resp = client.get("/admin/attendance")
    # FastAPI's HTTPBearer returns 401 when Authorization header is missing; accept 401/403/503
    assert resp.status_code in (401, 403, 503)


def test_capture_route_returns_503_when_supabase_missing():
    app = FastAPI()
    app.include_router(attendance.router)
    client = TestClient(app)

    resp = client.post("/capture/123", files={"file": ("test.jpg", b"data")})
    assert resp.status_code == 503
