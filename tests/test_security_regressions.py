import unittest
import importlib
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import BackgroundTasks, HTTPException

class _FakeUploadFile:
    def __init__(self, content: bytes, filename: str = "1.jpg", content_type: str = "image/jpeg"):
        self._content = content
        self.filename = filename
        self.content_type = content_type

    async def read(self):
        return self._content


class SecurityRegressionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = importlib.import_module("app.main")
        cls.audit_logs = importlib.import_module("app.routes.audit_logs")
        cls.email = importlib.import_module("app.utils.email")

    async def test_upload_student_forces_non_super_admin_institution(self):
        class FakeStorageBucket:
            def __init__(self):
                self.upload_path = None
                self.upload_content = None
                self.removed = []

            def remove(self, paths):
                self.removed.extend(paths)

            def upload(self, path, file, file_options=None):
                self.upload_path = path
                self.upload_content = file

            def get_public_url(self, path):
                return f"https://example.test/{path}"

        class FakeStorage:
            def __init__(self):
                self.bucket = FakeStorageBucket()

            def from_(self, _bucket_name):
                return self.bucket

        class FakeTable:
            def __init__(self, name, admin):
                self.name = name
                self.admin = admin
                self.payload = None
                self.filters = []

            def select(self, *_args, **_kwargs):
                return self

            def eq(self, key, value):
                self.filters.append((key, value))
                return self

            def single(self):
                return self

            def upsert(self, payload):
                self.payload = payload
                return self

            def execute(self):
                if self.name == "profiles":
                    return SimpleNamespace(
                        data={"institution_id": "INST_A", "is_super_admin": False, "role": "admin"}
                    )
                if self.name == "students":
                    self.admin.upsert_payload = self.payload
                    return SimpleNamespace(data=[self.payload])
                return SimpleNamespace(data=[])

        class FakeSupabaseAdmin:
            def __init__(self):
                self.storage = FakeStorage()
                self.upsert_payload = None

            def table(self, name):
                return FakeTable(name, self)

        fake_admin = FakeSupabaseAdmin()
        with patch.object(self.main, "supabase_admin", fake_admin):
            result = await self.main.upload_student(
                background_tasks=BackgroundTasks(),
                student_id="S1",
                name="Alice",
                institution_id="INST_B",
                file=_FakeUploadFile(b"img-bytes", filename="1.jpg"),
                authorization="Bearer token",
                user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(result["institution_id"], "INST_A")
        self.assertEqual(fake_admin.upsert_payload["institution_id"], "INST_A")
        self.assertTrue(result["file_path"].startswith("INST_A/"))

    async def test_list_students_ignores_requested_institution_for_non_super_admin(self):
        class FakeTable:
            def __init__(self, name, admin):
                self.name = name
                self.admin = admin

            def select(self, *_args, **_kwargs):
                return self

            def order(self, *_args, **_kwargs):
                return self

            def eq(self, key, value):
                if self.name == "students" and key == "institution_id":
                    self.admin.captured_institution_filter = value
                return self

            def single(self):
                return self

            def execute(self):
                if self.name == "profiles":
                    return SimpleNamespace(
                        data={"institution_id": "INST_A", "is_super_admin": False, "role": "admin"}
                    )
                if self.name == "students":
                    return SimpleNamespace(data=[{"id": "S1", "institution_id": "INST_A"}])
                return SimpleNamespace(data=[])

        class FakeSupabaseAdmin:
            def __init__(self):
                self.captured_institution_filter = None

            def table(self, name):
                return FakeTable(name, self)

        fake_admin = FakeSupabaseAdmin()
        with patch.object(self.main, "supabase_admin", fake_admin):
            response = await self.main.list_students(
                institution_id="INST_B",
                user=SimpleNamespace(id="user-1"),
            )

        self.assertEqual(fake_admin.captured_institution_filter, "INST_A")
        self.assertEqual(response["count"], 1)

    async def test_delete_student_blocks_cross_tenant_non_super_admin(self):
        class FakeTable:
            def __init__(self, name):
                self.name = name
                self.student_filter = None

            def select(self, *_args, **_kwargs):
                return self

            def eq(self, key, value):
                if self.name == "students" and key == "id":
                    self.student_filter = value
                return self

            def single(self):
                return self

            def limit(self, *_args, **_kwargs):
                return self

            def execute(self):
                if self.name == "profiles":
                    return SimpleNamespace(
                        data={"institution_id": "INST_A", "is_super_admin": False, "role": "admin"}
                    )
                if self.name == "students":
                    return SimpleNamespace(data=[{"institution_id": "INST_B"}])
                return SimpleNamespace(data=[])

        class FakeSupabaseAdmin:
            def table(self, name):
                return FakeTable(name)

        with patch.object(self.main, "supabase_admin", FakeSupabaseAdmin()):
            with self.assertRaises(HTTPException) as ctx:
                await self.main.delete_student("S1", user=SimpleNamespace(id="user-1"))

        self.assertEqual(ctx.exception.status_code, 403)

    async def test_audit_logs_treats_super_admin_without_is_admin_as_super(self):
        class FakeQuery:
            def __init__(self):
                self.filters = []

            def select(self, *_args, **_kwargs):
                return self

            def order(self, *_args, **_kwargs):
                return self

            def range(self, *_args, **_kwargs):
                return self

            def eq(self, key, value):
                self.filters.append((key, value))
                return self

            def execute(self):
                return SimpleNamespace(data=[], count=0)

        class FakeProfilesQuery:
            def select(self, *_args, **_kwargs):
                return self

            def eq(self, *_args, **_kwargs):
                return self

            def limit(self, *_args, **_kwargs):
                return self

            def execute(self):
                return SimpleNamespace(
                    data=[{"is_super_admin": True, "is_admin": False, "institution_id": None}]
                )

        class FakeSupabaseAdmin:
            def __init__(self):
                self.audit_query = FakeQuery()

            def table(self, name):
                if name == "profiles":
                    return FakeProfilesQuery()
                if name == "audit_logs":
                    return self.audit_query
                raise AssertionError(f"Unexpected table access: {name}")

        fake_admin = FakeSupabaseAdmin()
        with patch.object(self.audit_logs, "supabase_admin", fake_admin):
            await self.audit_logs.get_audit_logs(
                page=1,
                limit=10,
                action=None,
                institution_id="INST_X",
                start_date=None,
                end_date=None,
                current_user=SimpleNamespace(id="user-1"),
            )

        self.assertIn(("institution_id", "INST_X"), fake_admin.audit_query.filters)

    def test_send_email_no_password_returns_without_name_error(self):
        with patch.object(self.email, "SMTP_PASSWORD", ""), patch.object(self.email, "logger"):
            self.email._send_email(["admin@example.com"], "subject", "<p>body</p>")


if __name__ == "__main__":
    unittest.main()
