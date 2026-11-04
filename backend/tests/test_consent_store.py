"""
Tests for Consent Store, SQLite Persistence, Revocation, and Data Minimization.

Verifies:
1. Consent creation, read-back, and validation via API & Service
2. Revocation endpoint (DELETE /api/v1/consent/{id}) and idempotency
3. Auto-creation of SQLite database when file or directory is missing
4. Data minimization: strictly NO images, questionnaire answers, or raw PII in SQLite
5. Automatic purge of revoked and expired records
6. Absence of false AES-256 security claims in README and code
"""

import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.consent_store import ConsentStore, consent_store, TERMS_VERSION, _hash_client_token
from app.core.config import settings

client = TestClient(app)


def test_consent_create_and_read_back_api():
    """Verify consent creation returns valid metadata and can be read back."""
    user_id = f"test-anon-user-{uuid.uuid4().hex[:8]}"
    payload = {
        "user_id": user_id,
        "biometric_image_consent": True,
        "research_opt_in": True,
        "data_retention_days": 60,
    }

    # 1. POST /api/v1/consent
    response = client.post("/api/v1/consent", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["consent_recorded"] is True
    assert data["retention_days"] == 60
    assert data["terms_version"] == TERMS_VERSION
    assert "consent_id" in data
    consent_id = data["consent_id"]

    # Verify UUID format
    parsed_uuid = uuid.UUID(consent_id)
    assert str(parsed_uuid) == consent_id

    # Verify AES-256 claim was removed, SQLite documented
    assert "AES-256" not in data.get("storage_security", "")
    assert "SQLite" in data.get("storage_security", "")

    # 2. Read-back via GET /api/v1/consent/{consent_id}
    read_resp = client.get(f"/api/v1/consent/{consent_id}")
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert read_data["status"] == "success"
    record = read_data["consent"]
    assert record["id"] == consent_id
    assert record["terms_version"] == TERMS_VERSION
    assert record["retention_days"] == 60
    assert record["scope_analysis"] == 1
    assert record["scope_progress"] == 1
    assert record["revoked_at"] is None

    # Verify the client token in the DB is a salted SHA-256 hash, not the raw user_id
    assert record["client_token"] != user_id
    assert len(record["client_token"]) == 64
    assert record["client_token"] == _hash_client_token(user_id)


def test_consent_biometric_refusal_raises_400():
    """Without explicit biometric consent, registration is rejected."""
    payload = {
        "user_id": "anon-no-consent",
        "biometric_image_consent": False,
        "data_retention_days": 30,
    }
    response = client.post("/api/v1/consent", json=payload)
    assert response.status_code == 400
    assert "Biometric image consent is required" in response.text


def test_consent_revocation_endpoint():
    """Verify consent can be revoked via DELETE /api/v1/consent/{id}."""
    # Create record
    res = client.post("/api/v1/consent", json={
        "user_id": "user-to-revoke",
        "biometric_image_consent": True,
        "data_retention_days": 30,
    })
    consent_id = res.json()["consent_id"]

    # Revoke
    del_resp = client.delete(f"/api/v1/consent/{consent_id}")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["status"] == "revoked"
    assert del_data["consent_id"] == consent_id
    assert del_data["revoked_at"] is not None

    # Read back to confirm revoked_at is populated
    read_resp = client.get(f"/api/v1/consent/{consent_id}")
    assert read_resp.status_code == 200
    assert read_resp.json()["consent"]["revoked_at"] is not None

    # Non-existent ID returns 404
    missing_resp = client.delete(f"/api/v1/consent/non-existent-{uuid.uuid4().hex}")
    assert missing_resp.status_code == 404


def test_no_image_or_questionnaire_data_persisted_in_db():
    """
    CRITICAL ACCEPTANCE CHECK:
    Ensure that NO image bytes, photo filenames, questionnaire answers,
    or medical findings are ever stored in the SQLite consent store.
    """
    conn = sqlite3.connect(consent_store.db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("PRAGMA table_info(consent_records)")
        columns = [row["name"] for row in cursor.fetchall()]
    finally:
        conn.close()

    expected_columns = {
        "id",
        "created_at",
        "terms_version",
        "client_token",
        "scope_analysis",
        "scope_progress",
        "retention_days",
        "revoked_at",
    }
    # Check that columns in DB strictly match expected metadata columns
    assert set(columns) == expected_columns, f"Unexpected columns in consent_records: {columns}"

    # Specifically assert forbidden clinical / image fields do not exist
    forbidden_terms = [
        "image", "photo", "bytes", "file", "questionnaire", "symptom", "pull_test",
        "itch", "sebum", "flake", "diagnosis", "prescription", "score", "ip_address"
    ]
    for col in columns:
        for forbidden in forbidden_terms:
            assert forbidden not in col.lower(), f"Forbidden field '{col}' found in consent database schema!"


def test_consent_store_auto_creates_missing_db():
    """Verify store auto-creates parent directory and SQLite DB when absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        target_db = os.path.join(tmpdir, "nested", "storage", "consent_test.db")
        assert not os.path.exists(target_db)

        # Instantiate store pointing to non-existent path
        custom_store = ConsentStore(db_path=target_db)
        assert os.path.exists(target_db)

        # Verify operational
        rec = custom_store.create(raw_client_identifier="test-device-42", retention_days=14)
        assert rec["id"] is not None
        assert custom_store.get(rec["id"]) is not None


def test_consent_purge_expired_and_revoked():
    """Verify purge_expired() removes revoked and retention-exceeded rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "purge_test.db")
        store = ConsentStore(db_path=test_db)

        # 1. Active record (retention 30 days)
        active = store.create("user-active", retention_days=30)

        # 2. Revoked record
        revoked = store.create("user-revoked", retention_days=30)
        store.revoke(revoked["id"])

        # 3. Artificially expired record (inserted directly with created_at 40 days ago)
        expired_id = str(uuid.uuid4())
        conn = sqlite3.connect(test_db)
        try:
            conn.execute(
                """
                INSERT INTO consent_records
                (id, created_at, terms_version, client_token, scope_analysis, scope_progress, retention_days, revoked_at)
                VALUES (?, datetime('now', '-40 days'), '1.0.0', 'hash-expired', 1, 0, 30, NULL)
                """,
                (expired_id,)
            )
            conn.commit()
        finally:
            conn.close()

        # Purge
        deleted_count = store.purge_expired()
        assert deleted_count == 2, f"Expected 2 purged records (1 revoked + 1 expired), got {deleted_count}"

        # Active record remains
        assert store.get(active["id"]) is not None
        # Revoked and expired are gone
        assert store.get(revoked["id"]) is None
        assert store.get(expired_id) is None


def test_readme_and_privacy_docs_match_behaviour():
    """Verify README does not claim unimplemented AES-256 and privacy doc exists."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 1. Verify docs/privacy.md exists and covers DPDP Act
    privacy_md_path = os.path.join(repo_root, "docs", "privacy.md")
    assert os.path.exists(privacy_md_path), "docs/privacy.md must exist"
    with open(privacy_md_path, "r", encoding="utf-8") as f:
        privacy_text = f.read()
    assert "DPDP" in privacy_text, "docs/privacy.md must reference India's DPDP Act"
    assert "retention" in privacy_text.lower()
    assert "client_token" in privacy_text

    # 2. Check README.md for unimplemented AES claims
    readme_path = os.path.join(repo_root, "README.md")
    assert os.path.exists(readme_path)
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # The SQL schema file row in README table should not claim "AES-256 encrypted bucket"
    assert "SQL schema for AES-256 encrypted bucket" not in readme_text, (
        "README contains outdated claim of implemented AES-256 storage bucket"
    )
