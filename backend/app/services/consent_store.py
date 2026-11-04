"""
Follica Consent Store - SQLite-backed consent record persistence.

Schema (consent_records table):
  id              TEXT PRIMARY KEY  - random UUID v4
  created_at      TEXT              - ISO-8601 UTC timestamp
  terms_version   TEXT              - semver string of terms/privacy policy in effect
  client_token    TEXT              - salted SHA-256 of an anonymous client identifier;
                                     never contains raw IP, device ID, or any photo
  scope_analysis  INTEGER (0/1)     - consent to run the scalp analysis pipeline
  scope_progress  INTEGER (0/1)     - consent to optional progress storage
  retention_days  INTEGER           - user-selected retention window
  revoked_at      TEXT or NULL      - ISO-8601 UTC timestamp if consent was revoked

What is NOT stored: images, questionnaire answers, analysis results, or any PII.

Retention policy: records where revoked_at IS NOT NULL or
  (julianday('now') - julianday(created_at)) > retention_days
are purged by purge_expired().
"""

import hashlib
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional, Tuple, Generator

from app.core.config import settings

# Salt is loaded from an env var; falls back to a dev salt.
# In production set FOLLICA_CONSENT_SALT to a random 32-byte hex string.
_SALT = os.getenv("FOLLICA_CONSENT_SALT", "follica-dev-salt-not-for-production")

# Current terms/privacy version - bump when policy changes so old records are distinguished.
TERMS_VERSION = "1.0.0"


def _hash_client_token(raw_identifier: str) -> str:
    """
    Returns a salted SHA-256 hex digest of the raw identifier.
    The raw value (IP, name, device ID) is never stored.
    """
    salted = f"{_SALT}:{raw_identifier}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


class ConsentStore:
    """SQLite-backed store for consent records."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.CONSENT_DB_PATH
        self._init_db()

    # ------------------------------------------------------------------
    # Schema bootstrap & Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the DB file and table if they do not exist yet."""
        if self.db_path != ":memory:":
            dir_path = os.path.dirname(os.path.abspath(self.db_path))
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consent_records (
                    id              TEXT PRIMARY KEY,
                    created_at      TEXT NOT NULL,
                    terms_version   TEXT NOT NULL,
                    client_token    TEXT NOT NULL,
                    scope_analysis  INTEGER NOT NULL DEFAULT 1,
                    scope_progress  INTEGER NOT NULL DEFAULT 0,
                    retention_days  INTEGER NOT NULL DEFAULT 30,
                    revoked_at      TEXT
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD Operations
    # ------------------------------------------------------------------

    def create(
        self,
        raw_client_identifier: str,
        scope_analysis: bool = True,
        scope_progress: bool = False,
        retention_days: int = 30,
        terms_version: str = TERMS_VERSION,
    ) -> dict:
        """
        Persist a new consent record and return it as a dict.
        The raw_client_identifier is hashed with a salt before storage.
        """
        record_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        client_token = _hash_client_token(raw_client_identifier)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO consent_records
                    (id, created_at, terms_version, client_token,
                     scope_analysis, scope_progress, retention_days, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    record_id,
                    created_at,
                    terms_version,
                    client_token,
                    int(scope_analysis),
                    int(scope_progress),
                    retention_days,
                ),
            )
            conn.commit()

        return self.get(record_id)

    def get(self, record_id: str) -> Optional[dict]:
        """Return a single consent record by UUID, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM consent_records WHERE id = ?", (record_id,)
            ).fetchone()
        return dict(row) if row else None

    def revoke(self, record_id: str) -> Tuple[bool, Optional[dict]]:
        """
        Soft-delete: set revoked_at to UTC now.
        Returns (True, updated_record) on success, (False, None) if not found.
        """
        record = self.get(record_id)
        if record is None:
            return False, None
        if record.get("revoked_at") is not None:
            return True, record

        revoked_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE consent_records SET revoked_at = ? WHERE id = ?",
                (revoked_at, record_id),
            )
            conn.commit()
        return True, self.get(record_id)

    def purge_expired(self) -> int:
        """
        Hard-delete records that are either revoked or older than retention_days.
        Returns the number of rows deleted.
        (provisional / uncalibrated - retention period is user-selected, not regulatory advice)
        """
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM consent_records
                WHERE revoked_at IS NOT NULL
                   OR (julianday('now') - julianday(created_at)) > retention_days
                """
            )
            conn.commit()
            return cur.rowcount


consent_store = ConsentStore()
