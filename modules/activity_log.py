"""Local SQLite audit log for authentication and application activity."""
from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ActivityLog:
    """Persist small, privacy-conscious user activity events."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / ".streamlit" / "activity_log.sqlite3"
        configured_path = database_path or os.environ.get("CSV_ANALYSER_ACTIVITY_DB")
        self.path = Path(configured_path) if configured_path else default_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_email TEXT,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at_utc TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_email_event "
                "ON activity_events(user_email, action, event_id DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_activity_event_id "
                "ON activity_events(event_id DESC)"
            )

    @staticmethod
    def _normalize_email(email: str | None) -> str | None:
        return email.strip().lower() if email and email.strip() else None

    def record(
        self,
        user_email: str | None,
        action: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Store an event; never pass secrets, prompts, or row-level data in details."""
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO activity_events
                        (user_email, action, details_json, created_at_utc)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        self._normalize_email(user_email),
                        action,
                        json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    ),
                )
        except sqlite3.Error:
            # Audit storage must not take down a user's analysis session.
            return

    def latest_login(self, user_email: str) -> str | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT created_at_utc FROM activity_events
                    WHERE user_email = ? AND action = 'login_success'
                    ORDER BY event_id DESC LIMIT 1
                    """,
                    (self._normalize_email(user_email),),
                ).fetchone()
            return row["created_at_utc"] if row else None
        except sqlite3.Error:
            return None

    def recent(self, user_email: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        try:
            with self._connect() as connection:
                if user_email is None:
                    rows = connection.execute(
                        """
                        SELECT user_email, action, details_json, created_at_utc
                        FROM activity_events ORDER BY event_id DESC LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """
                        SELECT user_email, action, details_json, created_at_utc
                        FROM activity_events WHERE user_email = ?
                        ORDER BY event_id DESC LIMIT ?
                        """,
                        (self._normalize_email(user_email), limit),
                    ).fetchall()
            return [
                {
                    "user": row["user_email"],
                    "action": row["action"],
                    "details": json.loads(row["details_json"]),
                    "timestamp": row["created_at_utc"],
                }
                for row in rows
            ]
        except (sqlite3.Error, json.JSONDecodeError):
            return []

    def export_csv(self, user_email: str | None = None, limit: int = 500) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["user", "action", "details", "timestamp_utc"])
        for event in self.recent(user_email, limit):
            writer.writerow(
                [
                    event["user"],
                    event["action"],
                    json.dumps(event["details"], ensure_ascii=False, sort_keys=True),
                    event["timestamp"],
                ]
            )
        return output.getvalue()

