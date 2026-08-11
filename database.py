import json
import os
import sqlite3
from contextlib import contextmanager

import config


@contextmanager
def _get_connection():
    os.makedirs(os.path.dirname(config.SQLITE_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _get_connection() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                filename     TEXT UNIQUE NOT NULL,
                ingested_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                chunk_count  INTEGER DEFAULT 0,
                status       TEXT DEFAULT 'ready',
                summary      TEXT
            )
            """
        )
        existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(documents)")}
        if "summary" not in existing_columns:
            conn.execute("ALTER TABLE documents ADD COLUMN summary TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id       TEXT NOT NULL,
                role             TEXT NOT NULL,
                raw_query        TEXT,
                rewritten_query  TEXT,
                content          TEXT NOT NULL,
                intent           TEXT,
                chunks_used      INTEGER DEFAULT 0,
                sources          TEXT,
                latency_ms       INTEGER,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def register_document(filename, chunk_count, status="ready"):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO documents (filename, chunk_count, status)
            VALUES (?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                chunk_count = excluded.chunk_count,
                status = excluded.status
            """,
            (filename, chunk_count, status),
        )


def update_document_summary(filename, summary):
    with _get_connection() as conn:
        conn.execute(
            "UPDATE documents SET summary = ? WHERE filename = ?",
            (summary, filename),
        )


def get_document_summary(filename):
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT summary FROM documents WHERE filename = ?", (filename,)
        ).fetchone()
        return row["summary"] if row else None


def get_registered_documents():
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT filename FROM documents WHERE status = 'ready'"
        ).fetchall()
        return [row["filename"] for row in rows]


def delete_document(filename):
    with _get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))


def add_message(
    session_id,
    role,
    raw_query,
    rewritten_query,
    content,
    intent,
    chunks_used,
    sources,
    latency_ms,
):
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (
                session_id, role, raw_query, rewritten_query, content,
                intent, chunks_used, sources, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                raw_query,
                rewritten_query,
                content,
                intent,
                chunks_used,
                json.dumps(sources) if sources is not None else None,
                latency_ms,
            ),
        )


def get_last_message(session_id):
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT COALESCE(rewritten_query, raw_query) AS query
            FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        return row["query"] if row else None


def get_last_n_messages(session_id, n=4):
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, n),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]


def get_all_messages(session_id):
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            """,
            (session_id,),
        ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]


def get_last_assistant_message(session_id):
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT content, intent, latency_ms, chunks_used, sources
            FROM messages
            WHERE session_id = ? AND role = 'assistant'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "content": row["content"],
            "intent": row["intent"],
            "latency_ms": row["latency_ms"],
            "chunks_used": row["chunks_used"],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
        }
