import sqlite3
import json
from pathlib import Path
from datetime import datetime

DB_PATH = "pipeline.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                date                  TEXT NOT NULL,
                ran_at                TEXT NOT NULL,
                style_id              TEXT NOT NULL,
                style_name            TEXT NOT NULL,
                variation             TEXT NOT NULL,
                prompt_enhanced       TEXT,
                image_provider        TEXT,
                source_image_path     TEXT,
                sizes_generated       TEXT,
                etsy_title            TEXT,
                output_folder         TEXT,
                github_url            TEXT,
                generation_time_sec   REAL,
                estimated_cost_usd    REAL,
                status                TEXT DEFAULT 'pending',
                error_message         TEXT,
                created_at            TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS used_variations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                style_id     TEXT NOT NULL,
                variation    TEXT NOT NULL,
                used_on      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pipeline_state (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS etsy_performance (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         INTEGER REFERENCES runs(id),
                listing_id     TEXT,
                views          INTEGER DEFAULT 0,
                favourites     INTEGER DEFAULT 0,
                sales          INTEGER DEFAULT 0,
                revenue_usd    REAL DEFAULT 0,
                recorded_at    TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """)
    conn.close()


def insert_run(data: dict) -> int:
    conn = get_connection()
    with conn:
        cur = conn.execute(
            """INSERT INTO runs
               (date, ran_at, style_id, style_name, variation, status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (
                data["date"],
                datetime.utcnow().isoformat(),
                data["style_id"],
                data["style_name"],
                data["variation"],
            ),
        )
        return cur.lastrowid


def update_run(run_id: int, fields: dict):
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [run_id]
    conn = get_connection()
    with conn:
        conn.execute(f"UPDATE runs SET {sets} WHERE id = ?", values)
    conn.close()


def mark_variation_used(style_id: str, variation: str, date: str):
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO used_variations (style_id, variation, used_on) VALUES (?, ?, ?)",
            (style_id, variation, date),
        )
    conn.close()


def was_variation_used(style_id: str, variation: str) -> bool:
    conn = get_connection()
    row = conn.execute(
        "SELECT 1 FROM used_variations WHERE style_id = ? AND variation = ?",
        (style_id, variation),
    ).fetchone()
    conn.close()
    return row is not None
