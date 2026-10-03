"""SQLite checkpointing so re-runs skip completed work."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any

from .config import DB_PATH
from .logging_setup import get_logger

log = get_logger("store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS company (
    symbol           TEXT PRIMARY KEY,
    scrip_code       TEXT,
    status           TEXT,
    transcripts_found INTEGER DEFAULT 0,
    ann_count        INTEGER DEFAULT 0,
    updated_at       TEXT
);

CREATE TABLE IF NOT EXISTS transcript (
    symbol      TEXT,
    quarter     TEXT,
    news_dt     TEXT,
    attachment  TEXT,
    url_path    TEXT,
    pdf_path    TEXT,
    txt_path    TEXT,
    n_pages     INTEGER,
    n_chars     INTEGER,
    extract_method TEXT,
    kind        TEXT,
    failure     TEXT,
    updated_at  TEXT,
    PRIMARY KEY (symbol, quarter)
);

CREATE TABLE IF NOT EXISTS analysis (
    symbol   TEXT,
    quarter  TEXT,
    payload  TEXT,
    updated_at TEXT,
    PRIMARY KEY (symbol, quarter)
);

CREATE TABLE IF NOT EXISTS diff (
    symbol   TEXT PRIMARY KEY,
    payload  TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS runlog (
    symbol   TEXT,
    quarter  TEXT,
    stage    TEXT,
    detail   TEXT,
    ts       TEXT
);
"""


class Store:
    def __init__(self, path=DB_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # ------------------------------------------------------------- helpers
    def _write(self, sql: str, args: tuple = ()) -> None:
        with self._lock:
            self.conn.execute(sql, args)
            self.conn.commit()

    def _read(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self.conn.execute(sql, args))

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------ company
    def upsert_company(self, symbol: str, **fields: Any) -> None:
        fields["updated_at"] = self._now()
        cols = ", ".join(fields)
        ph = ", ".join("?" for _ in fields)
        upd = ", ".join(f"{k}=excluded.{k}" for k in fields)
        self._write(
            f"INSERT INTO company (symbol, {cols}) VALUES (?, {ph}) "
            f"ON CONFLICT(symbol) DO UPDATE SET {upd}",
            (symbol, *fields.values()),
        )

    def get_company(self, symbol: str) -> dict | None:
        rows = self._read("SELECT * FROM company WHERE symbol=?", (symbol,))
        return dict(rows[0]) if rows else None

    def all_companies(self) -> list[dict]:
        return [dict(r) for r in self._read("SELECT * FROM company")]

    # --------------------------------------------------------- transcripts
    def upsert_transcript(self, symbol: str, quarter: str, **f: Any) -> None:
        f["updated_at"] = self._now()
        cols = ", ".join(f)
        ph = ", ".join("?" for _ in f)
        upd = ", ".join(f"{k}=excluded.{k}" for k in f)
        self._write(
            f"INSERT INTO transcript (symbol, quarter, {cols}) VALUES (?, ?, {ph}) "
            f"ON CONFLICT(symbol, quarter) DO UPDATE SET {upd}",
            (symbol, quarter, *f.values()),
        )

    def get_transcripts(self, symbol: str) -> list[dict]:
        return [dict(r) for r in self._read(
            "SELECT * FROM transcript WHERE symbol=? ORDER BY news_dt DESC", (symbol,))]

    def all_transcripts(self) -> list[dict]:
        return [dict(r) for r in self._read(
            "SELECT * FROM transcript ORDER BY symbol, news_dt DESC")]

    # ------------------------------------------------------------ analysis
    def save_analysis(self, symbol: str, quarter: str, payload: dict) -> None:
        self._write(
            "INSERT INTO analysis (symbol, quarter, payload, updated_at) VALUES (?,?,?,?) "
            "ON CONFLICT(symbol, quarter) DO UPDATE SET payload=excluded.payload, "
            "updated_at=excluded.updated_at",
            (symbol, quarter, json.dumps(payload), self._now()),
        )

    def get_analysis(self, symbol: str, quarter: str) -> dict | None:
        rows = self._read(
            "SELECT payload FROM analysis WHERE symbol=? AND quarter=?", (symbol, quarter))
        return json.loads(rows[0]["payload"]) if rows else None

    def get_all_analysis(self, symbol: str) -> list[tuple[str, dict]]:
        rows = self._read(
            "SELECT quarter, payload FROM analysis WHERE symbol=?", (symbol,))
        return [(r["quarter"], json.loads(r["payload"])) for r in rows]

    def save_diff(self, symbol: str, payload: dict) -> None:
        self._write(
            "INSERT INTO diff (symbol, payload, updated_at) VALUES (?,?,?) "
            "ON CONFLICT(symbol) DO UPDATE SET payload=excluded.payload, "
            "updated_at=excluded.updated_at",
            (symbol, json.dumps(payload), self._now()),
        )

    def get_diff(self, symbol: str) -> dict | None:
        rows = self._read("SELECT payload FROM diff WHERE symbol=?", (symbol,))
        return json.loads(rows[0]["payload"]) if rows else None

    # -------------------------------------------------------------- runlog
    def clear_analysis(self) -> int:
        """Drop cached LLM output so it can be regenerated. Keeps PDFs/text."""
        with self._lock:
            n = self.conn.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
            n += self.conn.execute("SELECT COUNT(*) FROM diff").fetchone()[0]
            self.conn.execute("DELETE FROM analysis")
            self.conn.execute("DELETE FROM diff")
            self.conn.commit()
        return n

    def log_event(self, symbol: str, quarter: str, stage: str, detail: str) -> None:
        self._write(
            "INSERT INTO runlog (symbol, quarter, stage, detail, ts) VALUES (?,?,?,?,?)",
            (symbol, quarter or "", stage, detail, self._now()),
        )

    def runlog(self) -> list[dict]:
        return [dict(r) for r in self._read("SELECT * FROM runlog ORDER BY ts")]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
