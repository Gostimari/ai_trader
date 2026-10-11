"""SQLite storage layer: settings, market snapshots, decisions, cycles, news cache."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = os.environ.get(
    "MEMICOIN_DB", str(Path.home() / ".local" / "share" / "memicoin_trader" / "memicoin.db")
)

_lock = threading.Lock()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


class Database:
    def __init__(self, path: str = DEFAULT_DB_PATH):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()
        # lightweight migration: demo_wallet.cost_basis_usd (added later
        # than the original schema; runs after tables exist)
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(demo_wallet)")]
        if "cost_basis_usd" not in cols:
            self.conn.execute("ALTER TABLE demo_wallet ADD COLUMN cost_basis_usd REAL DEFAULT 0")

    def _migrate(self) -> None:
        cur = self.conn
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cycles (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                ai_input TEXT,
                ai_output TEXT,
                analysis_summary TEXT,
                confidence TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                cycle_id TEXT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                asset_in TEXT,
                asset_out TEXT,
                amount REAL,
                value_usd REAL,
                pnl_per_trade REAL,
                status TEXT NOT NULL,
                reason TEXT,
                tx_id TEXT
            );
            CREATE TABLE IF NOT EXISTS positions (
                asset TEXT PRIMARY KEY,
                chain TEXT,
                quantity REAL NOT NULL,
                cost_basis_usd REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cash (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                usd REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS demo_wallet (
                id TEXT PRIMARY KEY,
                chain TEXT NOT NULL,
                asset TEXT NOT NULL,
                contract TEXT DEFAULT '',
                balance REAL NOT NULL DEFAULT 0,
                is_gas INTEGER NOT NULL DEFAULT 0,
                cost_basis_usd REAL DEFAULT 0,
                UNIQUE(chain, asset, contract)
            );
            """
        )
        self.conn.commit()

    # ---------- settings ----------
    def get_setting(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with _lock:
            self.conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value), utcnow()),
            )
            self.conn.commit()

    # ---------- market snapshots ----------
    def save_market_snapshot(self, source: str, payload: Any) -> str:
        snap_id = new_id()
        with _lock:
            self.conn.execute(
                "INSERT INTO market_snapshots (id, timestamp, source, payload) VALUES (?,?,?,?)",
                (snap_id, utcnow(), source, json.dumps(payload)),
            )
            self.conn.commit()
        return snap_id

    def latest_market_snapshot(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM market_snapshots ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    # ---------- cycles ----------
    def start_cycle(self) -> str:
        cid = new_id()
        with _lock:
            self.conn.execute(
                "INSERT INTO cycles (id, timestamp, status) VALUES (?,?,?)",
                (cid, utcnow(), "running"),
            )
            self.conn.commit()
        return cid

    def finish_cycle(self, cid: str, status: str, ai_input: str, ai_output: str,
                     summary: str = "", confidence: str = "", error: str = "") -> None:
        with _lock:
            self.conn.execute(
                "UPDATE cycles SET status=?, ai_input=?, ai_output=?, analysis_summary=?, "
                "confidence=?, error=? WHERE id=?",
                (status, ai_input, ai_output, summary, confidence, error, cid),
            )
            self.conn.commit()

    def list_cycles(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM cycles ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- decisions ----------
    def add_decision(self, d: dict) -> None:
        with _lock:
            self.conn.execute(
                "INSERT INTO decisions (id, cycle_id, timestamp, action, asset_in, asset_out, "
                "amount, value_usd, pnl_per_trade, status, reason, tx_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    d.get("id", new_id()), d.get("cycle_id"), d.get("timestamp", utcnow()),
                    d["action"], d.get("asset_in"), d.get("asset_out"),
                    d.get("amount"), d.get("value_usd"), d.get("pnl_per_trade"),
                    d["status"], d.get("reason", ""), d.get("tx_id"),
                ),
            )
            self.conn.commit()

    def list_decisions(self, limit: int = 200) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- positions & cash ----------
    def get_positions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM positions").fetchall()
        return [dict(r) for r in rows]

    def get_position(self, asset: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM positions WHERE asset=?", (asset,)).fetchone()
        return dict(row) if row else None

    def upsert_position(self, asset: str, chain: str | None, quantity: float,
                        cost_basis_usd: float) -> None:
        with _lock:
            self.conn.execute(
                "INSERT INTO positions (asset, chain, quantity, cost_basis_usd, updated_at) "
                "VALUES (?,?,?,?,?) ON CONFLICT(asset) DO UPDATE SET "
                "chain=excluded.chain, quantity=excluded.quantity, "
                "cost_basis_usd=excluded.cost_basis_usd, updated_at=excluded.updated_at",
                (asset, chain, quantity, cost_basis_usd, utcnow()),
            )
            self.conn.commit()

    def delete_position(self, asset: str) -> None:
        with _lock:
            self.conn.execute("DELETE FROM positions WHERE asset=?", (asset,))
            self.conn.commit()

    def get_cash(self) -> float:
        row = self.conn.execute("SELECT usd FROM cash WHERE id=1").fetchone()
        return row["usd"] if row else 0.0

    def set_cash(self, usd: float) -> None:
        with _lock:
            self.conn.execute(
                "INSERT INTO cash (id, usd) VALUES (1,?) ON CONFLICT(id) DO UPDATE SET usd=excluded.usd",
                (usd,),
            )
            self.conn.commit()

    # ---------- demo wallet ----------
    def demo_rows(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM demo_wallet").fetchall()
        return [dict(r) for r in rows]

    def demo_get(self, chain: str, asset: str, contract: str = "") -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM demo_wallet WHERE chain=? AND asset=? AND contract=?",
            (chain, asset, contract)).fetchone()
        return dict(row) if row else None

    def demo_set(self, chain: str, asset: str, balance: float,
                 contract: str = "", is_gas: bool = False,
                 cost_basis_usd: float | None = None) -> None:
        """User-facing row setter. Cost basis is ONLY set when explicitly
        given (bot trades pass it); manual user rows get NO basis (0) so
        their P&L shows '-' honestly instead of a bogus number."""
        with _lock:
            self.conn.execute(
                "INSERT INTO demo_wallet (id, chain, asset, contract, balance, is_gas, cost_basis_usd) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(chain, asset, contract) DO UPDATE SET "
                "balance=excluded.balance, is_gas=excluded.is_gas, "
                "cost_basis_usd=COALESCE(excluded.cost_basis_usd, demo_wallet.cost_basis_usd)",
                (new_id(), chain, asset, contract, balance, 1 if is_gas else 0,
                 cost_basis_usd or 0.0),
            )
            self.conn.commit()

    def demo_adjust(self, chain: str, asset: str, delta: float,
                    contract: str = "", is_gas: bool = False,
                    usd_value: float = 0.0) -> float:
        """Add delta to a demo balance (can be negative); creates at 0 if new.
        usd_value: the USD value of the delta, tracked for cost basis.
        Returns the new balance. Refuses to go below 0 (returns old)."""
        with _lock:
            row = self.conn.execute(
                "SELECT balance, cost_basis_usd FROM demo_wallet "
                "WHERE chain=? AND asset=? AND contract=?",
                (chain, asset, contract)).fetchone()
            cur = row["balance"] if row else 0.0
            cur_basis = row["cost_basis_usd"] if row else 0.0
            new = cur + delta
            if new < 0:
                return cur  # insufficient: caller checks
            # cost basis: buying adds USD value; selling removes proportionally
            if delta > 0:
                new_basis = cur_basis + usd_value
            elif cur > 0:
                new_basis = cur_basis * (new / cur)
            else:
                new_basis = 0.0
            self.conn.execute(
                "INSERT INTO demo_wallet (id, chain, asset, contract, balance, is_gas, cost_basis_usd) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(chain, asset, contract) DO UPDATE SET "
                "balance=excluded.balance, cost_basis_usd=excluded.cost_basis_usd",
                (new_id(), chain, asset, contract, new, 1 if is_gas else 0, new_basis),
            )
            self.conn.commit()
            return new

    def demo_delete(self, chain: str, asset: str, contract: str = "") -> None:
        with _lock:
            self.conn.execute(
                "DELETE FROM demo_wallet WHERE chain=? AND asset=? AND contract=?",
                (chain, asset, contract))
            self.conn.commit()

    def demo_clear(self) -> None:
        with _lock:
            self.conn.execute("DELETE FROM demo_wallet")
            self.conn.commit()
