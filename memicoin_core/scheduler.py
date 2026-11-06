"""Daily scheduler: triggers the decision cycle at the configured UTC time.

A failed cycle (AI unreachable, market data down, ...) is retried after a
cooldown instead of being lost until the next day. A cycle that is skipped
because another one is already running is neither marked done nor failed.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from .ai_engine import DecisionEngine
from .db import Database
from .settings import load_constraints

RETRY_COOLDOWN_S = 1800  # retry a failed daily cycle every 30 minutes


class DailyScheduler:
    def __init__(self, db: Database, engine_factory):
        self.db = db
        self.engine_factory = engine_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _due(self, now: datetime, target_today_reached: bool, today: str) -> bool:
        if not target_today_reached:
            return False
        last = self.db.get_setting("last_auto_cycle_date")
        if last != today:
            return True
        # already ran today: retry only if the last attempt failed and the
        # cooldown has elapsed
        try:
            failed_at = float(self.db.get_setting("last_auto_cycle_failed_at") or 0)
        except (TypeError, ValueError):
            failed_at = 0.0
        return bool(failed_at) and time.time() - failed_at >= RETRY_COOLDOWN_S

    def _tick(self) -> None:
        constraints = load_constraints(self.db)
        if constraints.emergency_stopped or constraints.trading_paused:
            return
        hh, mm = constraints.decision_time_utc.split(":")
        now = datetime.now(timezone.utc)
        target = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        today = now.date().isoformat()
        if not self._due(now, now >= target, today):
            return
        engine: DecisionEngine = self.engine_factory()
        result = engine.run_cycle() or {}
        status = result.get("status")
        if status == "skipped":
            return  # a manual cycle is running; try again next poll
        if status in ("completed", "blocked"):
            self.db.set_setting("last_auto_cycle_date", today)
            self.db.set_setting("last_auto_cycle_failed_at", "")
        else:
            self.db.set_setting("last_auto_cycle_failed_at", str(time.time()))

    def _loop(self) -> None:
        while not self._stop.wait(30):
            try:
                self._tick()
            except Exception:
                continue
