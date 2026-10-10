"""Daily scheduler: triggers the decision cycle at the configured UTC time."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

from .ai_engine import DecisionEngine
from .db import Database
from .settings import load_constraints


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

    def _loop(self) -> None:
        while not self._stop.wait(30):
            try:
                constraints = load_constraints(self.db)
                if constraints.emergency_stopped or constraints.trading_paused:
                    continue
                hh, mm = constraints.decision_time_utc.split(":")
                now = datetime.now(timezone.utc)
                target = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                last = self.db.get_setting("last_auto_cycle_date")
                today = now.date().isoformat()
                if now >= target and last != today:
                    self.db.set_setting("last_auto_cycle_date", today)
                    engine: DecisionEngine = self.engine_factory()
                    engine.run_cycle()
            except Exception:
                continue
