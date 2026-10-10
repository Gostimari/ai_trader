"""AI Decision Engine (§4.2): Mistral API, structured JSON responses, validated.

Critical rule: the LLM proposes; deterministic code validates and executes.
Invalid or unreachable AI responses mean NO trades that day.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from .constraints import ConstraintValidator, STABLECOIN
from .db import Database, utcnow
from .market import MarketDataError, MarketDataModule
from .news import NewsModule
from .portfolio import PortfolioTracker
from .settings import Constraints

MAX_ACTIONS_PER_CYCLE = 6


class AIAction(BaseModel):
    type: str = Field(pattern="^(buy|sell|swap)$")
    asset: str
    amount_or_percent: str | float
    reason: str
    asset_out: Optional[str] = None

    @field_validator("amount_or_percent")
    @classmethod
    def _amount_ok(cls, v):
        if isinstance(v, str):
            s = v.strip().rstrip("%")
            try:
                float(s)
            except ValueError:
                raise ValueError(f"Unparseable amount_or_percent: {v!r}")
        return v


class AIDecision(BaseModel):
    analysis_summary: str
    actions: list[AIAction]
    confidence: str = Field(pattern="^(high|medium|low)$")

    @field_validator("actions")
    @classmethod
    def _cap_actions(cls, v):
        if len(v) > MAX_ACTIONS_PER_CYCLE:
            raise ValueError(f"Too many actions ({len(v)} > {MAX_ACTIONS_PER_CYCLE})")
        return v


BASE_INSTRUCTIONS = """You are a crypto portfolio manager focused on meme coins.
You manage a portfolio to maximize profit while respecting the user's constraints.
You MUST respond with ONLY a JSON object, no markdown, matching exactly this schema:
{
  "analysis_summary": "2-4 sentence market assessment",
  "actions": [{"type": "buy|sell|swap", "asset": "TICKER", "amount_or_percent": "absolute amount of the base asset or a percent of portfolio like '10%'", "reason": "short why", "asset_out": "TICKER (swap only)"}],
  "confidence": "high|medium|low"
}
Rules:
- Use tickers exactly as they appear in the market data provided.
- amount_or_percent: prefer an absolute quantity of the asset; you may use "N%" meaning N% of current portfolio value for buys, or N% of current holdings for sells.
- "hold" requires no action: return an empty actions array.
- The execution layer will reject anything violating the user's constraints; do not propose violations.
- Cash is held in USD (paper mode). To exit a position, sell it.
"""


class DecisionEngine:
    def __init__(self, db: Database, constraints: Constraints,
                 market: MarketDataModule, news: NewsModule):
        self.db = db
        self.constraints = constraints
        self.market = market
        self.news = news
        self.tracker = PortfolioTracker(db)
        self.validator = ConstraintValidator(db, constraints)

    # ---------- cycle ----------
    def run_cycle(self) -> dict:
        cycle_id = self.db.start_cycle()
        try:
            prices = self.market.get_prices(force=True)
            assets = self.market.get_asset_map(force=True)
            portfolio = self.tracker.summary(prices)
        except MarketDataError as e:
            self.db.finish_cycle(cycle_id, "failed", "", "", error=f"market data: {e}")
            self.db.add_decision({"cycle_id": cycle_id, "action": "hold",
                                  "status": "failed", "reason": f"Market data unavailable: {e}"})
            return {"cycle_id": cycle_id, "status": "failed", "error": str(e)}

        stop = self.validator.check_stop_conditions(prices)
        if not stop.ok:
            self.db.finish_cycle(cycle_id, "blocked", "", "", error=stop.reason)
            self.db.add_decision({"cycle_id": cycle_id, "action": "hold",
                                  "status": "rejected", "reason": stop.reason})
            return {"cycle_id": cycle_id, "status": "blocked", "error": stop.reason}

        prompt = self._build_context(prices, assets, portfolio)
        try:
            raw = self._call_mistral(prompt)
        except Exception as e:
            self.db.finish_cycle(cycle_id, "failed", prompt, "", error=f"AI API: {e}")
            self.db.add_decision({"cycle_id": cycle_id, "action": "hold",
                                  "status": "failed",
                                  "reason": "AI unreachable; no trades made today (safety rule)."})
            return {"cycle_id": cycle_id, "status": "failed", "error": str(e)}

        try:
            decision = self._parse(raw)
        except (ValidationError, ValueError) as e:
            self.db.finish_cycle(cycle_id, "invalid", prompt, raw,
                                 error=f"invalid AI response: {e}")
            self.db.add_decision({"cycle_id": cycle_id, "action": "hold",
                                  "status": "failed",
                                  "reason": "AI response invalid; no trades made (safety rule)."})
            return {"cycle_id": cycle_id, "status": "invalid", "error": str(e)}

        executed, rejected = [], []
        for action in decision.actions:
            a = action.model_dump()
            result = self.validator.validate_action(a, prices, assets)
            if not result.ok:
                rejected.append(a)
                self.db.add_decision({
                    "cycle_id": cycle_id, "action": a["type"], "asset_in": a.get("asset"),
                    "asset_out": a.get("asset_out"), "status": "rejected",
                    "reason": f"Rejected ({result.reason}). AI said: {a['reason']}",
                })
                continue
            self._execute(a, prices)
            executed.append(a)

        self.db.finish_cycle(
            cycle_id, "completed", prompt, raw,
            summary=decision.analysis_summary, confidence=decision.confidence,
        )
        return {
            "cycle_id": cycle_id, "status": "completed",
            "analysis_summary": decision.analysis_summary,
            "confidence": decision.confidence,
            "executed": executed, "rejected": rejected,
        }

    # ---------- context & AI call ----------
    def _build_context(self, prices: dict[str, float], assets: dict[str, dict],
                       portfolio: dict) -> str:
        c = self.constraints
        parts = [BASE_INSTRUCTIONS, ""]
        parts.append(f"## User constraints (hard, enforced by code)\n{json.dumps(c.to_json(), indent=2)}")
        if c.custom_prompt.strip():
            parts.append(f"## User custom prompt (instructions the user wants you to follow)\n{c.custom_prompt.strip()}")
        parts.append(f"## Portfolio state\n{json.dumps(portfolio, indent=2)}")
        market_rows = [
            {k: a.get(k) for k in ("asset", "name", "chain", "price_usd", "change_24h_pct",
                                   "change_7d_pct", "volume_24h_usd", "market_cap_usd")}
            for a in assets.values()
        ]
        parts.append(f"## Market data (meme coins)\n{json.dumps(market_rows, indent=2)}")
        recent = self.db.list_decisions(limit=15)
        if recent:
            parts.append(f"## Recent decisions\n{json.dumps(recent, indent=2)}")
        return "\n".join(parts)

    def _call_mistral(self, prompt: str) -> str:
        from mistralai import Mistral
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY not configured")
        client = Mistral(api_key=api_key)
        model = os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""

    @staticmethod
    def _parse(raw: str) -> AIDecision:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in AI response")
        return AIDecision.model_validate(json.loads(text[start:end + 1]))

    # ---------- execution (paper mode; live is Phase 2) ----------
    def _execute(self, action: dict, prices: dict[str, float]) -> None:
        cycle_id = self.db.get_setting("last_cycle_id")
        asset = action["asset"]
        amount = self._resolve_amount(action, prices)
        price = prices.get(asset, 0.0)
        if action["type"] == "buy":
            cost = amount * price
            self.tracker.apply_buy(asset, self._chain_of(asset), amount, cost)
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "buy", "asset_in": asset, "asset_out": "USD",
                "amount": amount, "value_usd": cost, "status": "executed",
                "reason": action["reason"], "tx_id": f"paper-{utcnow()}-{asset}",
            })
        elif action["type"] == "sell":
            proceeds = amount * price
            pnl = self.tracker.apply_sell(asset, amount, proceeds)
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "sell", "asset_in": asset, "asset_out": "USD",
                "amount": amount, "value_usd": proceeds, "pnl_per_trade": pnl,
                "status": "executed", "reason": action["reason"],
                "tx_id": f"paper-{utcnow()}-{asset}",
            })
        elif action["type"] == "swap":
            dest = action.get("asset_out", "")
            proceeds = amount * price
            pnl = self.tracker.apply_sell(asset, amount, proceeds)
            dest_price = prices.get(dest, 0.0)
            qty = proceeds / dest_price if dest_price > 0 else 0.0
            self.tracker.apply_buy(dest, self._chain_of(dest), qty, proceeds)
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "swap", "asset_in": asset, "asset_out": dest,
                "amount": amount, "value_usd": proceeds, "pnl_per_trade": pnl,
                "status": "executed", "reason": action["reason"],
                "tx_id": f"paper-{utcnow()}-{asset}-{dest}",
            })

    def _chain_of(self, asset: str) -> str | None:
        snap = self.db.latest_market_snapshot()
        if not snap:
            return None
        import json as _json
        for a in _json.loads(snap["payload"]):
            if a["asset"] == asset:
                return a.get("chain")
        return None

    def _resolve_amount(self, action: dict, prices: dict[str, float]) -> float:
        """Convert absolute or 'N%' amount into an absolute asset quantity."""
        raw = action.get("amount_or_percent")
        if isinstance(raw, str) and raw.strip().endswith("%"):
            pct = float(raw.strip().rstrip("%"))
            if action["type"] == "buy":
                total = self.tracker.total_value(prices)
                budget = total * pct / 100
                price = prices.get(action["asset"], 0.0)
                return budget / price if price > 0 else 0.0
            if action["type"] in ("sell", "swap"):
                pos = self.db.get_position(action["asset"])
                return (pos["quantity"] * pct / 100) if pos else 0.0
        return float(raw)
