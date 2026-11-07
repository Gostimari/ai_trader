"""AI Decision Engine (§4.2): Mistral API, structured JSON responses, validated.

Critical rule: the LLM proposes; deterministic code validates and executes.
Invalid or unreachable AI responses mean NO trades that day.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

from .constraints import ConstraintValidator, STABLECOIN
from .db import Database, utcnow
from .market import MarketDataError, MarketDataModule
from .portfolio import PortfolioTracker
from .settings import Constraints

log = logging.getLogger("memicoin.engine")

MAX_ACTIONS_PER_CYCLE = 6

# One cycle at a time app-wide: a scheduled cycle and a manual "Run now"
# must never execute concurrently against the same balances.
_cycle_lock = threading.Lock()


class AIAction(BaseModel):
    type: str = Field(pattern="^(buy|sell|swap|bridge)$")
    asset: str = ""
    amount_or_percent: str | float
    reason: str
    asset_out: Optional[str] = None
    from_chain: Optional[str] = None
    to_chain: Optional[str] = None

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
        # CLAMP, never invalidate: an enthusiastic response with more
        # actions than the cap keeps the first MAX_ACTIONS_PER_CYCLE
        # (all still go through validation). Discarding a whole cycle
        # over action count turned good decisions into holds.
        if len(v) > MAX_ACTIONS_PER_CYCLE:
            v = v[:MAX_ACTIONS_PER_CYCLE]
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
- amount_or_percent: BUY sizing is in USD — a plain number is the USD amount to spend (e.g. 25 means "spend $25"), or use "N%" for N% of total portfolio value (e.g. "10%" of a $100 portfolio = $10). Never express a buy as a coin quantity: many meme coins cost fractions of a cent, so coin quantities produce dust trades that are rejected. SELL/SWAP sizing: a plain number is a coin quantity, or "N%" for N% of your current holdings of that asset.
- Size buys meaningfully: several USD at minimum, and respect the max position size constraint.
- "hold" requires no action: return an empty actions array.
- CONSISTENCY: if your analysis_summary says the portfolio "needs rebalancing"
  or identifies opportunities, you MUST propose actions that follow through.
  An empty actions array is only acceptable when you genuinely recommend
  caution - and then your summary must say so explicitly ("recommend holding
  because..."), not describe opportunities you then ignore. If investable
  funds exist and momentum is favorable, propose the trades.
- DIVERSIFY: when buying, prefer proposing MULTIPLE buys across different
  coins (each within the max position size) instead of one large buy.
  A good cycle builds or adjusts several positions at once — e.g. 2-4
  distinct actions — unless the market genuinely favors caution.
- Percent amounts are checked against portfolio value (buys) or holdings
  (sells) by the execution layer BEFORE execution. Anything over the
  user's max position size will be rejected; size proposals to fit.
- NEW TOKENS: the market list automatically includes freshly launched /
  boosted tokens WITH their contract addresses - you MAY buy them without
  asking the user. But they are the riskiest assets: weigh liquidity,
  24h volume and momentum, size them small, and never concentrate the
  portfolio in a token with thin liquidity.
- CROSS-CHAIN FUNDING: USDC can be moved between EVM chains (ethereum,
  arc, and other CCTP chains listed in your context) with a "bridge"
  action: {"type": "bridge", "from_chain": "...", "to_chain": "...",
  "amount_or_percent": "50%"}. Bridging takes minutes and the funds only
  become investable on the NEXT cycle.
- GAS MANAGEMENT: gas reserves are adaptive - small gas balances reserve
  only 1 transaction, leaving the rest investable. If a chain's gas token
  is running low (see the wallet gas figures) you MAY propose swapping a
  small slice of another held asset into the gas token (e.g. sell a little
  USDC for ETH on the same chain) to keep future transactions affordable.
  On a chain holding ONLY the gas token, topping up is impossible without
  external funds - do not attempt it; the user will see a gas warning.
- SAME-CHAIN FIRST: bridges are EXPENSIVE — gas on BOTH chains plus a
  service fee, and the funds sit unusable until the next cycle. Strongly
  prefer investing with funds ALREADY on a chain before bridging. Only
  propose a bridge when (a) the source chain has genuinely idle funds,
  (b) the destination opportunity is clearly better than anything
  reachable with existing destination funds, and (c) the expected edge
  exceeds the bridge costs. Never bridge when the destination already
  has sufficient funds for the intended buy.
- SOLANA: only propose Solana assets when the context says a Solana key
  is imported; otherwise Solana actions are rejected.
- TOKEN IDENTIFIERS: each asset in the market data includes its "contract"
  address. For buy/sell/swap actions ALWAYS set "asset" to the token's
  contract address from the market data (the execution layer requires
  it; trades by bare symbol are rejected).
- HOLDINGS TRUTH: you can only SELL or SWAP assets that show a positive balance in the Portfolio state RIGHT NOW. Do NOT infer current holdings from Recent decisions — the wallet may have been edited, reset, or spent since those decisions. Proposing to sell an asset with no balance is always rejected.
- The execution layer will reject anything violating the user's constraints; do not propose violations.
- Cash is held in USD (paper mode). To exit a position, sell it.
"""


class DecisionEngine:
    def __init__(self, db: Database, constraints: Constraints,
                 market: MarketDataModule, news=None,
                 onchain_secrets=None):
        self.db = db
        self.constraints = constraints
        self.market = market
        self.news = news  # unused; retained for API compatibility
        self._onchain_secrets = onchain_secrets
        self.tracker = PortfolioTracker(db)
        solana_present = False
        if onchain_secrets is not None:
            try:
                solana_present = onchain_secrets.has("onchain_solana_private_key")
            except Exception:
                pass
        self.validator = ConstraintValidator(db, constraints,
                                             solana_key_present=solana_present,
                                             market=market)

    # ---------- cycle ----------
    def run_cycle(self) -> dict:
        if not _cycle_lock.acquire(blocking=False):
            return {"status": "skipped",
                    "error": "another decision cycle is already running"}
        try:
            return self._run_cycle_locked()
        finally:
            _cycle_lock.release()

    def _run_cycle_locked(self) -> dict:
        cycle_id = self.db.start_cycle()
        # mode flag FIRST: stop checks and amount resolution must both be
        # demo-aware from the very beginning of the cycle
        self.db.demo_mode_percent = (not self.constraints.paper_trading
                                     and self.constraints.live_execution_mode == "demo")
        if (not self.constraints.paper_trading
                and self.constraints.live_execution_mode == "onchain"):
            self._resume_pending_bridges(cycle_id)
        try:
            prices = self.market.get_prices(force=True)
            assets = self.market.get_asset_map(force=True)
            portfolio = self.tracker.summary(prices)
            if (not self.constraints.paper_trading
                    and self.constraints.live_execution_mode == "demo"):
                portfolio = self._demo_portfolio_summary(prices)
            elif (not self.constraints.paper_trading
                    and self.constraints.live_execution_mode == "onchain"):
                try:
                    portfolio = self._onchain_portfolio_summary(prices)
                    self.validator.set_live_holdings(
                        portfolio["holdings"], portfolio["total_value_usd"])
                except Exception:
                    pass  # fall back to paper book; validator stays conservative
        except MarketDataError as e:
            self.db.finish_cycle(cycle_id, "failed", "", "", error=f"market data: {e}")
            self.db.add_decision({"cycle_id": cycle_id, "action": "hold",
                                  "status": "failed", "reason": f"Market data unavailable: {e}"})
            return {"cycle_id": cycle_id, "status": "failed", "error": str(e)}

        # ---- per-position stop-loss sweep (deterministic, code-enforced) ----
        stop_sold = self._apply_per_position_stops(cycle_id, prices, assets)
        if stop_sold is None:  # market data failed mid-sweep; be safe, abort cycle
            self.db.finish_cycle(cycle_id, "failed", "", "", error="market data failure during stop-loss sweep")
            return {"cycle_id": cycle_id, "status": "failed", "error": "stop-loss sweep: market data"}

        live_total = None
        if not self.constraints.paper_trading:
            # prefer the total already read with the portfolio summary: one
            # wallet round-trip per cycle, and execution resolves percent
            # buys against the SAME total validation used
            live_total = (getattr(self.validator, "live_wallet_total", None)
                          or self._live_wallet_total(prices))
            self._ensure_pnl_baseline(live_total)
        stop = self.validator.check_stop_conditions(prices, live_total_usd=live_total)
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

        executed, rejected, failed = [], [], []

        def try_actions(actions) -> None:
            for action in actions:
                a = action.model_dump() if hasattr(action, "model_dump") else dict(action)
                if a.get("type") == "bridge":
                    before = len(executed)
                    self._execute_bridge(cycle_id, a)
                    # count it: executed if a decision row says executed
                    rows = self.db.list_decisions(limit=3)
                    if any(d.get("action") == "bridge" and d["status"] == "executed"
                           and d.get("cycle_id") == cycle_id for d in rows):
                        executed.append(a)
                    else:
                        failed.append(a)
                    continue
                # annotate the token's chain so the percent resolver can
                # pick the right demo funds
                info = assets.get(a.get("asset", ""))
                if info is None:
                    for cand in assets.values():
                        if (cand.get("contract") or "").lower() == (a.get("asset") or "").lower():
                            info = cand
                            break
                if info:
                    a["_chain"] = info.get("chain", "")
                    a["_price"] = prices.get(info.get("asset", ""), 0.0)
                result = self.validator.validate_action(a, prices, assets)
                if not result.ok:
                    rejected.append(a)
                    self.db.add_decision({
                        "cycle_id": cycle_id, "action": a["type"], "asset_in": a.get("asset"),
                        "asset_out": a.get("asset_out"), "status": "rejected",
                        "reason": f"Rejected ({result.reason}). AI said: {a['reason']}",
                    })
                    continue
                self.db.set_setting("last_cycle_id", cycle_id)
                clamp_note = a.pop("_clamped", None)
                if clamp_note:
                    a["reason"] = (a.get("reason", "") + " | " + clamp_note).strip(" |")
                err = self._execute(cycle_id, a, prices,
                                       live_total_usd=live_total)
                if err:
                    failed.append(a)
                    self.db.add_decision({
                        "cycle_id": cycle_id, "action": a["type"], "asset_in": a.get("asset"),
                        "asset_out": a.get("asset_out"), "status": "failed",
                        "reason": f"Execution failed: {err}",
                    })
                    continue
                executed.append(a)

        # Empty-actions challenge: the AI proposed nothing. If investable
        # funds exist, push back ONCE - models sometimes describe a plan in
        # the summary but forget to commit it to actions.
        if not decision.actions:
            has_funds = False
            try:
                summary = self.tracker.summary(prices)
                if summary.get("cash_usd", 0) > 1:
                    has_funds = True
            except Exception:
                pass
            if (not self.constraints.paper_trading
                    and self.constraints.live_execution_mode == "demo"):
                from .demowallet import DemoWallet
                demo = DemoWallet(self.db)
                for row in demo.rows():
                    if row["is_gas"]:
                        continue
                    if row["balance"] > 0:
                        has_funds = True
                        break
                # gas itself is spendable (investable portion)
                for ch in {r["chain"] for r in demo.rows()}:
                    if demo.investable_gas(ch) > 0:
                        has_funds = True
                        break
            if has_funds:
                challenge = (
                    "Your previous response proposed ZERO actions, but your own "
                    "analysis identified opportunities and the portfolio has "
                    "investable funds. Re-decide: either propose SPECIFIC actions "
                    "(buy/sell/swap/bridge with amounts) that act on your "
                    "analysis, or explicitly justify holding with concrete risk "
                    "reasons. Same JSON schema as before.")
                try:
                    raw2 = self._call_mistral(challenge)
                    decision2 = self._parse(raw2)
                    if decision2.actions:
                        decision = decision2  # adopt the committed decision
                        self.db.add_decision({
                            "cycle_id": cycle_id, "action": "hold",
                            "status": "replaced",
                            "reason": "AI initially proposed no actions; "
                                      "challenged once and it committed to "
                                      f"{len(decision2.actions)} action(s).",
                        })
                except Exception:
                    pass  # challenge is best-effort; original hold stands

        try_actions(decision.actions)

        # One bounded replan: if everything the AI proposed was blocked and
        # nothing executed, give the AI the rejection reasons and let it
        # propose alternatives once. All normal validation still applies.
        if rejected and not executed and decision.actions:
            feedback = (
                "Your previous proposed actions were ALL rejected by the user's "
                "constraints:\n" + "\n".join(
                    f"- {r['type']} {r.get('asset', '')} "
                    f"{r.get('amount_or_percent', '')}: "
                    + next((d["reason"] for d in self.db.list_decisions(limit=10)
                            if d["asset_in"] == r.get("asset") and d["status"] == "rejected"
                            and d["cycle_id"] == cycle_id), "constraint violation")
                    for r in rejected)
                + "\n\nPropose DIFFERENT actions that fit the constraints. "
                  "Same JSON schema as before. Portfolio state is unchanged."
            )
            try:
                raw2 = self._call_mistral(feedback)
                decision2 = self._parse(raw2)
            except Exception:
                decision2 = None  # replan is a bonus round; failure is non-fatal
            if decision2 and decision2.actions:
                rejected2_before = len(rejected)
                try_actions(decision2.actions)
                # merge the replan's rejects into the report; keep first-round
                # rejections out of the final count to avoid confusion
                if len(rejected) > rejected2_before:
                    rejected = rejected[rejected2_before:]

        self.db.finish_cycle(
            cycle_id, "completed", prompt, raw,
            summary=decision.analysis_summary, confidence=decision.confidence,
        )
        return {
            "cycle_id": cycle_id, "status": "completed",
            "analysis_summary": decision.analysis_summary,
            "confidence": decision.confidence,
            "executed": executed, "rejected": rejected, "failed": failed,
        }


    def _live_wallet_total(self, prices: dict[str, float]) -> float | None:
        """USD value of the wallet the bot actually trades (demo wallet or
        the imported on-chain wallet). None when it cannot be determined."""
        if self.constraints.live_execution_mode == "demo":
            from .demowallet import DemoWallet, NATIVE_WRAPPED
            demo = DemoWallet(self.db)
            wrapped_needed = [NATIVE_WRAPPED[r["chain"]] for r in demo.rows()
                              if r["is_gas"] and r["chain"] in NATIVE_WRAPPED
                              and r["chain"] != "arc"]
            wrapped_prices: dict = {}
            if wrapped_needed:
                try:
                    wrapped_prices = self.market.get_prices_for_addresses(wrapped_needed)
                except Exception:
                    wrapped_prices = {}
            total = 0.0
            for r in demo.rows():
                p = prices.get(r["asset"].upper(), 0.0)
                if p <= 0 and r["asset"].upper() in ("USDC", "USDT"):
                    p = 1.0
                if p <= 0 and r["is_gas"]:
                    if r["chain"] == "arc":
                        p = 1.0
                    else:
                        w = NATIVE_WRAPPED.get(r["chain"])
                        if w:
                            p = wrapped_prices.get(w.lower(), 0.0)
                total += r["balance"] * p
            return total
        try:
            return self._onchain_portfolio_summary(prices)["total_value_usd"]
        except Exception:
            return None

    def _ensure_pnl_baseline(self, live_total: float | None) -> None:
        """Snapshot the wallet value the first time a cycle runs after the
        wallet changed (mode switch, edit, clear). This baseline is what the
        profit-target / max-loss stops compare against in demo and on-chain
        modes - the paper books are meaningless there. Reset by the wallet
        endpoints; empty/zero here re-snapshots on the next funded cycle."""
        if live_total is None or live_total <= 0:
            return
        baseline = self.db.get_setting("pnl_baseline_usd")
        if not baseline or float(baseline) <= 0:
            self.db.set_setting("pnl_baseline_usd", float(live_total))

    def _apply_per_position_stops(self, cycle_id: str, prices: dict[str, float],
                                  assets: dict[str, dict]) -> list | None:
        """Sell every position that is down >= per_position_stop_pct into cash.
        Deterministic — not an AI decision. Returns list of sold assets, or
        None on market-data failure (caller aborts the cycle safely)."""
        c = self.constraints
        threshold = getattr(c, "per_position_stop_pct", None) or 0
        if threshold <= 0:
            return []
        # demo mode: positions are non-gas demo rows with a cost basis
        if getattr(self.db, "demo_mode_percent", False):
            from .demowallet import DemoWallet
            demo = DemoWallet(self.db)
            sold = []
            for r in demo.rows():
                if r["is_gas"] or r["balance"] <= 0 or r["cost_basis_usd"] <= 0:
                    continue
                price = prices.get(r["asset"].upper(), 0.0)
                if price <= 0 and r.get("contract"):
                    try:
                        addr_p = self.market.get_prices_for_addresses([r["contract"]])
                        price = addr_p.get(r["contract"].lower(), 0.0)
                    except Exception:
                        pass
                if price <= 0:
                    continue
                value = r["balance"] * price
                dd = (value - r["cost_basis_usd"]) / r["cost_basis_usd"] * 100
                if dd > -threshold:
                    continue
                action = {"type": "sell", "asset": r["asset"],
                          "_contract": r.get("contract", ""),
                          "amount": r["balance"],
                          "amount_or_percent": r["balance"],
                          "reason": (f"automatic stop-loss: position {dd:.1f}% "
                                     f"below cost (limit {threshold:.0f}%)")}
                self.db.set_setting("last_cycle_id", cycle_id)
                err = self._execute(cycle_id, action, prices)
                if err is None:
                    sold.append(r["asset"])
                else:
                    self.db.add_decision({
                        "cycle_id": cycle_id, "action": "sell", "asset_in": r["asset"],
                        "status": "failed",
                        "reason": f"Stop-loss sell failed: {err}",
                    })
            return sold
        sold = []
        for p in list(self.db.get_positions()):
            asset = p["asset"]
            price = prices.get(asset, 0.0)
            if price <= 0:
                continue  # no reliable price: do NOT trade blindly
            value = p["quantity"] * price
            cost = p["cost_basis_usd"]
            if cost <= 0:
                continue
            drawdown_pct = (value - cost) / cost * 100
            if drawdown_pct > -threshold:
                continue  # within limit
            action = {"type": "sell", "asset": asset,
                      "amount": p["quantity"],
                      "amount_or_percent": p["quantity"],
                      "reason": f"automatic stop-loss: position {drawdown_pct:.1f}% "
                                f"below cost (limit {threshold:.0f}%)"}
            self.db.set_setting("last_cycle_id", cycle_id)
            err = self._execute(cycle_id, action, prices)
            if err is None:
                sold.append(asset)
            else:
                self.db.add_decision({
                    "cycle_id": cycle_id, "action": "sell", "asset_in": asset,
                    "status": "failed",
                    "reason": f"Stop-loss sell failed: {err}",
                })
        return sold

    # ---------- context & AI call ----------
    def _demo_portfolio_summary(self, prices: dict[str, float]) -> dict:
        """Portfolio state built from the DEMO wallet (what the AI must see
        in demo mode), with per-chain investable amounts. Gas/native tokens
        (ETH etc.) are priced via their wrapped contracts since they are not
        in the meme-coin symbol map."""
        from .demowallet import DemoWallet, NATIVE_WRAPPED
        demo = DemoWallet(self.db)
        rows = demo.rows()
        # batch-price wrapped natives for gas rows (ETH->WETH etc.)
        wrapped_needed = [NATIVE_WRAPPED[r["chain"]] for r in rows
                          if r["is_gas"] and r["chain"] in NATIVE_WRAPPED
                          and r["chain"] != "arc"]
        wrapped_prices: dict = {}
        if wrapped_needed:
            try:
                wrapped_prices = self.market.get_prices_for_addresses(wrapped_needed)
            except Exception:
                pass
        by_chain: dict = {}
        total = 0.0
        for r in rows:
            p = prices.get(r["asset"].upper(), 0.0)
            if p <= 0 and r["asset"].upper() in ("USDC", "USDT"):
                p = 1.0
            if p <= 0 and r["is_gas"]:
                w = NATIVE_WRAPPED.get(r["chain"])
                if r["chain"] == "arc":
                    p = 1.0
                elif w:
                    p = wrapped_prices.get(w.lower(), 0.0)
            entry = {"asset": r["asset"], "chain": r["chain"],
                     "quantity": r["balance"], "is_gas": bool(r["is_gas"]),
                     "value_usd": r["balance"] * p}
            by_chain.setdefault(r["chain"], []).append(entry)
            total += entry["value_usd"]
        # investable gas per chain (balance minus reserve), with the
        # per-swap fee valued in USD so the AI can weigh it against
        # small trade sizes
        for chain, entries in by_chain.items():
            inv = demo.investable_gas(chain)
            reserve = demo.gas_reserve(chain)
            per_swap = demo.per_tx_gas(chain)
            unit = 0.0
            for e in entries:
                if e.get("is_gas") and e["quantity"] > 0:
                    unit = e["value_usd"] / e["quantity"]
                    break
            note = (f"investable gas on {chain}: {inv:.6f} "
                    f"(reserve {reserve:.6f} = {per_swap:.6f}/tx kept for fees)")
            if unit > 0:
                note += f"; one swap costs ~{per_swap:.6f} = ~${per_swap * unit:.4f}"
            entries.append({"_meta": True, "note": note})
        return {"demo_wallet": True, "chains": by_chain,
                "total_value_usd": total,
                "note": "Portfolio is the DEMO wallet. Fund buys with USDC "
                        "where available, or with investable gas-token amounts. "
                        "Buys settle against these balances (simulated)."}

    def _onchain_portfolio_summary(self, prices: dict[str, float]) -> dict:
        """Portfolio state built from the IMPORTED WALLET (real mode) — what
        the AI must see in onchain mode, so it can only trade assets that
        actually exist on-chain right now."""
        from .onchain import (read_wallet_balances, EVM_CHAINS,
                              build_extra_contracts)
        from .demowallet import NATIVE_WRAPPED
        try:
            extra = build_extra_contracts(self._onchain_secrets, self.market)
        except Exception:
            extra = None
        raw = read_wallet_balances(self._onchain_secrets,
                                   extra_contracts=extra) or {}
        evm = raw.get("evm") or {}
        # batch-price every token contract and wrapped native once
        contracts: list[str] = []
        for entry in (evm.get("chains") or {}).values():
            for t in entry.get("tokens", []):
                if t.get("contract"):
                    contracts.append(t["contract"])
        wrapped_needed = []
        for chain, entry in (evm.get("chains") or {}).items():
            if entry.get("native") and chain in NATIVE_WRAPPED and chain != "arc":
                wrapped_needed.append(NATIVE_WRAPPED[chain])
        contracts += wrapped_needed
        addr_prices: dict = {}
        if contracts:
            try:
                addr_prices = self.market.get_prices_for_addresses(contracts)
            except Exception:
                addr_prices = {}

        def _price(asset, chain, is_gas, contract=""):
            p = prices.get((asset or "").upper(), 0.0)
            if p <= 0 and (asset or "").upper() in ("USDC", "USDT"):
                p = 1.0
            if p <= 0 and is_gas:
                if chain == "arc":
                    p = 1.0
                else:
                    w = NATIVE_WRAPPED.get(chain)
                    if w:
                        p = addr_prices.get(w.lower(), 0.0)
            if p <= 0 and contract:
                p = addr_prices.get(contract.lower(), 0.0)
            return p

        by_chain: dict = {}
        total = 0.0
        for chain, entry in (evm.get("chains") or {}).items():
            sym = EVM_CHAINS.get(chain, {}).get("native_symbol", "ETH")
            rows = []
            if entry.get("native"):
                q = entry["native"]
                v = q * _price(sym, chain, True)
                rows.append({"asset": sym, "chain": chain, "quantity": q,
                             "is_gas": True, "value_usd": v})
            for t in entry.get("tokens", []):
                q = t.get("balance") or 0.0
                v = q * _price(t.get("symbol"), chain, False, t.get("contract", ""))
                rows.append({"asset": t.get("symbol") or (t.get("contract") or "?")[:8],
                             "chain": chain, "quantity": q, "is_gas": False,
                             "value_usd": v, "contract": t.get("contract", "")})
            if rows:
                by_chain[chain] = rows
                total += sum(r["value_usd"] for r in rows)
        sol = raw.get("solana")
        if sol and sol.get("address"):
            rows = []
            if sol.get("native"):
                q = sol["native"]
                v = q * _price("SOL", "solana", True)
                rows.append({"asset": "SOL", "chain": "solana", "quantity": q,
                             "is_gas": True, "value_usd": v})
            for t in sol.get("tokens", []):
                q = t.get("balance") or 0.0
                v = q * _price(t.get("symbol"), "solana", False, t.get("contract", ""))
                rows.append({"asset": t.get("symbol") or (t.get("contract") or "?")[:8],
                             "chain": "solana", "quantity": q, "is_gas": False,
                             "value_usd": v, "contract": t.get("contract", "")})
            if rows:
                by_chain["solana"] = rows
                total += sum(r["value_usd"] for r in rows)
        holdings = []
        for rows in by_chain.values():
            for r in rows:
                if r["quantity"] <= 0:
                    continue
                h = {"asset": r["asset"], "chain": r["chain"],
                     "quantity": r["quantity"], "is_gas": r["is_gas"]}
                if r.get("contract"):
                    h["contract"] = r["contract"]
                holdings.append(h)
        summary = {"real_wallet": True, "chains": by_chain, "holdings": holdings,
                   "total_value_usd": total,
                   "note": "Portfolio is the IMPORTED WALLET (real funds). You can "
                           "only SELL or SWAP assets listed above with a positive "
                           "quantity, referencing tokens by their contract address "
                           "(on-chain mode requires contracts). Fund buys with the "
                           "investable gas-token amounts shown in the Gas reserve "
                           "section."}
        if raw.get("error"):
            summary["warning"] = f"wallet read incomplete: {raw['error']}"
        return summary

    def _build_context(self, prices: dict[str, float], assets: dict[str, dict],
                       portfolio: dict) -> str:
        c = self.constraints
        parts = [BASE_INSTRUCTIONS, ""]
        parts.append(f"## User constraints (hard, enforced by code)\n{json.dumps(c.to_json(), indent=2)}")
        if c.custom_prompt.strip():
            parts.append(f"## User custom prompt (instructions the user wants you to follow)\n{c.custom_prompt.strip()}")
        parts.append(f"## Portfolio state\n{json.dumps(portfolio, indent=2)}")
        solana_key = "yes" if (self._onchain_secrets is not None and
                               self._onchain_secrets.has("onchain_solana_private_key")) else "NO"
        from .bridge import CCTP_DOMAINS, is_testnet_chain
        bridgeable = [ch for ch in CCTP_DOMAINS
                      if is_testnet_chain(ch) == c.onchain_testnet]
        parts.append(
            "## Wallet capabilities\n"
            f"- Solana key imported: {solana_key} "
            f"({'Solana assets allowed' if solana_key == 'yes' else 'do NOT propose Solana assets'})\n"
            f"- CCTP-bridgeable USDC chains: {', '.join(bridgeable)} (bridge action available)\n"
            "- Swaps settle on the TOKEN'S own chain (it must be in allowed "
            "chains): fund a buy with the investable gas-token funds of the "
            "token's chain, shown in the Gas reserve section"
            + ("" if c.onchain_testnet else
               "\n- Testnet mode is OFF: never propose sepolia/base-sepolia bridges or assets"))
        # gas reserves: what's investable per chain (real mode only)
        if not self.constraints.paper_trading and self._onchain_secrets is not None \
                and self._onchain_secrets.has("onchain_private_key"):
            try:
                from .onchain import EVMWallet, EVM_CHAINS
                from .demowallet import NATIVE_WRAPPED
                wrapped = [NATIVE_WRAPPED[ch] for ch in EVM_CHAINS
                           if ch != "arc" and ch in NATIVE_WRAPPED]
                addr_p: dict = {}
                if wrapped:
                    try:
                        addr_p = self.market.get_prices_for_addresses(wrapped)
                    except Exception:
                        addr_p = {}
                lines = []
                for chain in EVM_CHAINS:
                    if EVM_CHAINS[chain].get("testnet") and not self.constraints.onchain_testnet:
                        continue  # testnet RPCs only in testnet mode
                    try:
                        w = EVMWallet(self._onchain_secrets, chain)
                        sym = EVM_CHAINS[chain].get("native_symbol", "ETH")
                        per_tx = w.estimated_gas_cost()
                        if sym == "USDC":
                            p = 1.0
                        else:
                            p = prices.get(sym.upper(), 0.0)
                            if p <= 0:
                                waddr = NATIVE_WRAPPED.get(chain)
                                p = addr_p.get(waddr.lower(), 0.0) if waddr else 0.0
                        fee = f"~{per_tx:.6f} {sym}/swap"
                        if p > 0:
                            fee += f" (~${per_tx * p:.4f})"
                        lines.append(
                            f"  {chain}: {w.investable_native():.6f} {sym} "
                            f"investable (reserve {w.gas_reserve():.6f} kept for gas; "
                            f"each swap costs {fee})")
                    except Exception:
                        continue
                if lines:
                    parts.append("## Gas reserve (NEVER spend into the reserve)\n"
                                 "A gas reserve is held back on each chain for future "
                                 "transaction fees. Only the 'investable' amount may "
                                 "be traded. Every swap also PAYS the per-swap fee "
                                 "shown - weigh it against the trade size and avoid "
                                 "trades where fees eat a large share:\n" + "\n".join(lines))
            except Exception:
                pass
        market_rows = [
            {**{k: a.get(k) for k in ("asset", "name", "chain", "price_usd",
                                      "change_24h_pct", "change_7d_pct",
                                      "volume_24h_usd", "market_cap_usd")},
             # the contract is REQUIRED for demo/on-chain execution: the AI
             # must reference tokens by contract to trade them
             "contract": a.get("contract", ""),
             "liquidity_usd": a.get("liquidity_usd", 0)}
            for a in assets.values()
        ]
        parts.append(f"## Market data (meme coins)\n{json.dumps(market_rows, indent=2)}")
        recent = self.db.list_decisions(limit=15)
        if recent:
            parts.append(f"## Recent decisions\n{json.dumps(recent, indent=2)}")
        return "\n".join(parts)

    def _call_mistral(self, prompt: str) -> str:
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key and self._onchain_secrets is not None:
            try:
                api_key = self._onchain_secrets.get("mistral_api_key") or ""
            except Exception:
                pass
        if not api_key:
            raise RuntimeError("Mistral API key not configured (set it in Settings or the .env file)")
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

    # ---------- execution (paper or live, chosen by constraints) ----------
    def _execute(self, cycle_id: str, action: dict, prices: dict[str, float],
                 live_total_usd: float | None = None) -> str | None:
        """Execute one validated action. Returns error string or None on success."""
        asset = action["asset"]
        amount = self._resolve_amount(action, prices,
                                      live_total_usd=live_total_usd)
        price = prices.get(asset, 0.0)
        if amount <= 0:
            return f"resolved to non-positive amount for {asset}"

        if not self.constraints.paper_trading:
            return self._execute_live(cycle_id, action, asset, amount, prices)

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
        return None

    def _execute_bridge(self, cycle_id: str, action: dict) -> None:
        """CCTP USDC bridge: burn on src, auto-mint on dst (minutes later).
        Logged as a decision with the follow-up explanation."""
        from .bridge import bridge_supported, bridge_usdc
        src = (action.get("from_chain") or "").strip().lower()
        dst = (action.get("to_chain") or "").strip().lower()
        amt_raw = action.get("amount_or_percent")
        reason = action.get("reason", "")

        # resolve the amount against the SOURCE chain's USDC balance
        # (percentages of the whole portfolio make no sense for a bridge:
        # only what sits on src can move)
        # network gate: testnet chains only in testnet mode, mainnet chains
        # only with testnet OFF (a 'test' bridge must never burn real USDC
        # and a real bridge must never burn faucet tokens)
        from .bridge import is_testnet_chain
        want_testnet = self.constraints.onchain_testnet
        for ch in (src, dst):
            if is_testnet_chain(ch) != want_testnet:
                self.db.add_decision({
                    "cycle_id": cycle_id, "action": "bridge", "status": "rejected",
                    "reason": (f"Bridge {src}->{dst} rejected: "
                               + ("testnet mode is ON - only sepolia/base-sepolia "
                                  "bridges are allowed" if want_testnet else
                                  "testnet mode is OFF - enable it to bridge on "
                                  "sepolia/base-sepolia")
                               + f". {reason}")})
                return
        is_pct = isinstance(amt_raw, str) and amt_raw.strip().endswith("%")
        try:
            val = float(str(amt_raw).rstrip("%"))
        except (TypeError, ValueError):
            val = 0.0
        src_balance = 0.0
        if getattr(self.db, "demo_mode_percent", False):
            from .demowallet import DemoWallet
            demo = DemoWallet(self.db)
            src_balance = demo.balance(src, "USDC")
        elif (not self.constraints.paper_trading
              and self.constraints.live_execution_mode == "onchain"):
            # real wallet: read the ACTUAL USDC balance on the source chain
            # (paper cash is meaningless here)
            src_balance = self._onchain_usdc_balance(src)
        else:
            src_balance = self.db.get_cash() if src == "ethereum" else 0.0
        if is_pct:
            amt = src_balance * (val / 100.0)
        elif val > src_balance > 0:
            # absolute amount beyond the source balance: clamp to it
            # (consistent with the position-size clamp philosophy)
            amt = src_balance
        else:
            amt = val

        if not src or not dst or amt <= 0 or src_balance <= 0:
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "status": "rejected",
                "reason": (f"Malformed bridge proposal: {src}->{dst} {amt_raw!r}. "
                           f"{reason}" if not src or not dst else
                           f"Nothing to bridge: {src_balance:.4f} USDC on {src}. {reason}")})
            return
        # bridges move funds BETWEEN chains, so BOTH endpoints must be
        # user-allowed (constraint enforced in code, not by the AI)
        allowed = [c.lower() for c in self.constraints.allowed_chains]
        if self.constraints.autonomy_mode == "constrained" and (
                src not in allowed or dst not in allowed):
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "status": "rejected",
                "reason": (f"Bridge {src}->{dst} rejected: chain not in your "
                           f"allowed chains ({', '.join(allowed)}). {reason}")})
            return
        if getattr(self.db, "demo_mode_percent", False):
            # demo mode: settle against the demo wallet (gas deducted on src)
            from .demowallet import DemoWallet
            demo = DemoWallet(self.db)
            ok, msg = demo.bridge(src, dst, "USDC", amt)
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "asset_in": "USDC",
                "asset_out": dst, "amount": amt,
                "status": "executed" if ok else "failed",
                "reason": (f"[DEMO] Bridge {src}->{dst} of {amt} USDC: {msg}. "
                           f"Funds investable on {dst} next cycle. {reason}"
                           if ok else f"[DEMO] Bridge {src}->{dst} failed: {msg}"),
            })
        elif not self.constraints.paper_trading:
            # real mode: CCTP burn; mint lands after attestation
            if self._onchain_secrets is None or not self._onchain_secrets.has("onchain_private_key"):
                self.db.add_decision({
                    "cycle_id": cycle_id, "action": "bridge", "status": "failed",
                    "reason": "Bridge failed: no EVM wallet key imported"})
                return
            res = bridge_usdc(self._onchain_secrets, src, dst, amt)
            if res.pending:
                # burn confirmed, attestation not landed yet: persist so a
                # later cycle completes the mint (funds are NOT lost)
                self.db.add_pending_bridge(src, dst, amt,
                                           res.message_hash, res.message_bytes)
                self.db.add_decision({
                    "cycle_id": cycle_id, "action": "bridge", "asset_in": "USDC",
                    "asset_out": dst, "amount": amt, "status": "executed",
                    "reason": f"Bridge {src}->{dst}: {res.error}",
                    "tx_id": res.tx_id})
                return
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "asset_in": "USDC",
                "asset_out": dst, "amount": amt, "status": "executed" if res.ok else "failed",
                "reason": (f"Bridge {src}->{dst}: {res.steps}" if res.ok
                           else f"Bridge {src}->{dst} failed: {res.error}. {reason}"),
                "tx_id": res.tx_id})
        else:
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "asset_in": "USDC",
                "asset_out": dst, "amount": amt, "status": "executed",
                "reason": f"[PAPER] Bridge {src}->{dst} of {amt} USDC simulated. {reason}"})

    def _onchain_usdc_balance(self, chain: str) -> float:
        """Real USDC balance on an EVM chain (arc: USDC IS the native gas
        token, so it is read as the native balance)."""
        if self._onchain_secrets is None:
            return 0.0
        try:
            from .bridge import burn_token_for
            from .onchain import EVMWallet, EVM_CHAINS
            if chain not in EVM_CHAINS:
                return 0.0
            w = EVMWallet(self._onchain_secrets, chain)
            if not w.configured():
                return 0.0
            if chain == "arc":
                return w.native_balance()
            token = burn_token_for(chain)
            return w.erc20_balance(token) if token else 0.0
        except Exception as e:
            log.warning("could not read USDC balance on %s: %s", chain, e)
            return 0.0

    def _resume_pending_bridges(self, cycle_id: str) -> None:
        """Complete CCTP mints whose attestation landed since the burn."""
        from .bridge import resume_pending_bridges
        try:
            results = resume_pending_bridges(self.db, self._onchain_secrets)
        except Exception as e:
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "status": "failed",
                "reason": f"Bridge resume error: {e}"})
            return
        for r in results:
            if r.get("ok"):
                reason = (f"Bridge completion: {r['amount']} USDC minted on "
                          f"{r['dst_chain']} (burned earlier on "
                          f"{r['src_chain']}). Funds investable this cycle.")
            else:
                reason = (f"Bridge {r['src_chain']}->{r['dst_chain']} still "
                          f"pending: {r.get('error', 'attestation not ready')}")
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "asset_in": "USDC",
                "asset_out": r.get("dst_chain", ""), "amount": r.get("amount"),
                "status": "executed" if r.get("ok") else "failed",
                "reason": reason, "tx_id": r.get("mint_tx", "")})

    def _execute_live(self, cycle_id: str, action: dict, asset: str, amount: float,
                      prices: dict[str, float] | None = None) -> str | None:
        if self.constraints.live_execution_mode == "demo":
            return self._execute_demo(cycle_id, action, asset, amount)
        if self.constraints.live_execution_mode == "onchain":
            return self._execute_onchain(cycle_id, action, asset, amount, prices)

    def _execute_demo(self, cycle_id: str, action: dict, asset: str, amount: float) -> str | None:
        """Demo-wallet execution: identical validation, real DexScreener
        prices, simulated settlement against the user-built demo wallet
        (gas deducted, reserve enforced)."""
        from .demowallet import DemoWallet
        demo = DemoWallet(self.db)
        token = action.get("_contract") or action["asset"]
        if not self._looks_like_address(token):
            return (f"demo/on-chain swap needs a token contract address for {token!r}; "
                    f"the AI must provide contract/mint addresses")
        assets = self.market.get_asset_map()
        # the AI may reference the token by symbol OR contract; resolve both
        info = assets.get(asset)
        if info is None:
            # asset is likely a contract: find it in the asset map
            for a in assets.values():
                if (a.get("contract") or "").lower() == token.lower():
                    info = a
                    break
        chain = (info or {}).get("chain") or self.constraints.onchain_chain
        prices = self.market.get_prices()
        # price by the resolved symbol when possible, else by contract
        price_to = prices.get(asset, 0.0)
        if price_to <= 0 and info:
            price_to = prices.get(info.get("asset", ""), 0.0)
        if price_to <= 0:
            try:
                addr_prices = self.market.get_prices_for_addresses([token])
                price_to = addr_prices.get(token.lower(), 0.0)
            except Exception:
                pass
        if price_to <= 0:
            return f"no reliable price for {asset}: refusing demo trade"
        ok, why = demo.gas_ok(chain)
        if not ok:
            return f"gas check failed on {chain}: {why}"
        # funding: prefer USDC on the chain; fall back to the investable
        # portion of the gas token (mirrors real wallets where ETH is the
        # spendable asset after the gas reserve)
        stable = "USDC"
        stable_price = prices.get(stable, 1.0) or 1.0
        cost_usd = amount * price_to
        usdc_qty = demo.balance(chain, stable)
        usd_have = usdc_qty * stable_price
        gas_symbol = demo._gas_symbol(chain)
        gas_price = prices.get(gas_symbol, 0.0)
        if gas_price <= 0 and gas_symbol == "USDC":
            gas_price = 1.0
        if gas_price <= 0:
            # gas token not in the symbol map (e.g. plain ETH): price it via
            # its wrapped contract, the same as every other view
            from .demowallet import NATIVE_WRAPPED
            w = NATIVE_WRAPPED.get(chain)
            if w:
                try:
                    addr_p = self.market.get_prices_for_addresses([w])
                    gas_price = addr_p.get(w.lower(), 0.0)
                except Exception:
                    pass
        investable_gas_qty = demo.investable_gas(chain)
        symbol = (info or {}).get("asset", asset)  # human symbol when known
        if action["type"] == "buy":
            if usd_have + 1e-9 >= cost_usd:
                ok, msg = demo.swap(chain, stable, "", symbol, token,
                                    cost_usd / max(stable_price, 1e-9),
                                    stable_price, price_to)
            elif gas_price > 0 and investable_gas_qty * gas_price + 1e-9 >= cost_usd:
                # spend investable gas token instead (e.g. ETH on base)
                ok, msg = demo.swap(chain, gas_symbol, "", symbol, token,
                                    cost_usd / gas_price, gas_price, price_to)
            else:
                return (f"demo execution failed on {chain}: insufficient funds "
                        f"(USDC ${usd_have:.2f}, investable {gas_symbol} "
                        f"${investable_gas_qty * gas_price:.2f} < ${cost_usd:.2f})")
        else:
            ok, msg = demo.swap(chain, symbol, token, stable, "",
                                amount, price_to, stable_price)
        if not ok:
            return f"demo execution failed on {chain}: {msg}"
        self.db.add_decision({
            "cycle_id": cycle_id, "action": action["type"], "asset_in": asset,
            "asset_out": stable, "amount": amount, "value_usd": amount * price_to,
            "status": "executed", "reason": f"[DEMO] {action['reason']}",
            "tx_id": f"demo-{chain}-{asset[:10]}",
        })
        return None

    def _execute_onchain(self, cycle_id: str, action: dict, asset: str, amount: float,
                         prices: dict[str, float] | None = None) -> str | None:
        """Mode B: swap on a DEX with the imported hot wallet. The AI works in
        token tickers; on-chain mode requires contract addresses/mints, so we
        look them up from the market snapshot's chain data."""
        from .onchain import EVM_CHAINS, wallet_for
        prices = prices or {}
        # swaps settle on the TOKEN'S chain (annotated at validation, where
        # the buy was sized against that chain's investable funds);
        # onchain_chain is only the fallback for un-annotated actions
        chain = action.get("_chain") or self.constraints.onchain_chain
        # network gate: swaps must run on the network matching the toggle
        # (sepolia/base-sepolia only in testnet mode; mainnet chains only
        # with testnet OFF - no solana devnet support exists)
        chain_is_testnet = EVM_CHAINS.get(chain, {}).get("testnet", False)
        if chain_is_testnet != self.constraints.onchain_testnet:
            return (f"{'testnet mode is ON' if self.constraints.onchain_testnet else 'testnet mode is OFF'}: "
                    f"swaps on {chain} are refused while the toggle disagrees "
                    f"(use {'sepolia or base-sepolia' if self.constraints.onchain_testnet else 'a mainnet chain'})")
        # SAFETY FIRST: the asset identifier must be a contract/mint address
        # before anything else (including gas checks) runs. Tickertapes are
        # resolved from the validator's annotation, the live holdings, then
        # the market asset map; only an unresolvable asset is refused.
        token = action["asset"]
        if not self._looks_like_address(token):
            token = self._resolve_contract(token, action)
            if not token:
                return (f"on-chain swap needs a token contract address for "
                        f"{action['asset']!r}; not found in market data, "
                        f"watchlist or wallet holdings")
            action["_contract"] = token
        try:
            return self._swap_onchain(cycle_id, action, asset, amount,
                                      token, chain, prices)
        except Exception as e:
            # a crashed swap must never kill the cycle: it becomes a failed
            # action row in the feed and the cycle continues
            return f"on-chain swap error on {chain}: {type(e).__name__}: {e}"

    def _swap_onchain(self, cycle_id, action, asset, amount, token, chain,
                      prices):
        from .onchain import ARC_USDC, EVM_CHAINS, wallet_for
        wallet = wallet_for(chain, self._onchain_secrets)
        if not wallet.configured():
            return f"on-chain mode selected but no {chain} private key imported"
        # gas pre-flight: can the wallet actually pay for this transaction?
        if hasattr(wallet, "gas_ok"):
            ok, why = wallet.gas_ok()
            if not ok:
                return f"gas check failed on {chain}: {why}"
        # slippage bound: every real swap carries an explicit amountOutMin
        # computed from market prices (never the router's default of 1)
        slip = self.constraints.onchain_slippage_pct / 100.0
        basis_buy = basis_sell = basis_move = None
        if action["type"] == "buy":
            token_price = self._token_price(action, asset, token, prices)
            if token_price <= 0:
                return f"no reliable price for {asset}: refusing on-chain buy"
            usd_budget = amount * token_price
            if chain in EVM_CHAINS:
                weth = wallet._weth()
                if weth:
                    native_price = self._native_price(chain, prices)
                    if native_price <= 0:
                        return (f"no reliable price for the {chain} gas token: "
                                f"refusing on-chain buy")
                    # amount is a token QUANTITY (shared resolver contract); the
                    # router needs the native-token budget that buys it
                    native_qty = usd_budget / native_price
                    res = wallet.swap_exact_eth_for_tokens(
                        token, native_qty, min_amount_out=amount * (1 - slip))
                else:
                    # Arc: USDC is the native gas token; ArcSwap pairs
                    # token<->USDC directly (the router takes no native value)
                    res = wallet.swap_exact_tokens_for_tokens(
                        ARC_USDC, token, usd_budget,
                        min_amount_out=amount * (1 - slip))
            else:
                sol_price = self._native_price(chain, prices)
                if sol_price <= 0:
                    return (f"no reliable price for the {chain} gas token: "
                            f"refusing on-chain buy")
                res = wallet.swap(self._native_mint(chain), token,
                                  usd_budget / sol_price,
                                  slippage_bps=int(self.constraints.onchain_slippage_pct * 100))
            basis_buy = (chain, token, amount, usd_budget)
        else:
            dest = action.get("asset_out") or ("native" if chain in EVM_CHAINS else self._native_mint(chain))
            if action["type"] == "swap" and dest and self._looks_like_address(dest):
                if chain in EVM_CHAINS:
                    min_out = self._min_out_for(action, asset, token, dest, amount, prices, slip)
                    if min_out is None:
                        return (f"no reliable price for {dest}: refusing "
                                f"on-chain swap without slippage protection")
                    res = wallet.swap_exact_tokens_for_tokens(token, dest, amount,
                                                              min_amount_out=min_out)
                    basis_move = (chain, token, amount, dest, min_out)
                else:
                    res = wallet.swap(token, dest, amount,
                                      slippage_bps=int(self.constraints.onchain_slippage_pct * 100))
                    basis_sell = (chain, token, amount)
            else:
                if chain in EVM_CHAINS:
                    weth = wallet._weth()
                    if not weth and chain == "arc":
                        weth = ARC_USDC
                    min_out = self._min_out_for(action, asset, token, weth, amount, prices, slip)
                    if min_out is None:
                        return (f"no reliable price for {weth or 'the gas token'}: "
                                f"refusing on-chain sell without slippage protection")
                    res = wallet.swap_exact_tokens_for_tokens(token, weth, amount,
                                                              min_amount_out=min_out)
                    basis_sell = (chain, token, amount)
                else:
                    res = wallet.swap(token, self._native_mint(chain), amount,
                                      slippage_bps=int(self.constraints.onchain_slippage_pct * 100))
                    basis_sell = (chain, token, amount)
        if not res.ok:
            return res.error
        self._record_live_basis(basis_buy, basis_sell, basis_move)
        self.db.add_decision({
            "cycle_id": cycle_id, "action": action["type"], "asset_in": asset,
            "asset_out": action.get("asset_out") or "native",
            "amount": amount, "status": "executed",
            "reason": action["reason"], "tx_id": res.tx_id,
        })
        return None

    def _record_live_basis(self, basis_buy, basis_sell, basis_move) -> None:
        """Live cost-basis ledger: real-mode P&L needs a per-token cost
        tracked as the bot trades. Quantities come from the pinned action
        amounts and the slippage-adjusted estimates - slippage makes them
        approximations of the on-chain fills, so the ledger records intent,
        not exact fills."""
        try:
            if basis_buy:
                chain, token, qty, cost = basis_buy
                self.db.live_basis_add(chain, token, qty, cost)
            if basis_sell:
                chain, token, qty = basis_sell
                self.db.live_basis_reduce(chain, token, qty)
            if basis_move:
                chain, src, qty_in, dest, qty_out = basis_move
                row = self.db.live_basis_get(chain, src)
                if row and row["quantity"] > 0:
                    cost_out = row["cost_usd"] * min(1.0, qty_in / row["quantity"])
                    self.db.live_basis_add(chain, dest, qty_out, cost_out)
                self.db.live_basis_reduce(chain, src, qty_in)
        except Exception:
            log.exception("live cost-basis ledger update failed")

    def _token_price(self, action: dict, asset: str, token: str,
                     prices: dict[str, float]) -> float:
        p = action.get("_price") or prices.get(asset, 0.0)
        if p > 0:
            return p
        try:
            addr_p = self.market.get_prices_for_addresses([token])
            return addr_p.get((token or "").lower(), 0.0)
        except Exception:
            return 0.0

    def _native_price(self, chain: str, prices: dict[str, float]) -> float:
        """USD price of the chain's gas token, priced via its wrapped
        contract when the symbol map misses (ETH on base, etc.)."""
        from .onchain import EVM_CHAINS
        if chain not in EVM_CHAINS:
            return prices.get("SOL", 0.0)
        sym = EVM_CHAINS[chain].get("native_symbol", "ETH")
        if sym == "USDC":
            return 1.0
        p = prices.get(sym, 0.0)
        if p > 0:
            return p
        w = EVM_CHAINS[chain].get("weth")
        if w:
            try:
                addr_p = self.market.get_prices_for_addresses([w])
                return addr_p.get(w.lower(), 0.0)
            except Exception:
                return 0.0
        return 0.0

    def _min_out_for(self, action: dict, asset: str, token_in: str,
                     token_out: str, qty_in: float, prices: dict[str, float],
                     slip: float) -> float | None:
        """Minimum acceptable output quantity for a token->token swap,
        derived from market prices with the slippage tolerance applied."""
        price_in = self._token_price(action, asset, token_in, prices)
        if price_in <= 0:
            return None
        price_out = self._token_price({}, token_out, token_out, prices)
        if price_out <= 0:
            return None
        return qty_in * price_in / price_out * (1 - slip)

    @staticmethod
    def _looks_like_address(s: str) -> bool:
        return bool(s) and (s.startswith("0x") and len(s) == 42 or len(s) in (32, 44) and not s.isupper() == s.islower())

    def _resolve_contract(self, symbol: str, action: dict) -> str:
        """Ticker -> contract address: the validator's annotation first, then
        the wallet's live holdings (discovered tokens), then the market asset
        map. Empty string when nothing resolves."""
        cand = action.get("_contract")
        if cand and self._looks_like_address(cand):
            return cand
        sym = (symbol or "").upper()
        for r in (getattr(self.validator, "live_holdings", None) or []):
            if r.get("asset", "").upper() == sym and r.get("contract"):
                return r["contract"]
        try:
            for a in self.market.get_asset_map().values():
                if a.get("asset", "").upper() == sym and a.get("contract"):
                    return a["contract"]
        except Exception:
            pass
        # last resort: live DexScreener search for brand-new tokens that are
        # not in the cached universe yet (contract from the best pair)
        try:
            if getattr(self.constraints, "auto_market_expansion", True):
                best = self.market.search_contract(
                    symbol, action.get("_chain") or self.constraints.onchain_chain)
                if best:
                    addr = (best.get("baseToken") or {}).get("address") or ""
                    if addr:
                        try:
                            self.market.note_asset(best)
                        except Exception:
                            pass
                        return addr
        except Exception:
            pass
        return ""

    @staticmethod
    def _native_mint(chain: str) -> str:
        return "So11111111111111111111111111111111111111112" if chain == "solana" else ""

    def _chain_of(self, asset: str) -> str | None:
        snap = self.db.latest_market_snapshot()
        if not snap:
            return None
        import json as _json
        for a in _json.loads(snap["payload"]):
            if a["asset"] == asset:
                return a.get("chain")
        return None

    def _resolve_amount(self, action: dict, prices: dict[str, float],
                        live_total_usd: float | None = None) -> float:
        """Convert absolute or 'N%' amount into an absolute asset quantity.
        Delegates to the SAME shared resolver the constraint validator uses,
        WITH the market module: demo percent buys funded from gas-only chains
        price the gas token via wrapped contracts, and without the market
        they resolve to 0 here after passing validation."""
        from .constraints import resolve_action_amount
        val = resolve_action_amount(
            action, prices, self.tracker, self.db,
            market=self.market, live_total_usd=live_total_usd,
            live_holdings=getattr(self.validator, "live_holdings", None))
        return val if val is not None else 0.0
