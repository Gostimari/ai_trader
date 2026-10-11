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
from .portfolio import PortfolioTracker
from .settings import Constraints

MAX_ACTIONS_PER_CYCLE = 6


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
- amount_or_percent: prefer an absolute quantity of the asset; you may use "N%" meaning N% of current portfolio value for buys, or N% of current holdings for sells.
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
- The execution layer will reject anything violating the user's constraints; do not propose violations.
- Cash is held in USD (paper mode). To exit a position, sell it.
"""


class DecisionEngine:
    def __init__(self, db: Database, constraints: Constraints,
                 market: MarketDataModule, news=None, live_executor=None,
                 onchain_secrets=None):
        self.db = db
        self.constraints = constraints
        self.market = market
        self.news = news  # unused; retained for API compatibility
        self.live_executor = live_executor
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
        cycle_id = self.db.start_cycle()
        # mode flag FIRST: stop checks and amount resolution must both be
        # demo-aware from the very beginning of the cycle
        self.db.demo_mode_percent = (not self.constraints.paper_trading
                                     and self.constraints.live_execution_mode == "demo")
        try:
            prices = self.market.get_prices(force=True)
            assets = self.market.get_asset_map(force=True)
            portfolio = self.tracker.summary(prices)
            if (not self.constraints.paper_trading
                    and self.constraints.live_execution_mode == "demo"):
                portfolio = self._demo_portfolio_summary(prices)
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
                err = self._execute(cycle_id, a, prices)
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
        # investable gas per chain (balance minus reserve)
        for chain, entries in by_chain.items():
            inv = demo.investable_gas(chain)
            reserve = demo.gas_reserve(chain)
            per_swap = demo.per_tx_gas(chain)
            entries.append({"_meta": True, "note":
                            f"investable gas on {chain}: {inv:.6f} "
                            f"(reserve {reserve:.6f} = {per_swap:.6f}/tx kept for fees)"})
        return {"demo_wallet": True, "chains": by_chain,
                "total_value_usd": total,
                "note": "Portfolio is the DEMO wallet. Fund buys with USDC "
                        "where available, or with investable gas-token amounts. "
                        "Buys settle against these balances (simulated)."}

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
        from .bridge import CCTP_DOMAINS
        parts.append(
            "## Wallet capabilities\n"
            f"- Solana key imported: {solana_key} "
            f"({'Solana assets allowed' if solana_key == 'yes' else 'do NOT propose Solana assets'})\n"
            f"- CCTP-bridgeable USDC chains: {', '.join(CCTP_DOMAINS)} (bridge action available)")
        # gas reserves: what's investable per chain (real mode only)
        if not self.constraints.paper_trading and self._onchain_secrets is not None \
                and self._onchain_secrets.has("onchain_private_key"):
            try:
                from .onchain import EVMWallet, EVM_CHAINS
                lines = []
                for chain in EVM_CHAINS:
                    try:
                        w = EVMWallet(self._onchain_secrets, chain)
                        lines.append(
                            f"  {chain}: {w.investable_native():.6f} {EVM_CHAINS[chain].get('native_symbol', 'ETH')} "
                            f"investable (reserve {w.gas_reserve():.6f} kept for gas)")
                    except Exception:
                        continue
                if lines:
                    parts.append("## Gas reserve (NEVER spend into the reserve)\n"
                                 "A gas reserve is held back on each chain for future "
                                 "transaction fees. Only the 'investable' amount may "
                                 "be traded:\n" + "\n".join(lines))
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
    def _execute(self, cycle_id: str, action: dict, prices: dict[str, float]) -> str | None:
        """Execute one validated action. Returns error string or None on success."""
        asset = action["asset"]
        amount = self._resolve_amount(action, prices)
        price = prices.get(asset, 0.0)
        if amount <= 0:
            return f"resolved to non-positive amount for {asset}"

        if not self.constraints.paper_trading:
            return self._execute_live(cycle_id, action, asset, amount)

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

    def _execute_live(self, cycle_id: str, action: dict, asset: str, amount: float) -> str | None:
        if self.constraints.live_execution_mode == "demo":
            return self._execute_demo(cycle_id, action, asset, amount)
        if self.constraints.live_execution_mode == "onchain":
            return self._execute_onchain(cycle_id, action, asset, amount)

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
        if self.live_executor is None or not self.live_executor.configured():
            return "live mode selected but exchange API keys are not configured"
        order_action = {**action, "amount": amount}
        if action["type"] == "swap":
            # execute as sell + buy on the pair(s)
            dest = action.get("asset_out", "")
            sell_res = self.live_executor.sell(cycle_id, order_action)
            if not sell_res.ok:
                return sell_res.error
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "sell", "asset_in": asset, "asset_out": "USDC",
                "amount": sell_res.amount, "value_usd": sell_res.value_usd,
                "status": "executed", "reason": f"Swap leg 1/2: {action['reason']}",
                "tx_id": sell_res.tx_id,
            })
            buy_action = {**order_action, "type": "buy", "asset": dest,
                          "amount": sell_res.value_usd / max(self._live_price(dest), 1e-12)}
            buy_res = self.live_executor.buy(cycle_id, buy_action)
            if not buy_res.ok:
                return f"swap sell leg ok ({sell_res.tx_id}) but buy leg failed: {buy_res.error}"
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "buy", "asset_in": dest, "asset_out": "USDC",
                "amount": buy_res.amount, "value_usd": buy_res.value_usd,
                "status": "executed", "reason": f"Swap leg 2/2: {action['reason']}",
                "tx_id": buy_res.tx_id,
            })
            return None
        res = (self.live_executor.buy(cycle_id, order_action) if action["type"] == "buy"
               else self.live_executor.sell(cycle_id, order_action))
        if not res.ok:
            return res.error
        self.db.add_decision({
            "cycle_id": cycle_id, "action": action["type"], "asset_in": asset, "asset_out": "USDC",
            "amount": res.amount, "value_usd": res.value_usd, "status": "executed",
            "reason": action["reason"], "tx_id": res.tx_id,
        })
        return None

    def _execute_onchain(self, cycle_id: str, action: dict, asset: str, amount: float) -> str | None:
        """Mode B: swap on a DEX with the imported hot wallet. The AI works in
        token tickers; on-chain mode requires contract addresses/mints, so we
        look them up from the market snapshot's chain data."""
        from .onchain import EVM_CHAINS, wallet_for
        chain = self.constraints.onchain_chain
        # SAFETY FIRST: the asset identifier must be a contract/mint address
        # before anything else (including gas checks) runs.
        token = action["asset"]
        if not self._looks_like_address(token):
            return (f"on-chain swap needs a token contract address for {token!r}; "
                    f"the AI must provide contract/mint addresses in on-chain mode")
        wallet = wallet_for(chain, self._onchain_secrets)
        if not wallet.configured():
            return f"on-chain mode selected but no {chain} private key imported"
        # gas pre-flight: can the wallet actually pay for this transaction?
        if hasattr(wallet, "gas_ok"):
            ok, why = wallet.gas_ok()
            if not ok:
                return f"gas check failed on {chain}: {why}"
        # asset identifiers in on-chain mode are contract addresses/mints,
        # supplied by the AI as the asset field and validated against the
        # chain registry; a bare ticker cannot be traded on-chain.
        token = action["asset"]
        if not self._looks_like_address(token):
            return (f"on-chain swap needs a token contract address for {token!r}; "
                    f"the AI must provide contract/mint addresses in on-chain mode")
        if action["type"] == "buy":
            res = (wallet.swap_exact_eth_for_tokens(token, amount)
                   if chain in EVM_CHAINS else wallet.swap("native", token, amount))
        else:
            dest = action.get("asset_out") or ("native" if chain in EVM_CHAINS else self._native_mint(chain))
            if action["type"] == "swap" and dest and self._looks_like_address(dest):
                res = (wallet.swap_exact_tokens_for_tokens(token, dest, amount)
                       if chain in EVM_CHAINS else wallet.swap(token, dest, amount))
            else:
                res = (wallet.swap_exact_tokens_for_tokens(token, wallet._weth() or "", amount)
                       if chain in EVM_CHAINS else wallet.swap(token, self._native_mint(chain), amount))
        if not res.ok:
            return res.error
        self.db.add_decision({
            "cycle_id": cycle_id, "action": action["type"], "asset_in": asset,
            "asset_out": action.get("asset_out") or "native",
            "amount": amount, "status": "executed",
            "reason": action["reason"], "tx_id": res.tx_id,
        })
        return None

    @staticmethod
    def _looks_like_address(s: str) -> bool:
        return bool(s) and (s.startswith("0x") and len(s) == 42 or len(s) in (32, 44) and not s.isupper() == s.islower())

    @staticmethod
    def _native_mint(chain: str) -> str:
        return "So11111111111111111111111111111111111111112" if chain == "solana" else ""

    def _live_price(self, asset: str) -> float:
        try:
            snap = self.db.latest_market_snapshot()
            if snap:
                import json as _json
                for a in _json.loads(snap["payload"]):
                    if a["asset"] == asset.upper():
                        return a.get("price_usd", 0.0)
        except Exception:
            pass
        return 0.0

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
        """Convert absolute or 'N%' amount into an absolute asset quantity.
        Delegates to the SAME shared resolver the constraint validator uses,
        so what is validated is exactly what is executed."""
        from .constraints import resolve_action_amount
        val = resolve_action_amount(action, prices, self.tracker, self.db)
        return val if val is not None else 0.0
