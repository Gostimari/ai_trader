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
  become investable on the NEXT cycle — so when the best opportunity is
  on a chain where the wallet has no USDC, propose the bridge this cycle
  and the investment in the following one. Never propose buying on a
  chain with no funds there; bridge first.
- SOLANA: only propose Solana assets when the context says a Solana key
  is imported; otherwise Solana actions are rejected.
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
                                             solana_key_present=solana_present)

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
                    self._execute_bridge(cycle_id, a)
                    continue
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
        try:
            amt = float(str(amt_raw).rstrip("%"))
        except (TypeError, ValueError):
            amt = 0.0
        reason = action.get("reason", "")
        if not src or not dst or amt <= 0:
            self.db.add_decision({
                "cycle_id": cycle_id, "action": "bridge", "status": "rejected",
                "reason": f"Malformed bridge proposal: {src}->{dst} {amt_raw!r}. {reason}"})
            return
        if not self.constraints.paper_trading:
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
        token = action["asset"]
        if not self._looks_like_address(token):
            return (f"demo/on-chain swap needs a token contract address for {token!r}; "
                    f"the AI must provide contract/mint addresses")
        assets = self.market.get_asset_map()
        info = assets.get(asset)
        chain = (info or {}).get("chain") or self.constraints.onchain_chain
        prices = self.market.get_prices()
        price_to = prices.get(asset, 0.0)
        if price_to <= 0:
            return f"no reliable price for {asset}: refusing demo trade"
        ok, why = demo.gas_ok(chain)
        if not ok:
            return f"gas check failed on {chain}: {why}"
        stable = "USDC"
        stable_price = prices.get(stable, 1.0) or 1.0
        if action["type"] == "buy":
            cost_usd = amount * price_to
            ok, msg = demo.swap(chain, stable, "", asset, token,
                                cost_usd / max(stable_price, 1e-9),
                                stable_price, price_to)
        else:
            ok, msg = demo.swap(chain, asset, token, stable, "",
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
