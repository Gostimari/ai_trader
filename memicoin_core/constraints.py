"""Deterministic constraint enforcement (§7 of the brief).

The LLM proposes; this module disposes. Every AI-proposed action is validated
here BEFORE execution. Nothing in this module trusts the model.
"""
from __future__ import annotations

from dataclasses import dataclass

from .portfolio import PortfolioTracker
from .settings import Constraints

STABLECOIN = "USDC"


@dataclass
class ValidationResult:
    ok: bool
    reason: str


def resolve_action_amount(action: dict, prices: dict[str, float],
                           tracker, db, market=None) -> float | None:
    """Resolve an action's amount_or_percent into absolute asset units.
    Raises nothing; returns None when unresolvable."""
    raw = action.get("amount")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    a = action.get("amount_or_percent")
    if isinstance(a, (int, float)):
        return float(a) if a > 0 else None
    if not isinstance(a, str) or not a.strip():
        return None
    s = a.strip()
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    try:
        val = float(s)
    except ValueError:
        return None
    if val <= 0:
        return None
    if not is_pct:
        return val
    pct = val / 100.0
    asset = action.get("asset", "")
    if action.get("type") == "buy":
        # resolve against the funds that will actually pay:
        # paper mode -> paper portfolio; demo mode -> demo wallet spendables
        total = tracker.total_value(prices)
        price = prices.get(asset, 0.0)
        if getattr(db, "demo_mode_percent", False):
            from .demowallet import DemoWallet, NATIVE_WRAPPED
            demo = DemoWallet(db)
            chain = action.get("_chain", "")
            # price gas tokens via wrapped contracts (ETH not in symbol map)
            wrapped_needed = [NATIVE_WRAPPED[r["chain"]] for r in demo.rows()
                             if r["is_gas"] and r["chain"] in NATIVE_WRAPPED
                             and r["chain"] != "arc"]
            wrapped_prices: dict = {}
            if wrapped_needed and market is not None:
                try:
                    wrapped_prices = market.get_prices_for_addresses(wrapped_needed)
                except Exception:
                    wrapped_prices = {}
            spendable_usd = 0.0
            for r in demo.rows():
                if chain and r["chain"] != chain:
                    continue  # fund from the token's chain when known; else all
                if r["is_gas"]:
                    if r["chain"] == "arc":
                        p = 1.0
                    else:
                        w = NATIVE_WRAPPED.get(r["chain"])
                        p = prices.get(r["asset"], 0.0)
                        if p <= 0 and w:
                            p = wrapped_prices.get(w.lower(), 0.0)
                    spendable_usd += demo.investable_gas(r["chain"]) * p
                else:
                    p = prices.get(r["asset"], 0.0) or (1.0 if r["asset"] in ("USDC", "USDT") else 0.0)
                    spendable_usd += r["balance"] * p
            # the price may come annotated (contract-keyed lookups miss)
            price = action.get("_price") or price
            budget = spendable_usd * pct
            return budget / price if price > 0 else None
        budget = total * pct
        return budget / price if price > 0 else None
    pos = db.get_position(asset)
    if not pos or pos["quantity"] <= 0:
        # demo mode: sells resolve against the demo wallet holding
        if getattr(db, "demo_mode_percent", False):
            from .demowallet import DemoWallet
            demo = DemoWallet(db)
            for r in demo.rows():
                if r["asset"] == asset and r["balance"] > 0:
                    return r["balance"] * pct
        return None
    return pos["quantity"] * pct


class ConstraintValidator:
    def __init__(self, db, constraints: Constraints, solana_key_present: bool = False,
                 market=None):
        self.db = db
        self.constraints = constraints
        self.tracker = PortfolioTracker(db)
        self._solana_key = solana_key_present
        self._market = market

    def _market_addr_prices(self, contracts):
        if self._market is None:
            return {}
        return self._market.get_prices_for_addresses(contracts)

    def check_stop_conditions(self, prices: dict[str, float]) -> ValidationResult:
        c = self.constraints
        if c.emergency_stopped:
            return ValidationResult(False, "Emergency stop is active; all trading halted.")
        if c.trading_paused:
            return ValidationResult(False, "Trading is paused by the user.")
        # mode-aware valuation: paper books for paper mode; the live/demo
        # wallet value otherwise (the paper books are irrelevant there)
        if getattr(self.db, "demo_mode_percent", False):
            # demo mode: total = demo wallet value; start = today's first-seen
            # value is unknown, so drawdown vs cost basis is not measurable the
            # same way - the per-position stop and manual stops still apply.
            from .demowallet import DemoWallet
            demo = DemoWallet(self.db)
            total = 0.0
            for r in demo.rows():
                p = prices.get(r["asset"].upper(), 0.0)
                if p <= 0 and r["asset"].upper() in ("USDC", "USDT"):
                    p = 1.0
                total += r["balance"] * p
            pnl_pct = 0.0  # no external cost basis for demo funds
        else:
            summary = self.tracker.summary(prices)
            start = c.starting_capital_usd
            pnl_pct = ((summary["total_value_usd"] - start) / start * 100) if start > 0 else 0.0
        if c.max_loss_pct is not None and pnl_pct <= -c.max_loss_pct:
            return ValidationResult(False, f"Max loss stop hit ({pnl_pct:.1f}% <= -{c.max_loss_pct}%). Trading stopped.")
        if c.profit_target_pct is not None and pnl_pct >= c.profit_target_pct:
            return ValidationResult(False, f"Profit target reached ({pnl_pct:.1f}% >= {c.profit_target_pct}%). Trading stopped.")
        return ValidationResult(True, "")

    def validate_action(self, action: dict, prices: dict[str, float],
                        known_assets: dict[str, dict]) -> ValidationResult:
        """Validate one AI-proposed action against all hard constraints."""
        c = self.constraints
        stop = self.check_stop_conditions(prices)
        if not stop.ok:
            return stop

        a_type = action.get("type")
        asset = action.get("asset", "")
        # normalize symbol-or-contract to the SYMBOL first, so percent
        # amounts resolve against holdings whichever way the AI named it
        if asset and asset not in known_assets and known_assets.get(asset) is None:
            match = next((cand["asset"] for cand in known_assets.values()
                          if (cand.get("contract") or "").lower() == asset.lower()), None)
            if match:
                orig = asset
                asset = match
                action["asset"] = asset
                action["_contract"] = orig
        amount = self._resolve_amount(action, prices)

        if a_type not in ("buy", "sell", "swap"):
            return ValidationResult(False, f"Unknown action type: {a_type!r}")

        if a_type == "swap":
            dest = action.get("asset_out") or action.get("swap_to") or ""
            # resolve both assets by symbol OR contract (the AI may use either)
            resolved = []
            for a in (asset, dest):
                if a in known_assets:
                    resolved.append(a)
                    continue
                match = next((cand["asset"] for cand in known_assets.values()
                              if (cand.get("contract") or "").lower() == a.lower()), None)
                if match is None:
                    r = self._validate_asset(a, known_assets)
                    if not r.ok:
                        return r
                    resolved.append(a)
                else:
                    resolved.append(match)
            action["asset"], action["_swap_dest"] = resolved[0], resolved[1]
            if resolved[1] in known_assets and known_assets[resolved[1]].get("contract"):
                action["_swap_dest_contract"] = known_assets[resolved[1]]["contract"]
            if resolved[0] in known_assets and known_assets[resolved[0]].get("contract"):
                action["_contract"] = known_assets[resolved[0]]["contract"]
            # the swap SELLS the held asset: resolve the amount against holdings
            if amount is None or amount <= 0:
                return ValidationResult(False, "Missing or non-positive amount.")
            return ValidationResult(True, "")

        # resolve asset: the AI may reference by symbol OR contract address.
        # Keep BOTH: the symbol for validation/pricing, the contract for the
        # executor (which requires the contract to identify the token).
        info = known_assets.get(asset)
        if info is None:
            for a in known_assets.values():
                if (a.get("contract") or "").lower() == asset.lower():
                    info = a
                    action["_contract"] = asset  # preserve for the executor
                    asset = a["asset"]
                    action["asset"] = asset
                    break
        elif info.get("contract"):
            action["_contract"] = info["contract"]
        r = self._validate_asset(asset, known_assets)
        if not r.ok:
            return r

        if amount is None or amount <= 0:
            if getattr(self.db, "demo_mode_percent", False) and a_type == "buy":
                return ValidationResult(
                    False,
                    f"No investable funds on the token's chain for this buy "
                    f"(after the gas reserve); the position size resolved to "
                    f"{amount if amount is not None else 'nothing'}.")
            return ValidationResult(False, "Missing or non-positive amount.")

        if a_type == "buy":
            price = prices.get(asset)
            if price is None or price <= 0:
                price = action.get("_price") or 0.0
            if price is None or price <= 0:
                return ValidationResult(False, f"No reliable price for {asset}; refusing to buy.")
            cost = amount * price
            # portfolio total: paper books in paper mode; demo wallet otherwise
            if getattr(self.db, "demo_mode_percent", False):
                from .demowallet import DemoWallet, NATIVE_WRAPPED
                demo = DemoWallet(self.db)
                # gas tokens price via wrapped contracts (not in symbol map)
                wrapped_needed = [NATIVE_WRAPPED[r["chain"]] for r in demo.rows()
                                 if r["is_gas"] and r["chain"] in NATIVE_WRAPPED
                                 and r["chain"] != "arc"]
                wrapped_prices: dict = {}
                if wrapped_needed:
                    try:
                        wrapped_prices = self._market_addr_prices(wrapped_needed)
                    except Exception:
                        wrapped_prices = {}
                total = 0.0
                held_qty = 0.0
                for r in demo.rows():
                    p = prices.get(r["asset"].upper(), 0.0)
                    if p <= 0 and r["asset"].upper() in ("USDC", "USDT"):
                        p = 1.0
                    if p <= 0 and r["is_gas"]:
                        w = NATIVE_WRAPPED.get(r["chain"])
                        if r["chain"] == "arc":
                            p = 1.0
                        elif w:
                            p = wrapped_prices.get(w.lower(), 0.0)
                    total += r["balance"] * p
                    if r["asset"] == asset or r.get("contract", "").lower() == (action.get("asset") or "").lower():
                        held_qty = r["balance"]
                existing_value = held_qty * price
            else:
                summary = self.tracker.summary(prices)
                total = summary["total_value_usd"]
                existing = self.db.get_position(asset)
                existing_value = (existing["quantity"] * price) if existing else 0.0
            max_usd = total * c.effective_max_position_pct() / 100
            if existing_value + cost > max_usd + 1e-9:
                # CLAMP instead of reject: buy the most the constraint allows
                room_usd = max_usd - existing_value
                if room_usd <= 0:
                    return ValidationResult(
                        False,
                        f"Position in {asset} is already at the max size "
                        f"({c.effective_max_position_pct():.0f}% = ${max_usd:.2f}).",
                    )
                clamped_qty = room_usd / price
                if clamped_qty <= 0 or clamped_qty * price < 1e-9:
                    return ValidationResult(
                        False,
                        f"Max position for {asset} leaves an unusable amount (${room_usd:.4f}).",
                    )
                cash = self.db.get_cash()
                if room_usd > cash:
                    room_usd = cash
                    clamped_qty = cash / price
                    if clamped_qty <= 0:
                        return ValidationResult(
                            False, f"Buy exceeds available cash ${cash:.2f} and no room remains.")
                # adjust the action in place; the executor uses the clamped amount
                action["amount"] = clamped_qty
                action["amount_or_percent"] = clamped_qty
                action["_clamped"] = (f"AI asked for {amount:.6f} {asset} "
                                      f"(${cost:.2f}); clamped to {clamped_qty:.6f} "
                                      f"(${clamped_qty * price:.2f}) to respect the "
                                      f"{c.effective_max_position_pct():.0f}% position limit.")
                return ValidationResult(True, "")
        elif a_type == "sell":
            pos = self.db.get_position(asset)
            if not pos or pos["quantity"] <= 0:
                return ValidationResult(False, f"No position in {asset} to sell.")
            if amount > pos["quantity"] + 1e-9:
                return ValidationResult(False, f"Sell amount exceeds holdings of {asset}.")
        return ValidationResult(True, "")

    def _validate_asset(self, asset: str, known_assets: dict[str, dict]) -> ValidationResult:
        c = self.constraints
        if not asset:
            return ValidationResult(False, "Missing asset.")
        if asset == STABLECOIN:
            return ValidationResult(True, "")
        info = known_assets.get(asset)
        if info is None:
            return ValidationResult(False, f"Asset {asset!r} not found in market data.")
        chain = info.get("chain") or info.get("platform") or ""
        # Solana requires an imported Solana key: without one the app cannot
        # sign there, so Solana assets are rejected regardless of settings.
        if chain.lower() == "solana" and not self.has_solana_key():
            return ValidationResult(
                False,
                "Solana asset rejected: no Solana wallet key imported "
                "(import one in Settings, or remove solana from allowed chains).",
            )
        if c.autonomy_mode == "constrained":
            allowed = [a.lower() for a in c.allowed_chains]
            if chain and chain.lower() not in allowed:
                return ValidationResult(
                    False,
                    f"Chain {chain!r} for {asset} is not in allowed chains {allowed}.",
                )
        return ValidationResult(True, "")

    def has_solana_key(self) -> bool:
        if self._solana_key is None:
            return False
        return self._solana_key

    def _resolve_amount(self, action: dict, prices: dict[str, float]) -> float | None:
        """Single source of truth for how much an action means, in ABSOLUTE
        asset units. Used by BOTH validation and execution so the constraint
        always checks the exact amount that will be traded.
        Percent strings: buys = % of current portfolio value;
        sells/swaps = % of current holdings of the asset."""
        return resolve_action_amount(action, prices, self.tracker, self.db,
                                     market=self._market)
