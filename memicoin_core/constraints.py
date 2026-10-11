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
                           tracker, db) -> float | None:
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
        total = tracker.total_value(prices)
        price = prices.get(asset, 0.0)
        budget = total * pct
        return budget / price if price > 0 else None
    pos = db.get_position(asset)
    if not pos or pos["quantity"] <= 0:
        return None
    return pos["quantity"] * pct


class ConstraintValidator:
    def __init__(self, db, constraints: Constraints, solana_key_present: bool = False):
        self.db = db
        self.constraints = constraints
        self.tracker = PortfolioTracker(db)
        self._solana_key = solana_key_present

    def check_stop_conditions(self, prices: dict[str, float]) -> ValidationResult:
        c = self.constraints
        if c.emergency_stopped:
            return ValidationResult(False, "Emergency stop is active; all trading halted.")
        if c.trading_paused:
            return ValidationResult(False, "Trading is paused by the user.")
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
        amount = self._resolve_amount(action, prices)

        if a_type not in ("buy", "sell", "swap"):
            return ValidationResult(False, f"Unknown action type: {a_type!r}")

        if a_type == "swap":
            dest = action.get("asset_out") or action.get("swap_to") or ""
            for a in (asset, dest):
                r = self._validate_asset(a, known_assets)
                if not r.ok:
                    return r
            return ValidationResult(True, "")

        r = self._validate_asset(asset, known_assets)
        if not r.ok:
            return r

        if amount is None or amount <= 0:
            return ValidationResult(False, "Missing or non-positive amount.")

        if a_type == "buy":
            price = prices.get(asset)
            if price is None or price <= 0:
                return ValidationResult(False, f"No reliable price for {asset}; refusing to buy.")
            cost = amount * price
            cash = self.db.get_cash()
            if cost > cash:
                return ValidationResult(False, f"Buy of ${cost:.2f} exceeds available cash ${cash:.2f}.")
            summary = self.tracker.summary(prices)
            total = summary["total_value_usd"]
            max_usd = total * c.effective_max_position_pct() / 100
            existing = self.db.get_position(asset)
            existing_value = (existing["quantity"] * price) if existing else 0.0
            if existing_value + cost > max_usd + 1e-9:
                return ValidationResult(
                    False,
                    f"Max position size {c.effective_max_position_pct():.0f}% "
                    f"(${max_usd:.2f}) would be exceeded for {asset}.",
                )
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
        return resolve_action_amount(action, prices, self.tracker, self.db)
