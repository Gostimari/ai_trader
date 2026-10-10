"""Portfolio & P&L tracker: holdings, cost basis, realized/unrealized P&L in USD."""
from __future__ import annotations

from .db import Database, utcnow


class PortfolioTracker:
    def __init__(self, db: Database):
        self.db = db

    # ---------- valuation ----------
    def position_values(self, prices: dict[str, float]) -> list[dict]:
        """Attach current value & unrealized P&L to each position."""
        out = []
        for p in self.db.get_positions():
            price = prices.get(p["asset"], 0.0)
            value = p["quantity"] * price
            cost = p["cost_basis_usd"]
            out.append({
                "asset": p["asset"],
                "chain": p["chain"],
                "quantity": p["quantity"],
                "cost_basis_usd": cost,
                "price_usd": price,
                "value_usd": value,
                "unrealized_pnl_usd": value - cost,
                "unrealized_pnl_pct": ((value - cost) / cost * 100) if cost > 0 else 0.0,
                "updated_at": p["updated_at"],
            })
        return out

    def total_value(self, prices: dict[str, float]) -> float:
        return self.db.get_cash() + sum(
            p["quantity"] * prices.get(p["asset"], 0.0) for p in self.db.get_positions()
        )

    def total_cost_basis(self) -> float:
        return self.db.get_cash() + sum(
            p["cost_basis_usd"] for p in self.db.get_positions()
        )

    def summary(self, prices: dict[str, float]) -> dict:
        cash = self.db.get_cash()
        positions = self.position_values(prices)
        total = cash + sum(p["value_usd"] for p in positions)
        cost = cash + sum(p["cost_basis_usd"] for p in positions)
        pnl = total - cost
        return {
            "cash_usd": cash,
            "positions": positions,
            "total_value_usd": total,
            "total_cost_basis_usd": cost,
            "total_pnl_usd": pnl,
            "total_pnl_pct": (pnl / cost * 100) if cost > 0 else 0.0,
            "as_of": utcnow(),
        }

    # ---------- trade application (paper mode) ----------
    def apply_buy(self, asset: str, chain: str | None, quantity: float, cost_usd: float) -> None:
        self.db.set_cash(self.db.get_cash() - cost_usd)
        existing = self.db.get_position(asset)
        if existing:
            total_qty = existing["quantity"] + quantity
            total_cost = existing["cost_basis_usd"] + cost_usd
            self.db.upsert_position(asset, chain or existing["chain"], total_qty, total_cost)
        else:
            self.db.upsert_position(asset, chain, quantity, cost_usd)

    def apply_sell(self, asset: str, quantity: float, proceeds_usd: float) -> float:
        """Returns realized P&L for the sold portion."""
        existing = self.db.get_position(asset)
        if not existing or quantity <= 0:
            return 0.0
        qty = min(quantity, existing["quantity"])
        avg_cost = existing["cost_basis_usd"] / existing["quantity"]
        realized = proceeds_usd - qty * avg_cost
        remaining_qty = existing["quantity"] - qty
        remaining_cost = existing["cost_basis_usd"] - qty * avg_cost
        self.db.set_cash(self.db.get_cash() + proceeds_usd)
        if remaining_qty <= 1e-12:
            self.db.delete_position(asset)
        else:
            self.db.upsert_position(asset, existing["chain"], remaining_qty, remaining_cost)
        return realized
