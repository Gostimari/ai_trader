"""Execution Module (§4.3) — Mode A: exchange/custodial API via CCXT.

- trade-only API keys (withdrawals disabled on the exchange side; this module
  never touches withdrawal endpoints)
- idempotent orders: deterministic clientOrderId per (cycle, action) so a
  retried cycle never double-executes
- balance reconciliation after every order: balances must move as expected
  or the action is reported as failed
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from .secrets import SecretStore

log = logging.getLogger("memicoin.executor")

SECRET_KEYS = ("exchange_id", "exchange_api_key", "exchange_api_secret")


@dataclass
class ExecutionResult:
    ok: bool
    tx_id: str = ""
    amount: float = 0.0
    value_usd: float = 0.0
    pnl: float = 0.0
    error: str = ""


class LiveExecutor:
    """Executes validated actions on a real exchange via CCXT."""

    def __init__(self, secrets: SecretStore):
        self.secrets = secrets
        self._exchange = None

    # ---------- connection ----------
    def configured(self) -> bool:
        return all(self.secrets.has(k) for k in SECRET_KEYS)

    def exchange_id(self) -> str:
        return (self.secrets.get("exchange_id") or "").lower()

    def _get_exchange(self):
        if self._exchange is not None:
            return self._exchange
        import ccxt
        exchange_class = getattr(ccxt, self.exchange_id())
        kwargs = {}
        if self.secrets.has("exchange_api_password"):
            kwargs["password"] = self.secrets.get("exchange_api_password")
        self._exchange = exchange_class({
            "apiKey": self.secrets.get("exchange_api_key"),
            "secret": self.secrets.get("exchange_api_secret"),
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }, **kwargs)
        return self._exchange

    # ---------- helpers ----------
    @staticmethod
    def _client_order_id(cycle_id: str, action: dict) -> str:
        seed = (f"{cycle_id}:{action.get('type')}:{action.get('asset')}:"
                f"{action.get('asset_out', '')}:{action.get('amount_or_percent')}")
        return "mcb-" + hashlib.sha256(seed.encode()).hexdigest()[:24]

    @staticmethod
    def _symbol(asset: str) -> str:
        return "USDC/USDT" if asset.upper() == "USDC" else f"{asset.upper()}/USDC"

    def _free_balance(self, ex, currency: str) -> float:
        balance = ex.fetch_balance()
        return float(balance.get(currency, {}).get("free") or 0.0)

    def _reconcile(self, ex, base: str, quote: str, pre_base: float, pre_quote: float,
                   side: str, filled: float, avg: float) -> str:
        """Balance reconciliation: fail loudly if balances didn't move as expected."""
        post_base = self._free_balance(ex, base)
        post_quote = self._free_balance(ex, quote)
        tol_b = max(1e-8, filled * 0.02)
        tol_q = max(0.01, filled * avg * 0.02)
        if side == "buy":
            if post_base - pre_base < filled - tol_b:
                return f"reconciliation failed: base {post_base} vs pre {pre_base} + filled {filled}"
            if pre_quote - post_quote < filled * avg - tol_q:
                return f"reconciliation failed: quote {post_quote} vs pre {pre_quote} - cost {filled * avg}"
        else:
            if pre_base - post_base < filled - tol_b:
                return f"reconciliation failed: base {post_base} vs pre {pre_base} - filled {filled}"
            if post_quote - pre_quote < filled * avg - tol_q:
                return f"reconciliation failed: quote {post_quote} vs pre {pre_quote} + proceeds {filled * avg}"
        return ""

    # ---------- public API ----------
    def _order(self, cycle_id: str, action: dict, side: str) -> ExecutionResult:
        symbol = self._symbol(action["asset"])
        base, quote = symbol.split("/")
        try:
            ex = self._get_exchange()
        except Exception as e:
            return ExecutionResult(False, error=f"exchange init: {e}")
        coid = self._client_order_id(cycle_id, action)
        try:
            pre_base = self._free_balance(ex, base)
            pre_quote = self._free_balance(ex, quote)
            ticker = ex.fetch_ticker(symbol)
            price = float(ticker.get("last") or 0.0)
            if price <= 0:
                return ExecutionResult(False, error=f"no price for {symbol}")
            if side == "buy":
                cost = float(action["amount"]) * price
                if cost > pre_quote:
                    return ExecutionResult(False, error=f"insufficient {quote}: need ${cost:.2f}, have ${pre_quote:.2f}")
                amount = cost  # market buy expressed in quote currency
            else:
                amount = min(float(action["amount"]), pre_base)
                if amount <= 0:
                    return ExecutionResult(False, error=f"no {base} balance to sell")
            order = ex.create_order(symbol, "market", side, amount,
                                    params={"clientOrderId": coid})
            filled = float(order.get("filled") or 0.0)
            # market buys fill in quote; normalize to base units
            if side == "buy":
                avg = float(order.get("average") or price)
                filled_base = filled / avg if avg > 0 else 0.0
            else:
                filled_base = filled
                avg = float(order.get("average") or price)
            if filled_base <= 0:
                return ExecutionResult(False, error=f"order not filled (status={order.get('status')})")
            err = self._reconcile(ex, base, quote, pre_base, pre_quote, side,
                                  filled_base if side == "sell" else filled_base, avg)
            if err:
                log.error("%s %s: %s", side, symbol, err)
                return ExecutionResult(False, tx_id=order.get("id", coid), error=err)
            return ExecutionResult(True, tx_id=order.get("id", coid),
                                   amount=filled_base, value_usd=filled_base * avg)
        except Exception as e:
            log.exception("live %s failed", side)
            return ExecutionResult(False, error=str(e))

    def buy(self, cycle_id: str, action: dict) -> ExecutionResult:
        return self._order(cycle_id, action, "buy")

    def sell(self, cycle_id: str, action: dict) -> ExecutionResult:
        return self._order(cycle_id, action, "sell")

    def fetch_portfolio_snapshot(self) -> dict:
        """Live balances + prices for dashboard/P&L in live mode."""
        ex = self._get_exchange()
        balance = ex.fetch_balance()
        out = {"cash_usd": 0.0, "positions": []}
        for currency, total in (balance.get("total") or {}).items():
            total = float(total or 0.0)
            if total <= 0:
                continue
            if currency in ("USDC", "USDT", "USD"):
                out["cash_usd"] += total
                continue
            price = 0.0
            try:
                ticker = ex.fetch_ticker(f"{currency}/USDC")
                price = float(ticker.get("last") or 0.0)
            except Exception:
                pass
            out["positions"].append({
                "asset": currency, "chain": None, "quantity": total,
                "price_usd": price, "value_usd": total * price,
            })
        return out
