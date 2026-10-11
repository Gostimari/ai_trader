"""Demo wallet: a user-populated fake wallet the bot trades like a real one.

Everything mirrors the real-money flow — real DexScreener prices, gas
reserve enforcement, simulated gas costs deducted from the demo gas
balance, position-size and chain constraints — except no blockchain
transaction is signed. This is the realistic bridge between paper mode
(default $100 virtual) and real money.
"""
from __future__ import annotations

import logging

from .db import Database

log = logging.getLogger("memicoin.demo")

# Gas cost model per chain (in the chain's gas token), conservative:
# 250k gas at the chain's typical gas price. Arc: USDC at 20 gwei floor.
GAS_COST_PER_TX = {
    "ethereum": 0.0075,   # 250k * 30 gwei
    "base": 0.00015,      # 250k * ~0.6 gwei (cheap L2)
    "polygon": 0.015,     # 250k * 60 gwei in POL
    "binance-smart-chain": 0.00015,  # 250k * ~0.6 gwei BNB
    "arc": 0.005,         # 250k * 20 gwei (floor) in USDC
    "solana": 0.00025,    # ~0.00005 SOL fee x5 buffer for swaps+fees
}
RESERVE_TXS = 3  # mirror EVMWallet.RESERVE_SWAPS


class DemoWallet:
    def __init__(self, db: Database):
        self.db = db

    # ---------- reads ----------
    def rows(self) -> list[dict]:
        return self.db.demo_rows()

    def balance(self, chain: str, asset: str, contract: str = "") -> float:
        r = self.db.demo_get(chain, asset, contract)
        return r["balance"] if r else 0.0

    def gas_balance(self, chain: str) -> float:
        rows = [r for r in self.rows()
                if r["chain"] == chain and r["is_gas"]]
        return sum(r["balance"] for r in rows)

    def per_tx_gas(self, chain: str) -> float:
        return GAS_COST_PER_TX.get(chain, 0.0075)

    def gas_reserve(self, chain: str) -> float:
        return self.per_tx_gas(chain) * RESERVE_TXS

    def investable_gas(self, chain: str) -> float:
        return max(0.0, self.gas_balance(chain) - self.gas_reserve(chain))

    def gas_ok(self, chain: str) -> tuple[bool, str]:
        bal = self.gas_balance(chain)
        if bal < self.per_tx_gas(chain):
            sym = {"arc": "USDC", "solana": "SOL"}.get(chain, "ETH")
            return False, (f"insufficient gas on {chain}: {bal:.6f} < "
                           f"{self.per_tx_gas(chain):.6f}. Top up {sym} in the demo wallet.")
        return True, ""

    # ---------- execution (mirrors the real path) ----------
    def swap(self, chain: str, from_asset: str, from_contract: str,
             to_asset: str, to_contract: str, from_amount: float,
             price_from: float, price_to: float) -> tuple[bool, str]:
        """Simulated swap at REAL prices. Deducts the sold amount and the
        gas cost from the demo wallet; credits the bought amount."""
        # gas first (like reality: the tx burns gas regardless)
        ok, why = self.gas_ok(chain)
        if not ok:
            return False, why
        have = self.balance(chain, from_asset, from_contract)
        if have + 1e-12 < from_amount:
            return False, (f"insufficient {from_asset} on {chain}: "
                           f"{have} < {from_amount}")
        usd = from_amount * price_from
        if price_to <= 0:
            return False, f"no price for {to_asset}: cannot size the buy"
        bought = usd / price_to
        # deduct gas from the gas row (fails if it would go negative -> refused)
        gas_cost = self.per_tx_gas(chain)
        new_gas = self.db.demo_adjust(chain, self._gas_symbol(chain), -gas_cost,
                                      is_gas=True)
        if abs(new_gas - self.gas_balance(chain)) < 1e-12 and new_gas < gas_cost:
            return False, f"gas deduction failed on {chain}"
        # deduct sold asset, credit bought asset
        self.db.demo_adjust(chain, from_asset, -from_amount, from_contract)
        self.db.demo_adjust(chain, to_asset, bought, to_contract)
        return True, (f"swapped {from_amount} {from_asset} -> {bought:.6f} {to_asset} "
                      f"(${usd:.2f} at real prices); gas {gas_cost}")

    def _gas_symbol(self, chain: str) -> str:
        return {"arc": "USDC", "solana": "SOL"}.get(chain, "ETH")

    def bridge(self, src_chain: str, dst_chain: str, asset: str, amount: float) -> tuple[bool, str]:
        """Simulated CCTP bridge: deduct on src (with gas), credit on dst
        after the reserve on the destination is respected."""
        ok, why = self.gas_ok(src_chain)
        if not ok:
            return False, why
        have = self.balance(src_chain, asset)
        if have + 1e-12 < amount:
            return False, f"insufficient {asset} on {src_chain}"
        gas_cost = self.per_tx_gas(src_chain)
        self.db.demo_adjust(src_chain, self._gas_symbol(src_chain), -gas_cost, is_gas=True)
        self.db.demo_adjust(src_chain, asset, -amount)
        self.db.demo_adjust(dst_chain, asset, amount)
        return True, (f"bridged {amount} {asset} {src_chain}->{dst_chain}; "
                     f"gas {gas_cost} on {src_chain}. Available for the next cycle.")
