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

# Wrapped-native contracts price native coins (ETH/POL/BNB) via DexScreener;
# verified live. Solana: wrapped SOL mint.
NATIVE_WRAPPED = {
    "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "base": "0x4200000000000000000000000000000000000006",
    "polygon": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",
    "binance-smart-chain": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",
    "solana": "So11111111111111111111111111111111111111112",
}
# On Arc USDC is native: priced via the peg in the callers.

GAS_COST_PER_TX = {
    "ethereum": 0.0075,   # 250k * 30 gwei
    "base": 0.00015,      # 250k * ~0.6 gwei (cheap L2)
    "polygon": 0.015,     # 250k * 60 gwei in POL
    "binance-smart-chain": 0.00015,  # 250k * ~0.6 gwei BNB
    "arc": 0.005,         # 250k * 20 gwei (floor) in USDC
    "solana": 0.00025,    # ~0.00005 SOL fee x5 buffer for swaps+fees
}
# Adaptive gas reserve (user design): scale the number of reserved swaps to
# the balance so small gas wallets keep an investable remainder.
#   balance < 2 swaps  -> reserve exactly 1 swap (minimum protection)
#   2-8 swaps          -> reserve half the swaps (rounded)
#   > 8 swaps          -> reserve 3 swaps (the old fixed cap)
# A wallet that can't even cover 1 swap reserves everything (investable 0)
# and the existing gas warnings fire.
RESERVE_SWAPS_CAP = 3
RESERVE_SWAPS_MIN = 1


def reserve_swaps_for(balance: float, per_swap: float) -> int:
    if per_swap <= 0 or balance <= 0:
        return RESERVE_SWAPS_MIN
    swaps_left = balance / per_swap
    if swaps_left < 2:
        return RESERVE_SWAPS_MIN
    return max(RESERVE_SWAPS_MIN,
               min(RESERVE_SWAPS_CAP, int(swaps_left / 2)))


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
        return self.per_tx_gas(chain) * reserve_swaps_for(
            self.gas_balance(chain), self.per_tx_gas(chain))

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
        # deduct sold asset, credit bought asset (cost-basis tracked)
        self.db.demo_adjust(chain, from_asset, -from_amount, from_contract,
                            usd_value=usd)
        self.db.demo_adjust(chain, to_asset, bought, to_contract,
                            usd_value=usd)
        return True, (f"swapped {from_amount} {from_asset} -> {bought:.6f} {to_asset} "
                      f"(${usd:.2f} at real prices); gas {gas_cost}")

    def _gas_symbol(self, chain: str) -> str:
        return {"arc": "USDC", "solana": "SOL"}.get(chain, "ETH")

    # Realistic bridge cost model: gas on BOTH chains (burn on src, mint on
    # dst) plus a bridge service fee, as real bridging almost always goes
    # through a platform charging a spread/fee.
    BRIDGE_SERVICE_FEE_PCT = 0.10  # 0.10% service fee on the bridged amount

    def bridge(self, src_chain: str, dst_chain: str, asset: str, amount: float) -> tuple[bool, str]:
        """Simulated bridge with REALISTIC costs: gas on the source chain
        (burn), gas on the destination chain (mint/receive), plus a service
        fee. All-or-nothing checks before touching any balance."""
        ok, why = self.gas_ok(src_chain)
        if not ok:
            return False, why
        ok_dst, why_dst = self.gas_ok(dst_chain)
        if not ok_dst:
            return False, (f"cannot receive on {dst_chain}: {why_dst} "
                           f"(real bridges need gas on BOTH chains)")
        have = self.balance(src_chain, asset)
        gas_src = self.per_tx_gas(src_chain)
        gas_dst = self.per_tx_gas(dst_chain)
        fee = amount * self.BRIDGE_SERVICE_FEE_PCT / 100.0
        gas_sym_src = self._gas_symbol(src_chain)
        gas_sym_dst = self._gas_symbol(dst_chain)
        # requirement on src: amount + fee + src gas (same balance if asset is gas)
        need_src = amount + fee
        if gas_sym_src == asset:
            need_src += gas_src
        if have + 1e-12 < need_src:
            return False, (f"insufficient {asset} on {src_chain}: have {have:.4f}, "
                           f"need {need_src:.4f} (bridge + {self.BRIDGE_SERVICE_FEE_PCT}% "
                           f"fee + gas)")
        # dst gas must exist on the destination (checked via gas_ok above)
        # all checks passed: deduct fees, move the net amount
        self.db.demo_adjust(src_chain, gas_sym_src, -gas_src, is_gas=True)
        self.db.demo_adjust(dst_chain, gas_sym_dst, -gas_dst, is_gas=True)
        self.db.demo_adjust(src_chain, asset, -(amount + fee))
        self.db.demo_adjust(dst_chain, asset, amount)
        return True, (f"bridged {amount} {asset} {src_chain}->{dst_chain}; "
                     f"costs: {fee:.4f} {asset} service fee + gas on both chains "
                     f"({gas_src} src, {gas_dst} dst). Available next cycle.")
