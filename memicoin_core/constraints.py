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
                           tracker, db, market=None,
                           live_total_usd: float | None = None,
                           live_holdings: list[dict] | None = None) -> float | None:
    """Resolve an action's amount_or_percent into absolute asset units.
    Raises nothing; returns None when unresolvable."""
    raw = action.get("amount")
    if isinstance(raw, (int, float)) and raw > 0:
        return float(raw)
    a = action.get("amount_or_percent")
    if isinstance(a, (int, float)):
        val = float(a)
        if val <= 0:
            return None
        # same contract as strings: numeric buys are USD, not coin quantities
        if action.get("type") == "buy":
            price = action.get("_price") or prices.get(action.get("asset", ""), 0.0)
            return val / price if price > 0 else None
        return val
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
    asset = action.get("asset", "")
    if not is_pct:
        # Amount contract: for BUYS a plain number is a USD budget (e.g. 25
        # = spend $25), because AI-proposed coin quantities on sub-cent meme
        # coins resolve to dust. For sells/swaps it stays a coin quantity.
        if action.get("type") == "buy":
            price = action.get("_price") or prices.get(asset, 0.0)
            return val / price if price > 0 else None
        return val
    pct = val / 100.0
    if action.get("type") == "buy":
        # resolve against the funds that will actually pay:
        # onchain mode -> imported wallet; demo mode -> demo wallet
        # spendables; paper mode -> paper portfolio
        total = (live_total_usd if live_total_usd is not None
                 else tracker.total_value(prices))
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
        # onchain mode: sells resolve against the imported wallet holding
        # (matched by symbol OR contract, however the AI named the asset)
        if live_holdings is not None:
            q = sum(r["quantity"] for r in live_holdings
                    if (r.get("asset") == asset
                        or r.get("contract", "").lower() == asset.lower())
                    and r.get("quantity", 0) > 0)
            if q > 0:
                return q * pct
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
        # live holdings snapshot injected by the engine each onchain cycle,
        # so sell/swap validation sees the REAL wallet (paper books are empty)
        self.live_holdings: list[dict] | None = None
        self.live_wallet_total: float | None = None

    def _market_addr_prices(self, contracts):
        if self._market is None:
            return {}
        return self._market.get_prices_for_addresses(contracts)

    def check_stop_conditions(self, prices: dict[str, float],
                              live_total_usd: float | None = None) -> ValidationResult:
        c = self.constraints
        if c.emergency_stopped:
            return ValidationResult(False, "Emergency stop is active; all trading halted.")
        if c.trading_paused:
            return ValidationResult(False, "Trading is paused by the user.")
        # mode-aware valuation: paper books for paper mode; the wallet the bot
        # actually trades otherwise, compared against the pnl_baseline_usd
        # snapshot (taken at the first cycle after the wallet last changed).
        if not c.paper_trading and live_total_usd is not None and live_total_usd > 0:
            baseline = self.db.get_setting("pnl_baseline_usd")
            try:
                baseline = float(baseline) if baseline else 0.0
            except (TypeError, ValueError):
                baseline = 0.0
            pnl_pct = ((live_total_usd - baseline) / baseline * 100) if baseline > 0 else 0.0
        elif getattr(self.db, "demo_mode_percent", False):
            # engine passed no wallet total (direct validator use in tests):
            # value the demo wallet, but without a baseline the daily
            # profit/loss stops cannot measure drawdown.
            pnl_pct = 0.0
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
        info = known_assets.get(asset)
        if a_type in ("buy", "sell", "swap") and info is None \
                and self._market is not None \
                and getattr(self.constraints, "auto_market_expansion", True):
            # fresh token: resolve the ticker live on DexScreener so
            # brand-new coins are tradable without the user adding
            # contracts; the best pair supplies contract, chain and price
            try:
                pair = self._market.search_contract(asset, None)
            except Exception:
                pair = None
            if pair:
                from .market import DS_CHAIN_ALIAS
                base = pair.get("baseToken") or {}
                dsc = (pair.get("chainId") or "")
                chain = DS_CHAIN_ALIAS.get(dsc.lower(), dsc)
                sym = (base.get("symbol") or asset).upper()
                contract = base.get("address") or ""
                if self.constraints.live_execution_mode == "onchain":
                    allowed = [a.lower() for a in self.constraints.allowed_chains]
                    if (self.constraints.autonomy_mode == "constrained"
                            and chain and chain.lower() not in allowed):
                        return ValidationResult(
                            False,
                            f"{sym} trades on {chain}, which is not in your "
                            f"allowed chains ({', '.join(allowed)}).")
                action["_chain"] = chain
                info = {"asset": sym, "chain": chain, "contract": contract}
                if sym != asset:
                    asset = sym
                    action["asset"] = asset
                if contract:
                    action["_contract"] = contract
                try:
                    price = float(pair.get("priceUsd") or 0.0)
                except (TypeError, ValueError):
                    price = 0.0
                if price > 0 and not action.get("_price"):
                    action["_price"] = price
                try:
                    self._market.note_asset(pair)
                except Exception:
                    pass

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
            # a DEX swap settles on ONE chain: refuse token->token proposals
            # whose endpoints live on different chains (bridge moves USDC
            # across chains instead)
            src_chain = ((known_assets.get(resolved[0]) or {}).get("chain")
                         or action.get("_chain") or "")
            dst_info = known_assets.get(resolved[1]) or {}
            dst_chain = dst_info.get("chain") or ""
            from .onchain import EVM_CHAINS as _EVM
            src_native = "SOL" if src_chain == "solana" else (
                _EVM.get(src_chain, {}).get("native_symbol") or "ETH").upper()
            if (self.constraints.live_execution_mode == "onchain"
                    and src_chain and dst_chain and src_chain != dst_chain
                    and resolved[1].upper() != src_native):
                return ValidationResult(
                    False,
                    f"Cross-chain swap {resolved[0]} ({src_chain}) to "
                    f"{resolved[1]} ({dst_chain}) is not a DEX swap; propose "
                    f"a bridge action, or swap to the {src_native} of "
                    f"{src_chain}.")
            if resolved[1] in known_assets and known_assets[resolved[1]].get("contract"):
                action["_swap_dest_contract"] = known_assets[resolved[1]]["contract"]
            if resolved[0] in known_assets and known_assets[resolved[0]].get("contract"):
                action["_contract"] = known_assets[resolved[0]]["contract"]
            # the swap SELLS the held asset: resolve the amount against holdings
            held = self._held_qty(resolved[0])
            if held <= 0:
                return ValidationResult(
                    False,
                    f"Cannot swap {resolved[0]}: you do not hold it "
                    f"(wallet holds: {self._holdings_hint()}). "
                    f"Do not infer holdings from past decisions.")
            if amount is None or amount <= 0:
                return ValidationResult(False, "Missing or non-positive amount.")
            action["amount"] = amount
            return ValidationResult(True, "")

        # resolve asset: the AI may reference by symbol OR contract address.
        # Keep BOTH: the symbol for validation/pricing, the contract for the
        # executor (which requires the contract to identify the token).
        if info is None:
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
        if info and info.get("chain"):
            action["_chain"] = info["chain"]
        r = self._validate_asset(asset, known_assets, info)
        if not r.ok:
            return r

        if a_type in ("sell", "swap"):
            held = self._held_qty(asset)
            if held <= 0:
                return ValidationResult(
                    False,
                    f"Cannot {a_type} {asset}: you do not hold it "
                    f"(wallet holds: {self._holdings_hint()}). "
                    f"Do not infer holdings from past decisions.")

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
            if cost < c.min_trade_usd:
                return ValidationResult(
                    False,
                    f"Buy of ${cost:.2f} is below the minimum trade size "
                    f"${c.min_trade_usd:.2f} (dust guard). Buys are sized in USD.",
                )
            # real mode: cap the buy by the investable funds on the token's
            # chain (gas-token balance minus the gas reserve) so the executor
            # never attempts an unfundable buy
            if (self.constraints.live_execution_mode == "onchain"
                    and self.live_holdings is not None):
                investable = self._live_investable_usd(asset, known_assets)
                if investable < c.min_trade_usd:
                    tok_chain = (known_assets.get(asset) or {}).get("chain", "")
                    others = sorted({r.get("chain") for r in self.live_holdings
                                     if r.get("is_gas") and r.get("chain")
                                     and r.get("chain") != tok_chain})
                    hint = (f" Wallet holds gas on {', '.join(others)}: only "
                            f"tokens on those chains can be bought with the "
                            f"funds there." if others else "")
                    return ValidationResult(
                        False,
                        f"Only ${investable:.2f} is investable on "
                        f"{tok_chain or 'the token\'s chain'} after the gas "
                        f"reserve (below the ${c.min_trade_usd:.2f} minimum "
                        f"trade).{hint}")
                if cost > investable + 1e-9:
                    cost = investable
                    amount = investable / price
                    action["amount"] = amount
                    action["amount_or_percent"] = amount
                    action["_clamped"] = (f"AI asked for more; clamped to "
                                          f"${investable:.2f}, the investable funds "
                                          f"on the token's chain after the gas reserve.")
            # portfolio total: the imported wallet in onchain mode; the demo
            # wallet in demo mode; paper books otherwise
            if (self.constraints.live_execution_mode == "onchain"
                    and self.live_wallet_total is not None):
                total = self.live_wallet_total
                existing_value = self._held_qty(asset) * price
            elif getattr(self.db, "demo_mode_percent", False):
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
                if (self.constraints.live_execution_mode == "onchain"
                        and self.live_holdings is not None):
                    cash = self._live_investable_usd(asset, known_assets)
                else:
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
            held = self._held_qty(asset)
            if amount > held + 1e-9:
                return ValidationResult(
                    False, f"Sell amount exceeds holdings of {asset} ({held:.6g}).")
        action["amount"] = amount
        return ValidationResult(True, "")

    def set_live_holdings(self, rows: list[dict],
                          total_usd: float | None = None) -> None:
        self.live_holdings = rows
        self.live_wallet_total = total_usd

    def _live_investable_usd(self, asset: str, known_assets: dict) -> float:
        """Real mode: investable USD on the token's chain — the chain's
        gas-token balance minus the adaptive gas reserve, valued in USD."""
        from .demowallet import (GAS_COST_PER_TX, NATIVE_WRAPPED,
                                 reserve_swaps_for)
        chain = (known_assets.get(asset) or {}).get("chain", "")
        if not chain:
            return 0.0
        gas_rows = [r for r in self.live_holdings
                    if r.get("chain") == chain and r.get("is_gas")]
        if not gas_rows:
            return 0.0
        native_sym = gas_rows[0]["asset"]
        native_qty = sum(r["quantity"] for r in gas_rows)
        per_tx = GAS_COST_PER_TX.get(chain, 0.0)
        invest_qty = max(0.0, native_qty
                         - per_tx * reserve_swaps_for(native_qty, per_tx))
        if invest_qty <= 0:
            return 0.0
        if chain == "arc":
            return invest_qty
        w = NATIVE_WRAPPED.get(chain)
        if not w:
            return 0.0
        try:
            wp = self._market_addr_prices([w])
        except Exception:
            return 0.0
        return invest_qty * wp.get(w.lower(), 0.0)

    def _held_qty(self, asset: str) -> float:
        """Total held quantity of an asset in the ACTIVE books: the imported
        wallet in onchain mode, the demo wallet in demo mode, the paper
        portfolio otherwise."""
        if (self.constraints.live_execution_mode == "onchain"
                and self.live_holdings is not None):
            al = (asset or "").lower()
            return sum(r["quantity"] for r in self.live_holdings
                       if (r.get("asset") == asset
                           or r.get("contract", "").lower() == al)
                       and r["quantity"] > 0)
        if getattr(self.db, "demo_mode_percent", False):
            from .demowallet import DemoWallet
            return sum(r["balance"] for r in DemoWallet(self.db).rows()
                       if r["asset"] == asset and r["balance"] > 0)
        pos = self.db.get_position(asset)
        return pos["quantity"] if pos else 0.0

    def _holdings_hint(self) -> str:
        if (self.constraints.live_execution_mode == "onchain"
                and self.live_holdings is not None):
            return (", ".join(f"{r['quantity']:.6g} {r['asset']} on {r['chain']}"
                              for r in self.live_holdings) or "nothing")
        if getattr(self.db, "demo_mode_percent", False):
            from .demowallet import DemoWallet
            rows = [r for r in DemoWallet(self.db).rows() if r["balance"] > 0]
            return (", ".join(f"{r['balance']:.6g} {r['asset']} on {r['chain']}"
                               for r in rows) or "nothing")
        positions = [p for p in self.db.get_positions() if p["quantity"] > 0]
        return (", ".join(f"{p['quantity']:.6g} {p['asset']}"
                           for p in positions) or "nothing")

    def _validate_asset(self, asset: str, known_assets: dict[str, dict],
                        info: dict | None = None) -> ValidationResult:
        c = self.constraints
        if not asset:
            return ValidationResult(False, "Missing asset.")
        if asset == STABLECOIN:
            return ValidationResult(True, "")
        if info is None:
            info = known_assets.get(asset)
        if info is None and (self.constraints.live_execution_mode == "onchain"
                             and self.live_holdings is not None):
            al = (asset or "").lower()
            row = next((r for r in self.live_holdings
                        if r.get("asset") == asset
                        or r.get("contract", "").lower() == al), None)
            if row is not None:
                info = {"asset": row["asset"], "chain": row["chain"],
                        "contract": row.get("contract", "")}
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
                                     market=self._market,
                                     live_total_usd=self.live_wallet_total,
                                     live_holdings=self.live_holdings)
