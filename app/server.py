"""Thin local web server (FastAPI). No trading logic here — only API surface."""
from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from memicoin_core.ai_engine import DecisionEngine
from memicoin_core.db import Database
from memicoin_core.market import MarketDataError, MarketDataModule
from memicoin_core.news import NewsModule
from memicoin_core.portfolio import PortfolioTracker
from memicoin_core.scheduler import DailyScheduler
from memicoin_core.secrets import SecretStore
from memicoin_core.settings import Constraints, load_constraints

DB_PATH = os.environ.get("MEMICOIN_DB", os.path.join(os.getcwd(), "data", "memicoin.db"))
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

db = Database(DB_PATH)
_constraints0 = load_constraints(db)
if db.get_cash() == 0.0 and not db.get_positions() and not db.list_decisions(1):
    db.set_cash(_constraints0.starting_capital_usd)
market = MarketDataModule(db)
news = NewsModule(db)
tracker = PortfolioTracker(db)


secrets = SecretStore()


def make_engine() -> DecisionEngine:
    return DecisionEngine(db, load_constraints(db), market, news,
                          onchain_secrets=secrets)


scheduler = DailyScheduler(db, make_engine)
scheduler.start()

app = FastAPI(title="MemeCoin AI Trader", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


@app.get("/api/state")
def state():
    constraints = load_constraints(db)
    try:
        prices = market.get_prices()
        portfolio = tracker.summary(prices)
        market_ok = True
    except MarketDataError:
        portfolio = {"cash_usd": db.get_cash(), "positions": [], "total_value_usd": 0,
                     "total_pnl_usd": 0, "total_pnl_pct": 0, "total_cost_basis_usd": 0,
                     "as_of": None}
        market_ok = False
    return {
        "portfolio": portfolio,
        "constraints": constraints.to_json(),
        "market_ok": market_ok,
        "trading_paused": constraints.trading_paused,
        "emergency_stopped": constraints.emergency_stopped,
        "paper_trading": constraints.paper_trading,
        "total_pnl_pct": portfolio.get("total_pnl_pct", 0),
    }


class EtherscanKeyPayload(BaseModel):
    api_key: str


@app.post("/api/etherscan-key")
def set_etherscan_key(p: EtherscanKeyPayload):
    """Store the (free) Etherscan V2 API key for EVM token discovery.
    Encrypted at rest, never returned."""
    key = p.api_key.strip()
    if not key:
        secrets.delete("etherscan_api_key")
        return {"ok": True, "configured": False}
    secrets.set("etherscan_api_key", key)
    return {"ok": True, "configured": True}


class WatchlistPayload(BaseModel):
    asset_ids: list[str]


@app.get("/api/watchlist")
def get_watchlist():
    from memicoin_core.market import DEFAULT_EXTRA_IDS
    return {"asset_ids": db.get_setting("extra_asset_ids", DEFAULT_EXTRA_IDS) or []}


@app.post("/api/watchlist")
def set_watchlist(p: WatchlistPayload):
    ids = [i.strip() for i in p.asset_ids if i.strip()][:30]
    db.set_setting("extra_asset_ids", ids)
    return {"ok": True, "asset_ids": ids}


@app.get("/api/wallet/balances")
def wallet_balances():
    """Real wallet holdings: native + known tokens per chain, valued in USD
    via the (cached) DexScreener prices. Only meaningful in real-money mode."""
    if not (secrets.has("onchain_private_key") or secrets.has("onchain_solana_private_key")):
        return {"configured": False, "chains": {}, "total_usd": 0.0}
    from memicoin_core.onchain import read_wallet_balances
    # 1) watchlist contracts (priced + explicitly tracked by the user)
    extra_contracts: dict[str, list[dict]] = {}
    try:
        for a in market.get_asset_map().values():
            contract = a.get("contract") or ""
            chain = a.get("chain") or ""
            if contract.startswith("0x") and len(contract) == 42 and chain:
                extra_contracts.setdefault(chain, []).append(
                    {"address": contract, "symbol": a["asset"]})
    except MarketDataError:
        pass
    # 2) AUTOMATIC DISCOVERY: every token the wallet ever transacted
    #    (Etherscan-family V2 key, if provided). Merged with the watchlist;
    #    actual balances are still read on-chain afterwards.
    evm_addr = None
    try:
        if secrets.has("onchain_private_key"):
            from web3 import Web3
            evm_addr = Web3().eth.account.from_key(
                secrets.get("onchain_private_key")).address
            if secrets.has("etherscan_api_key"):
                from memicoin_core.discovery import discover_evm_tokens
                discovered = discover_evm_tokens(
                    evm_addr, secrets.get("etherscan_api_key")) or {}
                for chain, toks in discovered.items():
                    bucket = extra_contracts.setdefault(chain, [])
                    known = {t["address"].lower() for t in bucket}
                    for t in toks:
                        if t["address"].lower() not in known:
                            bucket.append(t)
    except Exception:
        pass
    raw = read_wallet_balances(secrets, extra_contracts=extra_contracts)
    # 3) Solana: token accounts are directly enumerable, no key needed
    sol = raw.get("solana")
    if sol and sol.get("address"):
        try:
            from memicoin_core.discovery import discover_solana_tokens
            for t in discover_solana_tokens(sol["address"]):
                sol.setdefault("tokens", []).append(
                    {"symbol": t["address"][:8], "contract": t["address"],
                     "balance": t["balance"]})
        except Exception:
            pass
    try:
        prices = market.get_prices()
        asset_map = market.get_asset_map()
    except MarketDataError:
        prices, asset_map = {}, {}

    # Native coins have no contract of their own; DexScreener prices their
    # WRAPPED equivalents. Map native -> wrapped contract per chain so the
    # batch pricer can value native balances (ETH on Base, POL, BNB, ...).
    NATIVE_WRAPPED = {
        "ethereum": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        "base": "0x4200000000000000000000000000000000000006",      # WETH (base)
        "polygon": "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270",   # WPOL
        "binance-smart-chain": "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
        "solana": "So11111111111111111111111111111111111111112",   # wrapped SOL
    }

    # batch-price every contract the wallet holds so discovered tokens get
    # real USD values: contracts come from the RAW balance data (rows are
    # built below), priced via DexScreener in batches of 30
    all_contracts = []
    evm_raw = (raw.get("evm") or {}).get("chains") or {}
    for entry in evm_raw.values():
        for t in entry.get("tokens", []):
            c = t.get("contract") or ""
            if c and len(c) >= 32:
                all_contracts.append(c)
    sol_raw = raw.get("solana") or {}
    for t in sol_raw.get("tokens", []):
        c = t.get("contract") or t.get("address") or ""
        if c and len(c) >= 32:
            all_contracts.append(c)
    # wrapped-native contracts price the native balances
    for chain, entry in evm_raw.items():
        if entry.get("native", 0) > 0 and chain in NATIVE_WRAPPED:
            all_contracts.append(NATIVE_WRAPPED[chain])
    if (sol_raw.get("native") or 0) > 0:
        all_contracts.append(NATIVE_WRAPPED["solana"])
    addr_prices: dict = {}
    try:
        if all_contracts:
            addr_prices = market.get_prices_for_addresses(all_contracts)
    except MarketDataError:
        pass

    # Fallback discovery: tokens whose balance we know but whose contract
    # produced no price (e.g. Arc, which the Etherscan V2 API may not
    # index). Resolve the SYMBOL via DexScreener search on the token's own
    # chain, take that contract's price. Bounded: one search per unpriced
    # symbol, results cached by the market module.
    def _resolve_symbol_price(symbol: str, chain: str) -> float:
        try:
            pairs = market._ds_search(symbol)
        except Exception:
            return 0.0
        best_price, best_liq = 0.0, -1.0
        for p in pairs:
            if (p.get("chainId") or "") != chain:
                continue
            base = p.get("baseToken") or {}
            if (base.get("symbol") or "").upper() != symbol.upper():
                continue
            liq = float((p.get("liquidity") or {}).get("usd") or 0.0)
            price = float(p.get("priceUsd") or 0.0)
            if price > 0 and liq > best_liq:
                best_price, best_liq = price, liq
        return best_price

    native_prices = {"ethereum": "ETH", "base": "ETH", "polygon": "POL",
                     "binance-smart-chain": "BNB", "arc": "USDC", "solana": "SOL"}
    # native coins we can price via watchlist lookups (symbols in market data)
    chains: dict = {}
    total = 0.0

    def value_of(symbol: str, balance: float, contract: str = "",
                chain: str = "") -> float:
        p = prices.get(symbol.upper(), 0.0)
        if p <= 0 and contract:
            p = addr_prices.get(contract.lower(), 0.0)
        if p <= 0 and symbol and chain:
            p = _symbol_fallback.get((symbol.upper(), chain), 0.0)
        # stablecoin peg floor: USDC/USDT are ~$1 by definition; on Arc
        # USDC is native with no wrapped contract, so market data can miss it
        if p <= 0 and symbol.upper() in ("USDC", "USDT"):
            p = 1.0
        return balance * p

    # pre-resolve unpriced symbols once (bounded searches, memoized)
    _symbol_fallback: dict = {}

    # stablecoins never go through symbol-search: searching "USDC" returns
    # decoy tokens with fake liquidity (verified live: a $0.0001 "USDC" on
    # Base with $100k liquidity). The peg floor is the honest price floor.
    STABLES = {"USDC", "USDT", "DAI", "FRAX", "USDE", "SUSDS"}

    def _price_unpriced(rows_by_chain: dict) -> None:
        for chain, rows in rows_by_chain.items():
            for r in rows:
                key = (r.get("symbol", "").upper(), chain)
                if key in _symbol_fallback or not r.get("symbol"):
                    continue
                if r.get("value_usd", 0) > 0 or r.get("kind") == "native":
                    continue
                if (r.get("symbol") or "").upper() in STABLES:
                    continue  # peg floor handles stables; search invites decoys
                _symbol_fallback[key] = _resolve_symbol_price(r["symbol"], chain)

    evm = raw.get("evm")
    if evm:
        for chain, entry in (evm.get("chains") or {}).items():
            sym = entry.get("native_symbol", native_prices.get(chain, "ETH"))
            wrapped = NATIVE_WRAPPED.get(chain, "")
            rows = []
            if entry.get("native", 0) > 0:
                rows.append({"symbol": sym, "balance": entry["native"], "value_usd": 0.0,
                             "chain": chain, "kind": "native", "contract": wrapped})
            for t in entry.get("tokens", []):
                rows.append({"symbol": t["symbol"], "balance": t["balance"],
                             "value_usd": 0.0, "chain": chain, "kind": "token",
                             "contract": t.get("contract", "")})
            chains[chain] = rows
        # native rows also get zeroed first for uniform final valuation
        for chain_rows in chains.values():
            for r in chain_rows:
                r["value_usd"] = 0.0
        # resolve unpriced symbols via DexScreener search (bounded, memoized)
        _price_unpriced(chains)
        # final valuation pass: symbol -> contract -> symbol-search -> peg
        for chain, chain_rows in chains.items():
            for r in chain_rows:
                r["value_usd"] = value_of(r["symbol"], r["balance"],
                                          r.get("contract", ""), chain)
        total += sum(r["value_usd"] for rows in chains.values() for r in rows)
    sol = raw.get("solana")
    if sol:
        rows = []
        if sol.get("native", 0) > 0:
            rows.append({"symbol": "SOL", "balance": sol["native"], "value_usd": 0.0,
                         "chain": "solana", "kind": "native",
                         "contract": NATIVE_WRAPPED["solana"]})
        for t in sol.get("tokens", []):
            rows.append({"symbol": t.get("symbol", "?"), "balance": t.get("balance", 0.0),
                         "value_usd": 0.0, "chain": "solana", "kind": "token",
                         "contract": t.get("contract", t.get("address", ""))})
        chains["solana"] = rows
    return {"configured": True, "evm_address": (evm or {}).get("address"),
            "solana_address": (sol or {}).get("address"),
            "chains": chains, "total_usd": total,
            "priced": bool(prices), "error": raw.get("error")}


@app.get("/api/decisions")
def decisions():
    return {"decisions": db.list_decisions()}


@app.get("/api/cycles")
def cycles():
    return {"cycles": db.list_cycles()}


@app.get("/api/news")
def get_news():
    return {"news": news.get_news()}


@app.post("/api/constraints")
def set_constraints(c: Constraints):
    current = load_constraints(db)
    if current.paper_trading and not c.paper_trading:
        raise HTTPException(400, "Switching to live trading requires explicit confirmation via /api/go-live")
    db.set_setting("constraints", c.to_json())
    return {"ok": True}


class GoLiveRequest(BaseModel):
    confirm: bool


@app.post("/api/go-live")
def go_live(req: GoLiveRequest):
    """Switch to real-money mode. Requires an imported wallet key."""
    if not req.confirm:
        raise HTTPException(400, "Confirmation required")
    if not (secrets.has("onchain_private_key") or secrets.has("onchain_solana_private_key")):
        raise HTTPException(400, "No wallet imported — import a key in Settings first")
    c = load_constraints(db)
    c.paper_trading = False
    c.live_execution_mode = "onchain"
    db.set_setting("constraints", c.to_json())
    return {"ok": True, "mode": "real"}


@app.post("/api/go-paper")
def go_paper():
    c = load_constraints(db)
    c.paper_trading = True
    db.set_setting("constraints", c.to_json())
    return {"ok": True, "mode": "paper"}


# ---------- on-chain wallet (Mode B) ----------
class WalletImport(BaseModel):
    chain: str
    private_key: str = ""
    seed_phrase: str = ""


@app.post("/api/wallet/import")
def wallet_import(w: WalletImport):
    """Import a hot wallet private key. chain='evm' imports ONE key valid for
    all EVM chains (Ethereum, Base, Polygon, BSC, Arc); chain='solana' imports
    a Solana keypair. Keys are stored encrypted and NEVER returned; only the
    derived public address is echoed once for user verification."""
    chain = w.chain.strip().lower()
    if w.seed_phrase and chain == "solana":
        raise HTTPException(400, "Seed phrases cannot be used for Solana in this app: Solana wallets use non-standard derivation. Export the raw key from your wallet app and import that.")
    try:
        if chain in ("evm", "ethereum", "base", "polygon", "binance-smart-chain", "arc"):
            from web3 import Web3
            if w.seed_phrase:
                words = w.seed_phrase.strip().lower().split()
                if len(words) not in (12, 15, 18, 21, 24):
                    raise HTTPException(400, f"Seed phrase must have 12/15/18/21/24 words — got {len(words)}")
                from eth_account import Account
                Account.enable_unaudited_hdwallet_features()
                acct = Account.from_mnemonic(" ".join(words))  # standard path m/44'/60'/0'/0/0
                secrets.set("onchain_private_key", acct.key.hex())
                return {"ok": True, "chain": "evm", "address": acct.address,
                        "derived_from": "seed (first account)"}
            key = w.private_key if w.private_key.startswith("0x") else "0x" + w.private_key
            acct = Web3().eth.account.from_key(key)
            secrets.set("onchain_private_key", key)
            return {"ok": True, "chain": "evm", "address": acct.address}
        if chain == "solana":
            import json as _json
            from solders.keypair import Keypair
            try:
                kp = Keypair.from_bytes(bytes(_json.loads(w.private_key)))
            except Exception:
                kp = Keypair.from_base58_text(w.private_key)
            secrets.set("onchain_solana_private_key", w.private_key)
            return {"ok": True, "chain": "solana", "address": str(kp.pubkey())}
        raise HTTPException(400, "Wallet type must be 'evm' or 'solana'.")
    except HTTPException:
        raise
    except ModuleNotFoundError as e:
        raise HTTPException(500, f"Server dependency missing: {e}. Run ./run.sh (or: ./.venv/bin/pip install -r requirements.txt) and restart.")
    except Exception as e:
        raise HTTPException(400, f"Invalid private key: {e}")


@app.get("/api/wallet/status")
def wallet_status():
    out = {"evm_configured": False, "evm_address": None,
           "solana_configured": False, "solana_address": None}
    if secrets.has("onchain_private_key"):
        out["evm_configured"] = True
        try:
            from web3 import Web3
            out["evm_address"] = Web3().eth.account.from_key(
                secrets.get("onchain_private_key")).address
        except Exception:
            pass
    if secrets.has("onchain_solana_private_key"):
        out["solana_configured"] = True
        try:
            from memicoin_core.onchain import SolanaWallet
            out["solana_address"] = SolanaWallet(secrets).address()
        except Exception:
            pass
    return out


@app.delete("/api/wallet")
def wallet_delete():
    secrets.delete("onchain_private_key")
    secrets.delete("onchain_solana_private_key")
    return {"ok": True}


@app.post("/api/cycle")
def run_cycle():
    result = make_engine().run_cycle()
    return result


@app.post("/api/pause")
def pause():
    c = load_constraints(db)
    c.trading_paused = not c.trading_paused
    db.set_setting("constraints", c.to_json())
    return {"trading_paused": c.trading_paused}


@app.post("/api/emergency-stop")
def emergency_stop():
    c = load_constraints(db)
    c.emergency_stopped = True
    c.trading_paused = True
    db.set_setting("constraints", c.to_json())
    return {"ok": True}


@app.post("/api/emergency-to-stablecoin")
def emergency_to_stablecoin():
    """Sell everything to USD cash (paper mode) — the safe stablecoin equivalent."""
    c = load_constraints(db)
    c.emergency_stopped = True
    c.trading_paused = True
    db.set_setting("constraints", c.to_json())
    cycle_id = db.start_cycle()
    c = load_constraints(db)
    if not c.paper_trading:
        # REAL mode: sell via the imported wallet. Chains without a router
        # (e.g. Arc today) are reported as not-liquidatable, never faked.
        from memicoin_core.onchain import EVM_CHAINS, SolanaWallet, EVMWallet
        results = []
        try:
            prices = market.get_prices()
        except MarketDataError:
            prices = {}
        from memicoin_core.onchain import (EVM_CHAINS, EVMWallet, SolanaWallet,
                                           SOLANA_USDC_MINT, read_wallet_balances,
                                           stablecoin_target)
        raw = read_wallet_balances(secrets)
        evm = (raw.get("evm") or {}).get("chains") or {}
        for chain, entry in evm.items():
            target_addr, target_label = stablecoin_target(chain)
            if chain in EVM_CHAINS and EVM_CHAINS[chain].get("router"):
                w = EVMWallet(secrets, chain)
                for t in entry.get("tokens", []):
                    res = w.swap_exact_tokens_for_tokens(t["contract"], target_addr, t["balance"])
                    results.append({"chain": chain, "symbol": t["symbol"], "ok": res.ok,
                                    "tx_id": res.tx_id, "target": target_label,
                                    "error": res.error})
            else:
                for t in entry.get("tokens", []):
                    results.append({"chain": chain, "symbol": t["symbol"], "ok": False,
                                    "target": target_label,
                                    "error": "no DEX router on this chain — cannot liquidate automatically"})
        sol = raw.get("solana")
        if sol and sol.get("tokens"):
            sw = SolanaWallet(secrets)
            for t in sol.get("tokens", []):
                res = sw.swap(t.get("contract", t.get("mint", "")), SOLANA_USDC_MINT,
                              t.get("balance", 0.0))
                results.append({"chain": "solana", "symbol": t.get("symbol", "?"),
                                "ok": res.ok, "tx_id": res.tx_id, "target": "USDC",
                                "error": res.error})
        db.finish_cycle(cycle_id, "emergency", "", "",
                        summary="Emergency liquidation to stablecoins (real mode)")
        ok_all = all(r["ok"] for r in results) if results else True
        for r in results:
            db.add_decision({
                "cycle_id": cycle_id, "action": "sell", "asset_in": r["symbol"],
                "asset_out": "native", "status": "executed" if r["ok"] else "failed",
                "reason": f"Emergency liquidation on {r['chain']} -> {r.get('target', 'stablecoin')}: {r['error'] or 'sold'}",
                "tx_id": r.get("tx_id", ""),
            })
        return {"ok": ok_all, "mode": "real", "results": results}

    # PAPER mode: sell virtual positions at market prices
    try:
        prices = market.get_prices()
    except MarketDataError as e:
        raise HTTPException(503, f"Cannot liquidate: market data unavailable ({e})")
    db.finish_cycle(cycle_id, "emergency", "", "", summary="Emergency convert to cash")
    for p in db.get_positions():
        qty = p["quantity"]
        price = prices.get(p["asset"], 0.0)
        proceeds = qty * price
        pnl = tracker.apply_sell(p["asset"], qty, proceeds)
        db.add_decision({
            "cycle_id": cycle_id, "action": "sell", "asset_in": p["asset"], "asset_out": "USD",
            "amount": qty, "value_usd": proceeds, "pnl_per_trade": pnl,
            "status": "executed", "reason": "Emergency stop: converted to stablecoin/cash.",
            "tx_id": f"paper-emergency-{p['asset']}",
        })
    return {"ok": True, "mode": "paper"}


@app.post("/api/resume")
def resume():
    c = load_constraints(db)
    c.emergency_stopped = False
    c.trading_paused = False
    db.set_setting("constraints", c.to_json())
    return {"ok": True}


@app.post("/api/reset-paper")
def reset_paper():
    """Reset the paper portfolio to the starting capital."""
    for p in db.get_positions():
        db.delete_position(p["asset"])
    c = load_constraints(db)
    db.set_cash(c.starting_capital_usd)
    return {"ok": True}


@app.get("/api/cycles/latest")
def latest_cycle():
    rows = db.list_cycles(limit=1)
    return rows[0] if rows else {}


@app.get("/api/setup-status")
def setup_status():
    """First-run guidance: which integrations are configured (booleans only)."""
    return {
        "mistral_key": bool(os.environ.get("MISTRAL_API_KEY")) or secrets.has("mistral_api_key"),
        "wallet": secrets.has("onchain_private_key") or secrets.has("onchain_solana_private_key"),
        "paper_trading": load_constraints(db).paper_trading,
    }


class MistralKeyPayload(BaseModel):
    api_key: str


@app.post("/api/mistral-key")
def set_mistral_key(p: MistralKeyPayload):
    """Store the Mistral API key encrypted. Never returned afterwards."""
    key = p.api_key.strip()
    if not key:
        secrets.delete("mistral_api_key")
        return {"ok": True, "configured": False}
    secrets.set("mistral_api_key", key)
    return {"ok": True, "configured": True}
