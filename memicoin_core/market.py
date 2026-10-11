"""Market Data Module (§4.1): DexScreener primary + CoinGecko fallback.

DexScreener indexes every token with a DEX pair — including the full Arc
ecosystem — and its token-pairs endpoint batches up to 30 addresses per call.
The watchlist accepts either CoinGecko asset ids (e.g. circle-wrapped-btc)
or raw token contract/mint addresses (0x... or Solana base58), which are
resolved through DexScreener's token-pairs endpoint.
"""
from __future__ import annotations

import time

import requests

from .db import Database

CHAIN_HINTS = {
    "circle-wrapped-btc": "arc",  # cirBTC: volatile BTC exposure on Arc
    "dogecoin": "dogecoin",
    "shiba-inu": "ethereum",
    "pepe": "ethereum",
    "bonk": "solana",
    "floki": "ethereum",
    "dogwifcoin": "solana",
    "brett-based": "base",
    "popcat": "solana",
    "mog-coin": "ethereum",
    "cat-in-a-dogs-world": "solana",
    "goatseus-maximus": "solana",
    "turbo": "ethereum",
}

# Default watchlist: CoinGecko ids and/or contract addresses.
# cirBTC on Arc (0x171a...) and on Ethereum (0x72df...) are both indexed by
# DexScreener; ids are kept for the CoinGecko fallback.
DEFAULT_EXTRA_IDS = [
    "circle-wrapped-btc",
    "0x171a4217b86a807a64eb94757db6849fb4bdbaa0",  # cirBTC on Arc (docs)
    "0x532f27101965dd16442E59d40670FaF5eBB142E4",  # BRETT on Base (DexScreener-verified)
]

# Symbols queried on DexScreener (search endpoint) per chain.
DEX_SYMBOLS_BY_CHAIN = {
    "ethereum": ["SHIB", "PEPE", "FLOKI", "TURBO", "MOG"],
    "solana": ["BONK", "WIF", "POPCAT", "MEW", "GOAT"],
    "binance-smart-chain": ["FLOKI", "TURBO", "BABYDOGE"],
    "base": ["BRETT", "DEGEN", "TYBG"],
    "arc": ["CIRBTC", "USYC", "EURC"],
}

# Addresses are looked up via the batch token-pairs endpoint (max 30/call).
_ADDRESS_HINT = ("0x", "0X")

USER_AGENT = "MemeCoinAITrader/0.1 (local desktop app)"
TIMEOUT = 15


class MarketDataError(Exception):
    pass


def _is_address(s: str) -> bool:
    s = s.strip()
    return s.startswith(_ADDRESS_HINT) and len(s) == 42 or (len(s) in (32, 44) and not s.startswith("0x"))


class MarketDataModule:
    def __init__(self, db: Database):
        self.db = db
        self._cache: dict = {}
        self._cache_ts: float = 0.0
        self.cache_ttl = 300  # seconds

    # ---------- public API ----------
    def get_market_overview(self, force: bool = False) -> list[dict]:
        if not force and self._cache and time.time() - self._cache_ts < self.cache_ttl:
            return self._cache["assets"]

        assets, source = [], ""
        try:
            assets, source = self._fetch_dexscreener()
        except (requests.RequestException, MarketDataError):
            try:
                assets, source = self._fetch_coingecko()
            except (requests.RequestException, MarketDataError) as e:
                raise MarketDataError(f"All market data sources failed: {e}") from e

        self._cache = {"assets": assets}
        self._cache_ts = time.time()
        self.db.save_market_snapshot(source, assets)
        return assets

    def get_prices(self, force: bool = False) -> dict[str, float]:
        return {a["asset"]: a["price_usd"] for a in self.get_market_overview(force)}

    def get_prices_for_addresses(self, contracts: list[str]) -> dict[str, float]:
        """Price arbitrary token contracts via DexScreener token-pairs
        (batches of 30). Returns {contract_address_lower: price_usd}.
        Used to value wallet holdings the AI universe doesn't cover."""
        out: dict[str, float] = {}
        contracts = [c for c in contracts if c and c.strip()]
        for i in range(0, len(contracts), 30):
            batch = contracts[i:i + 30]
            try:
                pairs = self._ds_token_pairs(batch)
            except requests.RequestException:
                continue
            for p in pairs:
                base = p.get("baseToken") or {}
                addr = (base.get("address") or "").lower()
                price = float(p.get("priceUsd") or 0.0)
                if addr and price > 0:
                    # keep the deepest-liquidity pair's price per contract
                    liq = float((p.get("liquidity") or {}).get("usd") or 0.0)
                    if addr not in out or liq > out.get(addr + "_liq", -1):
                        out[addr] = price
                        out[addr + "_liq"] = liq
        return {k: v for k, v in out.items() if not k.endswith("_liq")}

    def get_asset_map(self, force: bool = False) -> dict[str, dict]:
        return {a["asset"]: a for a in self.get_market_overview(force)}

    # ---------- watchlist helpers ----------
    def _watchlist(self) -> list[str]:
        return [i.strip() for i in (self.db.get_setting("extra_asset_ids", DEFAULT_EXTRA_IDS) or []) if i.strip()]

    # ---------- DexScreener (primary) ----------
    def _fetch_dexscreener(self) -> tuple[list[dict], str]:
        symbols = sorted({s for syms in DEX_SYMBOLS_BY_CHAIN.values() for s in syms})
        assets: dict[str, dict] = {}

        # 1) symbol search for the known meme coins (one call per symbol)
        allowed = {a.strip().lower() for a in
                   (self.db.get_setting("constraints") or {}).get("allowed_chains", [])
                   if a.strip()} or None
        for sym in symbols:
            try:
                pairs = self._ds_search(sym)
            except requests.RequestException:
                continue
            best = self._best_pair(pairs, wanted_chains=allowed, symbol=sym)
            if best:
                self._add_ds_asset(assets, best)

        # 2) batch token-pairs for watchlist contract addresses (<=30 per call)
        addresses = [a for a in self._watchlist() if _is_address(a)]
        for i in range(0, len(addresses), 30):
            batch = addresses[i:i + 30]
            try:
                pairs = self._ds_token_pairs(batch)
            except requests.RequestException:
                continue
            for p in pairs:
                self._add_ds_asset(assets, p)

        # 3) ids that are not addresses (CoinGecko-style) get resolved via search
        for entry in self._watchlist():
            if not _is_address(entry) and entry.upper() not in assets:
                try:
                    pairs = self._ds_search(entry)
                except requests.RequestException:
                    continue
                best = self._best_pair(pairs, wanted_chains=allowed, symbol=sym)
                if best:
                    self._add_ds_asset(assets, best)

        if not assets:
            raise MarketDataError("DexScreener returned no usable pairs")
        return list(assets.values()), "dexscreener"

    def _ds_search(self, query: str) -> list[dict]:
        url = "https://api.dexscreener.com/latest/dex/search"
        resp = requests.get(url, params={"q": query}, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        return (resp.json() or {}).get("pairs") or []

    def _ds_token_pairs(self, addresses: list[str]) -> list[dict]:
        """Token pairs by contract address. The /tokens/v1/ endpoint returns
        empty for many indexed tokens (verified live); /latest/dex/tokens/
        is the reliable variant and also accepts comma-separated batches."""
        url = "https://api.dexscreener.com/latest/dex/tokens/" + ",".join(addresses)
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("pairs") or []
        return data if isinstance(data, list) else []

    # DexScreener uses pseudo-chain ids for its own venues ("robinhood") and
    # search can return unverified/illiquid pairs. Only real chains count.
    REAL_CHAINS = {"ethereum", "base", "solana", "bsc", "binance-smart-chain",
                   "polygon", "arbitrum", "avalanche", "arc", "sui", "ton"}
    GOOD_QUOTES = ("USDC", "USDT", "WETH", "SOL", "WBTC", "ETH", "WBNB")

    @classmethod
    def _best_pair(cls, pairs: list[dict], wanted_chains: set[str] | None = None,
                   symbol: str | None = None) -> dict | None:
        """Pick the deepest verified pair on a real chain. Prefers chains in
        wanted_chains (the user's allowed list) when provided, else the
        highest-liquidity real-chain pair. Requires non-trivial liquidity."""
        def is_real(p):
            return (p.get("chainId") or "") in cls.REAL_CHAINS

        def liq(p):
            return float((p.get("liquidity") or {}).get("usd") or 0.0)

        def quote_ok(p):
            # the search endpoint does not return quoteTokenSymbol at all,
            # so only enforce the quote check when the field is present
            q = p.get("quoteTokenSymbol")
            return q is None or q.upper() in cls.GOOD_QUOTES

        def vol24(p):
            return float((p.get("volume") or {}).get("h24") or 0.0)

        # require real trading activity: pairs with 0 volume are usually
        # imposter/scam tokens squatting a popular symbol
        def base_sym(p):
            return ((p.get("baseToken") or {}).get("symbol") or "").upper()

        candidates = [p for p in pairs
                      if is_real(p) and liq(p) > 0 and quote_ok(p) and vol24(p) > 0
                      and (symbol is None or base_sym(p) == symbol.upper())]
        if not candidates:
            return None
        if wanted_chains:
            preferred = [p for p in candidates
                         if (p.get("chainId") or "").lower() in wanted_chains]
            if preferred:
                candidates = preferred
        # rank by 24h volume first (real activity beats fake deep books)
        return max(candidates, key=lambda p: (vol24(p), liq(p)))

    @staticmethod
    def _add_ds_asset(assets: dict, p: dict) -> None:
        key = (p.get("baseTokenSymbol") or p.get("baseToken", {}).get("symbol") or "").upper()
        if not key or key in assets:
            return
        base = p.get("baseToken") or {}
        assets[key] = {
            "asset": key,
            "name": base.get("name", key),
            "chain": p.get("chainId", ""),
            "price_usd": float(p.get("priceUsd") or 0.0),
            "market_cap_usd": float(p.get("marketCap") or p.get("fdv") or 0.0),
            "volume_24h_usd": float(p.get("volume", {}).get("h24") or 0.0),
            "change_24h_pct": float(p.get("priceChange", {}).get("h24") or 0.0),
            "change_7d_pct": float(p.get("priceChange", {}).get("h7") or 0.0),
            "liquidity_usd": float(p.get("liquidity", {}).get("usd") or 0.0),
            "contract": (base.get("address") or ""),
        }

    # ---------- CoinGecko (fallback) ----------
    def _fetch_coingecko(self) -> tuple[list[dict], str]:
        ids = [i.strip() for i in self._watchlist()
               if i.strip() and not _is_address(i)]
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(ids),
            "order": "market_cap_desc",
            "per_page": 50,
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
        }
        resp = requests.get(url, params=params, timeout=TIMEOUT,
                            headers={"User-Agent": USER_AGENT})
        if resp.status_code == 429:
            raise MarketDataError("CoinGecko rate limited")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or not data:
            raise MarketDataError("CoinGecko returned empty data")
        assets = []
        for c in data:
            assets.append({
                "asset": (c.get("symbol") or "").upper(),
                "name": c.get("name", ""),
                "chain": self._cg_platform(c),
                "price_usd": float(c.get("current_price") or 0.0),
                "market_cap_usd": float(c.get("market_cap") or 0.0),
                "volume_24h_usd": float(c.get("total_volume") or 0.0),
                "change_24h_pct": float(c.get("price_change_percentage_24h") or 0.0),
                "change_7d_pct": float(c.get("price_change_percentage_7d_in_currency") or 0.0),
            })
        return assets, "coingecko"

    @staticmethod
    def _cg_platform(coin: dict) -> str:
        hint = CHAIN_HINTS.get(coin.get("id", ""))
        if hint:
            return hint
        platforms = coin.get("platforms") or {}
        if platforms:
            first = next(iter(platforms))
            return first or "ethereum"
        return "ethereum"
