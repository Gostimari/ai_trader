"""Market Data Module (§4.1): CoinGecko primary + DexScreener fallback, cached + snapshotted."""
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

# Symbols traded on-chain per chain, used by the DexScreener fallback so we
# only query chains the user actually wants to see.
DEX_SYMBOLS_BY_CHAIN = {
    "ethereum": ["SHIB", "PEPE", "FLOKI", "TURBO", "MOG"],
    "solana": ["BONK", "WIF", "POPCAT", "MEW", "GOAT"],
    "binance-smart-chain": ["FLOKI", "TURBO", "BABYDOGE"],
    "base": ["BRETT", "DEGEN", "TYBG"],
}

# Default watchlist additions beyond the meme-coin list. The user can
# extend this via the watchlist editor (stored as the extra_asset_ids setting).
DEFAULT_EXTRA_IDS = ["circle-wrapped-btc"]

MEMECOIN_IDS = [
    "dogecoin", "shiba-inu", "pepe", "bonk", "floki", "dogwifcoin", "brett-based",
    "popcat", "mog-coin", "cat-in-a-dogs-world", " goatseus-maximus", "turbo",
]
USER_AGENT = "MemeCoinAITrader/0.1 (local desktop app)"
TIMEOUT = 15


class MarketDataError(Exception):
    pass


class MarketDataModule:
    def __init__(self, db: Database):
        self.db = db
        self._cache: dict = {}
        self._cache_ts: float = 0.0
        self.cache_ttl = 300  # seconds

    # ---------- public API ----------
    def get_market_overview(self, force: bool = False) -> list[dict]:
        """List of meme-coin assets with price, changes, volume, mcap, chain."""
        if not force and self._cache and time.time() - self._cache_ts < self.cache_ttl:
            return self._cache["assets"]

        assets, source = [], ""
        try:
            assets, source = self._fetch_coingecko()
        except (requests.RequestException, MarketDataError):
            try:
                assets, source = self._fetch_dexscreener()
            except requests.RequestException as e:
                raise MarketDataError(f"All market data sources failed: {e}") from e

        self._cache = {"assets": assets}
        self._cache_ts = time.time()
        self.db.save_market_snapshot(source, assets)
        return assets

    def get_prices(self, force: bool = False) -> dict[str, float]:
        return {a["asset"]: a["price_usd"] for a in self.get_market_overview(force)}

    def get_asset_map(self, force: bool = False) -> dict[str, dict]:
        return {a["asset"]: a for a in self.get_market_overview(force)}

    # ---------- CoinGecko ----------
    def _fetch_coingecko(self) -> tuple[list[dict], str]:
        extra = self.db.get_setting("extra_asset_ids", DEFAULT_EXTRA_IDS) or []
        ids = [i.strip() for i in list(MEMECOIN_IDS) + list(extra) if i.strip()]
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
        """Best-effort chain attribution: CoinGecko markets endpoint has no
        chain field, so use a curated hint first, then any platform data,
        else default to ethereum."""
        hint = CHAIN_HINTS.get(coin.get("id", ""))
        if hint:
            return hint
        platforms = coin.get("platforms") or {}
        if platforms:
            first = next(iter(platforms))
            return first or "ethereum"
        return "ethereum"

    # ---------- DexScreener fallback ----------
    def _fetch_dexscreener(self) -> tuple[list[dict], str]:
        symbols = sorted({s for syms in DEX_SYMBOLS_BY_CHAIN.values() for s in syms})
        assets: dict[str, dict] = {}
        for sym in symbols:
            url = f"https://api.dexscreener.com/latest/dex/search?q={sym}"
            resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
            resp.raise_for_status()
            pairs = (resp.json() or {}).get("pairs") or []
            best = None
            for p in pairs:
                if (p.get("quoteTokenSymbol") or "").upper() in ("USDC", "USDT", "WETH", "SOL"):
                    best = p
                    break
            if not best:
                continue
            key = (best.get("baseTokenSymbol") or sym).upper()
            if key in assets:
                continue
            assets[key] = {
                "asset": key,
                "name": best.get("baseToken", {}).get("name", key),
                "chain": best.get("chainId", ""),
                "price_usd": float(best.get("priceUsd") or 0.0),
                "market_cap_usd": float(best.get("marketCap") or best.get("fdv") or 0.0),
                "volume_24h_usd": float(best.get("volume", {}).get("h24") or 0.0),
                "change_24h_pct": float(best.get("priceChange", {}).get("h24") or 0.0),
                "change_7d_pct": 0.0,
                "liquidity_usd": float(best.get("liquidity", {}).get("usd") or 0.0),
            }
        if not assets:
            raise MarketDataError("DexScreener returned no usable pairs")
        return list(assets.values()), "dexscreener"
