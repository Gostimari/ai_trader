"""Market Data Module (§4.1): CoinGecko primary + DexScreener fallback, cached + snapshotted."""
from __future__ import annotations

import time

import requests

from .db import Database

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
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "ids": ",".join(i.strip() for i in MEMECOIN_IDS if i.strip()),
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
        platforms = coin.get("platforms") or {}
        if platforms:
            first = next(iter(platforms))
            return first or "ethereum"
        return "ethereum"

    # ---------- DexScreener fallback ----------
    def _fetch_dexscreener(self) -> tuple[list[dict], str]:
        symbols = ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI", "POPCAT", "TURBO"]
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
