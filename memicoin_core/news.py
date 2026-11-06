"""Crypto news feed (§news of the brief): free RSS feeds, no API key.

Aggregates the major crypto newsrooms' RSS feeds into one list, newest
first. Standard-library XML parsing only; a feed that fails (network
down, layout change) is skipped without breaking the others.
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

USER_AGENT = "MemeCoinAITrader/0.1 (local desktop app)"
TIMEOUT = 12
CACHE_TTL_S = 600  # 10 minutes


class NewsModule:
    def __init__(self):
        self._cache: list[dict] = []
        self._cache_ts: float = 0.0
        self.last_error: str = ""

    def get_news(self, limit: int = 30, force: bool = False) -> list[dict]:
        if not force and self._cache and time.time() - self._cache_ts < CACHE_TTL_S:
            return self._cache[:limit]
        items: list[dict] = []
        errors = []
        for source, url in FEEDS.items():
            try:
                items.extend(self._fetch_feed(source, url))
            except Exception as e:
                errors.append(f"{source}: {e}")
        items.sort(key=lambda x: x.get("published_ts") or 0.0, reverse=True)
        if items:
            self._cache = items
            self._cache_ts = time.time()
            self.last_error = ""
        elif errors:
            self.last_error = "; ".join(errors)
        return items[:limit]

    def _fetch_feed(self, source: str, url: str) -> list[dict]:
        resp = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            if not title or not link:
                continue
            pub = (item.findtext("pubDate") or "").strip()
            published_ts = 0.0
            if pub:
                try:
                    published_ts = parsedate_to_datetime(pub).timestamp()
                except (TypeError, ValueError):
                    published_ts = 0.0
            desc = (item.findtext("description") or "").strip()
            # strip HTML tags from the description; feeds embed markup
            if "<" in desc:
                try:
                    desc = " ".join("".join(ET.fromstring(desc).itertext()).split())
                except ET.ParseError:
                    desc = ""
            items.append({
                "title": title,
                "link": link,
                "source": source,
                "published": pub,
                "published_ts": published_ts,
                "summary": desc[:300],
            })
        return items
