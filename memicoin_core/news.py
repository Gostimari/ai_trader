"""News Module (§4.5): crypto news feed, cached locally so the UI never blocks."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

from .db import Database, new_id

SAMPLE_NEWS = [
    {"title": "Meme coin sector rebounds as major exchanges list new tokens",
     "source": "sample", "url": "", "published": ""},
    {"title": "On-chain volumes spike for Solana meme coins",
     "source": "sample", "url": "", "published": ""},
    {"title": "Regulators issue reminder on speculative token volatility",
     "source": "sample", "url": "", "published": ""},
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class NewsModule:
    def __init__(self, db: Database):
        self.db = db

    def get_news(self, force: bool = False) -> list[dict]:
        cached = self.db.latest_news()
        if not force and cached:
            import json
            return json.loads(cached["payload"])
        items = self._fetch()
        self.db.save_news(items)
        return items

    def _fetch(self) -> list[dict]:
        key = os.environ.get("NEWS_API_KEY", "")
        if key:
            try:
                resp = requests.get(
                    "https://cryptopanic.com/api/v1/posts/",
                    params={"auth_token": key, "currencies": "BTC,DOGE,SHIB", "public": "true"},
                    timeout=15,
                )
                resp.raise_for_status()
                posts = (resp.json() or {}).get("results") or []
                return [
                    {
                        "title": p.get("title", ""),
                        "source": (p.get("source") or {}).get("title", ""),
                        "url": p.get("url", ""),
                        "published": p.get("published_at", ""),
                    }
                    for p in posts[:30]
                ]
            except requests.RequestException:
                pass
        return SAMPLE_NEWS
