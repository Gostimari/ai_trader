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


def make_engine() -> DecisionEngine:
    return DecisionEngine(db, load_constraints(db), market, news)


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


@app.get("/api/market")
def market_overview():
    try:
        return {"assets": market.get_market_overview(force=True)}
    except MarketDataError as e:
        raise HTTPException(503, str(e))


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
    db.set_setting("constraints", c.to_json())
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
    try:
        prices = market.get_prices()
    except MarketDataError as e:
        raise HTTPException(503, f"Cannot liquidate: market data unavailable ({e})")
    cycle_id = db.start_cycle()
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
    return {"ok": True}


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
