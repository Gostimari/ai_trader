import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memicoin_core.ai_engine import AIDecision, DecisionEngine
from memicoin_core.constraints import ConstraintValidator
from memicoin_core.db import Database
from memicoin_core.portfolio import PortfolioTracker
from memicoin_core.settings import Constraints, load_constraints


PRICES = {"DOGE": 0.1, "SHIB": 0.00001, "PEPE": 0.0000005}
ASSETS = {
    "DOGE": {"asset": "DOGE", "chain": "binance-smart-chain"},
    "SHIB": {"asset": "SHIB", "chain": "ethereum"},
    "PEPE": {"asset": "PEPE", "chain": "ethereum"},
    "USDC": {"asset": "USDC", "chain": "ethereum"},
}


@pytest.fixture
def db(tmp_path):
    d = Database(f"{tmp_path}/test.db")
    d.set_cash(100.0)
    return d


class TestConstraints:
    def test_max_loss_stop_blocks_all_trading(self, db):
        c = Constraints(max_loss_pct=20.0)
        db.set_cash(0)
        db.upsert_position("DOGE", "binance-smart-chain", 800, 100)  # $80 value, $100 cost
        v = ConstraintValidator(db, c)
        r = v.check_stop_conditions(PRICES)
        assert not r.ok and "Max loss" in r.reason

    def test_profit_target_blocks(self, db):
        c = Constraints(profit_target_pct=10.0)
        db.set_cash(150)  # +50% -> above target
        v = ConstraintValidator(db, c)
        r = v.check_stop_conditions(PRICES)
        assert not r.ok and "Profit target" in r.reason

    def test_allowed_chains_enforced(self, db):
        c = Constraints(allowed_chains=["solana"])
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE", "amount_or_percent": 10}, PRICES, ASSETS)
        assert not r.ok and "not in allowed chains" in r.reason

    def test_max_position_size_enforced(self, db):
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE", "amount_or_percent": 600}, PRICES, ASSETS)
        assert not r.ok and "Max position size" in r.reason

    def test_buy_exceeding_cash_rejected(self, db):
        c = Constraints()
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE", "amount_or_percent": 2000}, PRICES, ASSETS)
        assert not r.ok and "exceeds available cash" in r.reason

    def test_sell_more_than_holdings_rejected(self, db):
        c = Constraints()
        db.upsert_position("DOGE", "binance-smart-chain", 100, 10)
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "sell", "asset": "DOGE", "amount_or_percent": 500}, PRICES, ASSETS)
        assert not r.ok and "exceeds holdings" in r.reason

    def test_unknown_asset_rejected(self, db):
        c = Constraints()
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "SCAMCOIN", "amount_or_percent": 10}, PRICES, ASSETS)
        assert not r.ok and "not found in market data" in r.reason

    def test_unknown_action_type_rejected(self, db):
        c = Constraints()
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "steal", "asset": "DOGE", "amount_or_percent": 1}, PRICES, ASSETS)
        assert not r.ok and "Unknown action type" in r.reason

    def test_emergency_stop_blocks(self, db):
        c = Constraints(emergency_stopped=True)
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE", "amount_or_percent": 10}, PRICES, ASSETS)
        assert not r.ok and "Emergency stop" in r.reason

    def test_paused_blocks(self, db):
        c = Constraints(trading_paused=True)
        v = ConstraintValidator(db, c)
        r = v.check_stop_conditions(PRICES)
        assert not r.ok and "paused" in r.reason

    def test_valid_buy_passes(self, db):
        c = Constraints()
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE", "amount_or_percent": 100}, PRICES, ASSETS)
        assert r.ok

    def test_full_control_still_enforces_position_limit(self, db):
        c = Constraints(autonomy_mode="full_control", max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE", "amount_or_percent": 600}, PRICES, ASSETS)
        assert not r.ok  # technical limits still apply even in full_control


class TestPortfolio:
    def test_pnl_math(self, db):
        t = PortfolioTracker(db)
        t.apply_buy("DOGE", "binance-smart-chain", 500, 50)  # 500 @ $0.10
        prices = {"DOGE": 0.20}
        s = t.summary(prices)
        assert s["positions"][0]["value_usd"] == 100
        assert s["positions"][0]["unrealized_pnl_usd"] == 50
        assert s["cash_usd"] == 50
        assert s["total_value_usd"] == 150
        assert s["total_pnl_pct"] == 50.0

    def test_realized_pnl_on_sell(self, db):
        t = PortfolioTracker(db)
        t.apply_buy("DOGE", "binance-smart-chain", 500, 50)
        pnl = t.apply_sell("DOGE", 250, 60)  # sell half for $60
        assert pnl == pytest.approx(35.0)  # proceeds 60 - cost 25
        assert db.get_cash() == pytest.approx(110)
        p = db.get_position("DOGE")
        assert p["quantity"] == pytest.approx(250)
        assert p["cost_basis_usd"] == pytest.approx(25)

    def test_full_sell_removes_position(self, db):
        t = PortfolioTracker(db)
        t.apply_buy("DOGE", "binance-smart-chain", 100, 10)
        t.apply_sell("DOGE", 100, 15)
        assert db.get_position("DOGE") is None

    def test_average_cost_basis_on_second_buy(self, db):
        t = PortfolioTracker(db)
        t.apply_buy("DOGE", "binance-smart-chain", 100, 10)
        t.apply_buy("DOGE", "binance-smart-chain", 100, 30)
        p = db.get_position("DOGE")
        assert p["quantity"] == 200
        assert p["cost_basis_usd"] == 40


class TestAIParsing:
    def test_valid_response_parses(self):
        raw = json.dumps({
            "analysis_summary": "Market is hot.",
            "actions": [{"type": "buy", "asset": "DOGE", "amount_or_percent": "10%", "reason": "momentum"}],
            "confidence": "medium",
        })
        d = AIDecision.model_validate(json.loads(raw))
        assert d.actions[0].asset == "DOGE"

    def test_markdown_fenced_json_parses(self):
        text = DecisionEngine._parse('```json\n{"analysis_summary":"ok","actions":[],"confidence":"high"}\n```')
        assert text.confidence == "high"

    def test_invalid_confidence_rejected(self):
        with pytest.raises(Exception):
            AIDecision.model_validate({"analysis_summary": "x", "actions": [], "confidence": "certain"})

    def test_invalid_action_type_rejected(self):
        body = {"analysis_summary": "x", "confidence": "high",
                "actions": [{"type": "rugpull", "asset": "DOGE", "amount_or_percent": "1", "reason": "r"}]}
        with pytest.raises(Exception):
            AIDecision.model_validate(body)

    def test_bad_amount_rejected(self):
        body = {"analysis_summary": "x", "confidence": "high",
                "actions": [{"type": "buy", "asset": "DOGE", "amount_or_percent": "lots", "reason": "r"}]}
        with pytest.raises(Exception):
            AIDecision.model_validate(body)

    def test_no_json_rejected(self):
        with pytest.raises(ValueError):
            DecisionEngine._parse("I think you should buy DOGE.")


class TestEngineCycle:
    def test_invalid_ai_response_never_trades(self, db, monkeypatch):
        c = Constraints()
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        monkeypatch.setattr(engine, "_call_mistral", lambda p: "garbage, not json")
        result = engine.run_cycle()
        assert result["status"] == "invalid"
        assert db.get_cash() == 100.0
        assert db.get_positions() == []

    def test_violating_actions_rejected_and_logged(self, db, monkeypatch):
        c = Constraints(max_position_pct=25.0, allowed_chains=["solana"])
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        ai_out = json.dumps({
            "analysis_summary": "Split views.",
            "actions": [
                {"type": "buy", "asset": "DOGE", "amount_or_percent": 600, "reason": "too big"},  # position limit
                {"type": "buy", "asset": "SHIB", "amount_or_percent": 10, "reason": "wrong chain"},
            ],
            "confidence": "low",
        })
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert len(result["rejected"]) == 2
        assert len(result["executed"]) == 0
        assert db.get_cash() == 100.0
        decisions = db.list_decisions()
        assert all(d["status"] == "rejected" for d in decisions)

    def test_valid_cycle_executes_and_logs(self, db, monkeypatch):
        c = Constraints()
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        ai_out = json.dumps({
            "analysis_summary": "DOGE momentum looks strong.",
            "actions": [{"type": "buy", "asset": "DOGE", "amount_or_percent": 200, "reason": "strong 24h momentum"}],
            "confidence": "high",
        })
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert len(result["executed"]) == 1
        assert db.get_cash() == pytest.approx(80.0)
        p = db.get_position("DOGE")
        assert p["quantity"] == pytest.approx(200)
        d = db.list_decisions()[0]
        assert d["status"] == "executed" and d["action"] == "buy"
        assert "momentum" in d["reason"]
        cycle = db.list_cycles()[0]
        assert cycle["status"] == "completed"
        assert cycle["ai_input"] and cycle["ai_output"]  # auditability

    def test_custom_prompt_included_in_context(self, db, monkeypatch):
        c = Constraints(custom_prompt="Never buy PEPE.")
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        captured = {}

        def fake_call(prompt):
            captured["prompt"] = prompt
            return json.dumps({"analysis_summary": "hold", "actions": [], "confidence": "low"})

        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        engine.run_cycle()
        assert "Never buy PEPE." in captured["prompt"]
        assert "max_loss_pct" in captured["prompt"]

    def test_ai_unreachable_no_trades(self, db, monkeypatch):
        c = Constraints()
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())

        def boom(p):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(engine, "_call_mistral", boom)
        result = engine.run_cycle()
        assert result["status"] == "failed"
        assert db.get_cash() == 100.0
        assert db.get_positions() == []


class _FakeMarket:
    def __init__(self):
        self.snap_id = "s1"

    def get_prices(self, force=False):
        return dict(PRICES)

    def get_asset_map(self, force=False):
        return {k: dict(v) for k, v in ASSETS.items()}

    def get_market_overview(self, force=False):
        return list(ASSETS.values())


class _FakeNews:
    def get_news(self, force=False):
        return []


class TestSettings:
    def test_settings_roundtrip_and_defaults(self, db):
        c = load_constraints(db)
        assert c.max_loss_pct == 20.0
        assert c.autonomy_mode == "constrained"
        assert c.paper_trading is True
        c.custom_prompt = "be careful"
        db.set_setting("constraints", c.to_json())
        c2 = load_constraints(db)
        assert c2.custom_prompt == "be careful"


class TestMarketChainAttribution:
    def test_chain_hints_used_first(self):
        from memicoin_core.market import MarketDataModule
        assert MarketDataModule._cg_platform({"id": "bonk"}) == "solana"
        assert MarketDataModule._cg_platform({"id": "dogecoin"}) == "dogecoin"
        assert MarketDataModule._cg_platform({"id": "brett-based"}) == "base"

    def test_platform_fallback_when_no_hint(self):
        from memicoin_core.market import MarketDataModule
        coin = {"id": "some-unknown-token", "platforms": {"binance-smart-chain": "0xabc"}}
        assert MarketDataModule._cg_platform(coin) == "binance-smart-chain"

    def test_default_ethereum(self):
        from memicoin_core.market import MarketDataModule
        assert MarketDataModule._cg_platform({"id": "mystery"}) == "ethereum"


class TestMarketChainFilterAPI:
    def test_filter_returns_only_selected_chains(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        all_assets = [
            {"asset": "BONK", "chain": "solana", "price_usd": 1},
            {"asset": "SHIB", "chain": "ethereum", "price_usd": 2},
            {"asset": "BRETT", "chain": "base", "price_usd": 3},
        ]
        monkeypatch.setattr(srv.market, "get_market_overview", lambda force=False: list(all_assets))
        c = TestClient(srv.app)
        r = c.get("/api/market?chains=solana,base")
        assert r.status_code == 200
        chains = {a["chain"] for a in r.json()["assets"]}
        assert chains == {"solana", "base"}
        assert "ethereum" in r.json()["available_chains"]

    def test_no_filter_returns_all_and_persists_empty(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        all_assets = [{"asset": "BONK", "chain": "solana", "price_usd": 1},
                      {"asset": "SHIB", "chain": "ethereum", "price_usd": 2}]
        monkeypatch.setattr(srv.market, "get_market_overview", lambda force=False: list(all_assets))
        c = TestClient(srv.app)
        r = c.get("/api/market")
        assert len(r.json()["assets"]) == 2
        assert srv.db.get_setting("market_chain_filter") == []

    def test_filter_persisted_and_exposed_in_state(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv.market, "get_market_overview",
                            lambda force=False: [{"asset": "BONK", "chain": "solana", "price_usd": 1}])
        c = TestClient(srv.app)
        c.get("/api/market?chains=solana")
        s = c.get("/api/state").json()
        assert s["market_chain_filter"] == ["solana"]

    def test_filter_is_view_only_engine_sees_full_data(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        all_assets = [{"asset": "BONK", "chain": "solana", "price_usd": 1},
                      {"asset": "SHIB", "chain": "ethereum", "price_usd": 2}]
        monkeypatch.setattr(srv.market, "get_market_overview", lambda force=False: list(all_assets))
        c = TestClient(srv.app)
        c.get("/api/market?chains=solana")
        prices = srv.market.get_prices()
        assert prices == {"BONK": 1, "SHIB": 2}
