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


class TestMarketRemovedFromUI:
    def test_public_market_endpoint_removed(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        calls = []
        monkeypatch.setattr(srv.market, "get_market_overview",
                            lambda force=False: calls.append(1) or [])
        c = TestClient(srv.app)
        r = c.get("/api/market")
        assert r.status_code == 404
        assert calls == []  # market data only fetched internally by the engine

    def test_dashboard_has_no_market_section(self):
        html = open(os.path.join(os.path.dirname(__file__), "..", "app", "static", "index.html")).read()
        assert 'id="market"' not in html
        assert 'id="market-table"' not in html
        js = open(os.path.join(os.path.dirname(__file__), "..", "app", "static", "app.js")).read()
        assert "api/market" not in js


class TestSecretStore:
    def test_roundtrip_file_backend(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMICOIN_SECRETS_DIR", str(tmp_path))
        import importlib
        import memicoin_core.secrets as secrets_mod
        importlib.reload(secrets_mod)
        s = secrets_mod.SecretStore()
        s._backend = "file"  # force encrypted-file backend in headless CI
        s.set("exchange_api_key", "supersecret")
        assert s.get("exchange_api_key") == "supersecret"
        assert s.has("exchange_api_secret") is False
        s.set("exchange_api_secret", "another")
        s.delete("exchange_api_key")
        assert s.get("exchange_api_key") is None
        assert s.get("exchange_api_secret") == "another"
        # stored data is encrypted, not plaintext
        raw = (tmp_path / "secrets.encrypted").read_bytes()
        assert b"another" not in raw

    def test_engine_live_requires_configured_executor(self, db, monkeypatch):
        c = Constraints(paper_trading=False)
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews(), live_executor=None)
        ai_out = json.dumps({
            "analysis_summary": "buy",
            "actions": [{"type": "buy", "asset": "DOGE", "amount_or_percent": 10, "reason": "r"}],
            "confidence": "high",
        })
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert len(result["executed"]) == 0
        d = db.list_decisions()[0]
        assert d["status"] == "failed"
        assert "not configured" in d["reason"]


class TestLiveModeGate:
    def test_cannot_flip_to_live_via_constraints(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        body = {"max_loss_pct": 20, "paper_trading": False}
        r = c.post("/api/constraints", json=body)
        assert r.status_code == 400
        assert "go-live" in r.json()["detail"]
        assert load_constraints(db).paper_trading is True

    def test_go_live_requires_configured_keys(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.executor, "configured", lambda: False)
        c = TestClient(srv.app)
        r = c.post("/api/go-live", json={"confirm": True})
        assert r.status_code == 400
        assert "not configured" in r.json()["detail"]

    def test_go_live_then_paper(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.executor, "configured", lambda: True)
        monkeypatch.setattr(srv.executor, "exchange_id", lambda: "binance")
        c = TestClient(srv.app)
        r = c.post("/api/go-live", json={"confirm": True})
        assert r.status_code == 200 and r.json()["mode"] == "live"
        assert load_constraints(db).paper_trading is False
        r = c.post("/api/go-paper")
        assert r.status_code == 200 and load_constraints(db).paper_trading is True

    def test_secrets_never_returned(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "set", lambda k, v: None)
        monkeypatch.setattr(srv.secrets, "backend_name", lambda: "file")
        c = TestClient(srv.app)
        r = c.post("/api/secrets", json={"exchange_id": "binance",
                                         "api_key": "AK123", "api_secret": "SEC456"})
        assert r.status_code == 200
        assert "AK123" not in r.text and "SEC456" not in r.text
        r = c.get("/api/live-status")
        assert "AK123" not in r.text


class TestLiveExecutor:
    def _make_executor(self, fake_ex):
        from memicoin_core.executor import LiveExecutor
        ex = LiveExecutor.__new__(LiveExecutor)
        ex.secrets = None
        ex._exchange = fake_ex
        return ex

    def test_client_order_id_deterministic_and_unique(self):
        from memicoin_core.executor import LiveExecutor
        a = {"type": "buy", "asset": "DOGE", "amount_or_percent": 10}
        id1 = LiveExecutor._client_order_id("c1", a)
        assert id1 == LiveExecutor._client_order_id("c1", a)  # idempotent
        assert id1 != LiveExecutor._client_order_id("c2", a)  # unique per cycle

    def test_sell_reconciles_balances(self):
        ex = self._make_executor(_FakeCCXT(base=500, quote=1000, price=0.5))
        res = ex.sell("c1", {"type": "sell", "asset": "DOGE", "amount": 200})
        assert res.ok, res.error
        assert res.amount == pytest.approx(200)
        assert res.value_usd == pytest.approx(100.0)

    def test_sell_reconciliation_failure_detected(self):
        ex = self._make_executor(_FakeCCXT(base=500, quote=1000, price=0.5, steal=True))
        res = ex.sell("c1", {"type": "sell", "asset": "DOGE", "amount": 200})
        assert not res.ok
        assert "reconciliation failed" in res.error

    def test_buy_rejects_insufficient_balance(self):
        ex = self._make_executor(_FakeCCXT(base=0, quote=10, price=100.0))
        res = ex.buy("c1", {"type": "buy", "asset": "DOGE", "amount": 1})
        assert not res.ok and "insufficient" in res.error

    def test_sell_clamps_to_available_balance(self):
        ex = self._make_executor(_FakeCCXT(base=50, quote=1000, price=2.0))
        res = ex.sell("c1", {"type": "sell", "asset": "DOGE", "amount": 99999})
        assert res.ok
        assert res.amount == pytest.approx(50)


class _FakeCCXT:
    """Minimal CCXT-like exchange double. No network."""
    def __init__(self, base, quote, price, steal=False):
        self.bal = {"DOGE": {"free": base}, "USDC": {"free": quote}}
        self.price = price
        self.steal = steal
        self.orders = []

    def fetch_balance(self):
        return self.bal

    def fetch_ticker(self, symbol):
        return {"last": self.price}

    def create_order(self, symbol, otype, side, amount, params=None):
        self.orders.append((symbol, otype, side, amount, params))
        if side == "buy":
            filled_quote = amount
            avg = self.price
            self.bal["USDC"]["free"] -= filled_quote
            got = filled_quote / avg
            if self.steal:
                got *= 0.5
            self.bal["DOGE"]["free"] += got
            return {"id": "o1", "filled": filled_quote, "average": avg, "status": "closed"}
        filled = amount
        avg = self.price
        self.bal["DOGE"]["free"] -= filled
        got = filled * avg
        if self.steal:
            got *= 0.5
        self.bal["USDC"]["free"] += got
        return {"id": "o1", "filled": filled, "average": avg, "status": "closed"}


class TestOnchainWallet:
    def test_evm_key_import_derives_address(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMICOIN_SECRETS_DIR", str(tmp_path))
        import importlib
        import memicoin_core.secrets as secrets_mod
        importlib.reload(secrets_mod)
        s = secrets_mod.SecretStore()
        s._backend = "file"
        from web3 import Web3
        acct = Web3().eth.account.create()
        s.set("onchain_private_key", acct.key.hex())
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet(s, "base")
        assert w.configured()
        assert w.address() == acct.address

    def test_solana_keypair_import(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMICOIN_SECRETS_DIR", str(tmp_path))
        import importlib
        import memicoin_core.secrets as secrets_mod
        importlib.reload(secrets_mod)
        s = secrets_mod.SecretStore()
        s._backend = "file"
        from solders.keypair import Keypair
        import json as _json
        kp = Keypair()
        s.set("onchain_solana_private_key", _json.dumps(list(bytes(kp))))
        from memicoin_core.onchain import SolanaWallet
        w = SolanaWallet(s)
        assert w.configured()
        assert w.address() == str(kp.pubkey())

    def test_ton_not_enabled(self):
        from memicoin_core.onchain import wallet_for
        with pytest.raises(RuntimeError):
            wallet_for("ton", None)


class TestOnchainAPI:
    def test_import_rejects_bad_chain_and_key(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/wallet/import", json={"chain": "robinhood", "private_key": "x"})
        assert r.status_code == 400
        r = c.post("/api/wallet/import", json={"chain": "ton", "private_key": "x"})
        assert r.status_code == 400 and "not enabled" in r.json()["detail"]

    def test_import_never_echoes_key(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/wallet/import",
                   json={"chain": "ethereum", "private_key": "0x" + "f" * 64})
        assert r.status_code == 400  # invalid key rejected...
        assert "0x" + "f" * 64 not in r.text  # ...and never echoed
        r = c.get("/api/wallet/status")
        assert "0x" not in r.text or r.json()["address"] is None

    def test_onchain_execution_requires_contract_address(self, db, monkeypatch):
        c = Constraints(paper_trading=False, live_execution_mode="onchain",
                        onchain_chain="base")
        from memicoin_core.secrets import SecretStore
        class _NullStore:
            def has(self, k): return True
            def get(self, k): return "0x" + "0" * 64
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews(),
                                onchain_secrets=_NullStore())
        ai_out = json.dumps({
            "analysis_summary": "buy",
            "actions": [{"type": "buy", "asset": "DOGE", "amount_or_percent": 5, "reason": "r"}],
            "confidence": "low",
        })
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        d = db.list_decisions()[0]
        assert d["status"] == "failed"
        assert "contract address" in d["reason"]


class TestArcChain:
    def test_arc_in_supported_chains(self):
        from memicoin_core.onchain import EVM_CHAINS, SUPPORTED_CHAINS
        assert "arc" in EVM_CHAINS
        assert "arc" in SUPPORTED_CHAINS
        assert EVM_CHAINS["arc"]["chain_id"] == 5042

    def test_arc_wallet_import_works_evm_compatible(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMICOIN_SECRETS_DIR", str(tmp_path))
        import importlib
        import memicoin_core.secrets as secrets_mod
        importlib.reload(secrets_mod)
        s = secrets_mod.SecretStore()
        s._backend = "file"
        from web3 import Web3
        acct = Web3().eth.account.create()
        s.set("onchain_private_key", acct.key.hex())
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet(s, "arc")
        assert w.configured()
        assert w.address() == acct.address  # standard EVM key derivation

    def test_arc_swap_refused_without_router(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MEMICOIN_ARC_ROUTER", raising=False)
        from memicoin_core.onchain import EVMWallet
        class _NoRPCWallet(EVMWallet):
            def __init__(self):  # skip RPC connect; only swap-guard is tested
                self.chain = "arc"
                self.cfg = {"router": None, "weth": None, "chain_id": 5042}
                self._acct = None
        w = _NoRPCWallet()
        res = w.swap_exact_tokens_for_tokens("0x" + "1" * 40, "0x" + "2" * 40, 5.0)
        assert not res.ok
        assert "No DEX router configured" in res.error

    def test_arc_router_via_env(self, monkeypatch):
        monkeypatch.setenv("MEMICOIN_ARC_ROUTER", "0x" + "a" * 40)
        from memicoin_core.onchain import EVMWallet
        class _NoRPCWallet(EVMWallet):
            def __init__(self):
                self.chain = "arc"
                self.cfg = {"router": None, "weth": None, "chain_id": 5042}
                self._acct = None
        w = _NoRPCWallet()
        assert w._router() == "0x" + "a" * 40
