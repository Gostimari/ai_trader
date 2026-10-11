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

    def test_go_live_requires_imported_wallet(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: False)
        c = TestClient(srv.app)
        r = c.post("/api/go-live", json={"confirm": True})
        assert r.status_code == 400
        assert "No wallet imported" in r.json()["detail"]

    def test_go_live_then_paper_with_wallet(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: True)
        c = TestClient(srv.app)
        r = c.post("/api/go-live", json={"confirm": True})
        assert r.status_code == 200 and r.json()["mode"] == "real"
        cc = load_constraints(db)
        assert cc.paper_trading is False and cc.live_execution_mode == "onchain"
        r = c.post("/api/go-paper")
        assert r.status_code == 200 and load_constraints(db).paper_trading is True


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
        r = c.post("/api/wallet/import", json={"chain": "evm", "private_key": "not-a-key"})
        assert r.status_code == 400

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


class TestWatchlist:
    def test_default_watchlist_includes_cirbtc(self, db):
        from memicoin_core.market import DEFAULT_EXTRA_IDS
        assert "circle-wrapped-btc" in DEFAULT_EXTRA_IDS
        assert db.get_setting("extra_asset_ids", DEFAULT_EXTRA_IDS) == DEFAULT_EXTRA_IDS

    def test_cirbtc_chain_hint_is_arc(self):
        from memicoin_core.market import MarketDataModule
        assert MarketDataModule._cg_platform({"id": "circle-wrapped-btc"}) == "arc"

    def test_watchlist_endpoint_roundtrip(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/watchlist", json={"asset_ids": [" bitcoin ", "", "ethereum"]})
        assert r.status_code == 200
        assert r.json()["asset_ids"] == ["bitcoin", "ethereum"]
        assert c.get("/api/watchlist").json()["asset_ids"] == ["bitcoin", "ethereum"]

    def test_dexscreener_primary_includes_watchlist_addresses(self, db, monkeypatch):
        import memicoin_core.market as market_mod
        searched = []
        token_pair_calls = []

        class _FakeReq:
            @staticmethod
            def get(url, params=None, timeout=None, headers=None):
                if "/latest/dex/search" in url:
                    searched.append(params["q"])
                    class _R:
                        status_code = 200
                        def raise_for_status(self): pass
                        def json(self):
                            return {"pairs": [
                                {"chainId": "arc", "baseToken": {"symbol": "CIRBTC",
                                                                 "name": "Circle Wrapped BTC",
                                                                 "address": "0x171a"},
                                 "quoteTokenSymbol": "USDC", "priceUsd": "83000",
                                 "marketCap": "1", "fdv": None,
                                 "volume": {"h24": 100}, "priceChange": {"h24": 2, "h7": 3},
                                 "liquidity": {"usd": 50}}]}
                    return _R()
                if "/tokens/v1/dexscreener/" in url:
                    token_pair_calls.append(url)
                    class _R2:
                        status_code = 200
                        def raise_for_status(self): pass
                        def json(self):
                            return [{"chainId": "arc",
                                     "baseToken": {"symbol": "USYC", "name": "USYC", "address": "0x2222"},
                                     "quoteTokenSymbol": "USDC", "priceUsd": "1.05",
                                     "volume": {"h24": 5}, "priceChange": {"h24": 0.1, "h7": 0.2},
                                     "liquidity": {"usd": 10}}]
                    return _R2()
                class _R3:
                    status_code = 500
                    def raise_for_status(self): raise RuntimeError("no cg")
                    def json(self): return {}
                return _R3()

        monkeypatch.setattr(market_mod, "requests", _FakeReq)
        db.set_setting("extra_asset_ids",
                       ["0x171a4217b86a807a64eb94757db6849fb4bdbaa0"])
        m = market_mod.MarketDataModule(db)
        assets = m.get_market_overview(force=True)
        by_sym = {a["asset"]: a for a in assets}
        assert "CIRBTC" in by_sym and by_sym["CIRBTC"]["chain"] == "arc"
        assert "USYC" in by_sym  # resolved via batch token-pairs endpoint
        assert any("0x171a4217" in u for u in token_pair_calls)
        # snapshot stored with dexscreener source
        snap = db.latest_market_snapshot()
        assert snap["source"] == "dexscreener"

    def test_watchlist_accepts_addresses_and_ids(self):
        from memicoin_core.market import _is_address
        assert _is_address("0x171a4217b86a807a64eb94757db6849fb4bdbaa0")
        assert not _is_address("circle-wrapped-btc")
        assert not _is_address("bitcoin")

    def test_arc_in_default_allowed_chains(self):
        assert "arc" in Constraints().allowed_chains


class TestEvmMultiChainImport:
    def test_one_evm_key_valid_for_all_evm_chains(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        from web3 import Web3
        acct = Web3().eth.account.create()
        r = c.post("/api/wallet/import",
                   json={"chain": "evm", "private_key": acct.key.hex()})
        assert r.status_code == 200
        assert r.json()["chain"] == "evm"
        assert r.json()["address"] == acct.address
        st = c.get("/api/wallet/status").json()
        assert st["evm_configured"] is True
        assert st["evm_address"] == acct.address
        # the same key derives the same address on every EVM chain
        from memicoin_core.onchain import EVMWallet
        for ch in ("ethereum", "base", "polygon", "binance-smart-chain", "arc"):
            assert EVMWallet(srv.secrets, ch).address() == acct.address

    def test_exchange_form_removed_from_ui(self):
        html = open(os.path.join(os.path.dirname(__file__), "..", "app", "static", "index.html")).read()
        assert "secrets-form" not in html
        assert "Live Trading (Phase 2)" not in html
        assert "btn-mode-paper" in html and "btn-mode-real" in html


class TestUserTestGaps:
    def test_wallet_balances_endpoint_unconfigured(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: False)
        c = TestClient(srv.app)
        r = c.get("/api/wallet/balances")
        assert r.status_code == 200
        assert r.json()["configured"] is False

    def test_wallet_balances_valued_from_prices(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        import memicoin_core.onchain as oc
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has",
                            lambda k: k == "onchain_private_key")
        monkeypatch.setattr(oc, "read_wallet_balances", lambda s, chains=None: {
            "evm": {"address": "0xabc", "chains": {
                "arc": {"native_symbol": "USDC", "native": 12.0,
                        "tokens": [{"symbol": "CIRBTC", "contract": "0x171", "balance": 2.0}]}
            }}, "solana": None})
        monkeypatch.setattr(srv.market, "get_prices",
                            lambda force=False: {"USDC": 1.0, "CIRBTC": 83000.0})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        c = TestClient(srv.app)
        r = c.get("/api/wallet/balances")
        d = r.json()
        assert d["configured"] is True
        assert d["total_usd"] == pytest.approx(12.0 + 166000.0)
        arc = d["chains"]["arc"]
        assert arc[0]["symbol"] == "USDC" and arc[0]["value_usd"] == pytest.approx(12.0)
        assert arc[1]["symbol"] == "CIRBTC" and arc[1]["value_usd"] == pytest.approx(166000.0)

    def test_emergency_real_mode_reports_unliquidatable(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        import memicoin_core.onchain as oc
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has",
                            lambda k: k == "onchain_private_key")
        c0 = load_constraints(db)
        c0.paper_trading = False
        db.set_setting("constraints", c0.to_json())
        monkeypatch.setattr(oc, "read_wallet_balances", lambda s, chains=None: {
            "evm": {"address": "0xabc", "chains": {
                "arc": {"native_symbol": "USDC", "native": 5.0,
                        "tokens": [{"symbol": "CIRBTC", "contract": "0x171", "balance": 1.0}]}
            }}, "solana": None})
        c = TestClient(srv.app)
        r = c.post("/api/emergency-to-stablecoin")
        d = r.json()
        assert d["mode"] == "real"
        res = d["results"][0]
        assert res["ok"] is False and "no DEX router" in res["error"]  # honest failure

    def test_setup_status_booleans_only(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        c = TestClient(srv.app)
        d = c.get("/api/setup-status").json()
        assert d["mistral_key"] is False
        assert isinstance(d["wallet"], bool)

    def test_latest_cycle_endpoint(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        assert c.get("/api/cycles/latest").json() == {}
        cid = db.start_cycle()
        db.finish_cycle(cid, "completed", "in", "out", summary="s")
        d = c.get("/api/cycles/latest").json()
        assert d["id"] == cid and d["status"] == "completed"


class TestStablecoinLiquidationAndKey:
    def test_stablecoin_targets_verified_addresses(self):
        from memicoin_core.onchain import stablecoin_target, USDC_ADDRESSES, SOLANA_USDC_MINT
        addr, label = stablecoin_target("ethereum")
        assert addr == "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48" and label == "USDC"
        addr, label = stablecoin_target("base")
        assert addr == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913" and label == "USDC"
        addr, label = stablecoin_target("arc")
        assert label == "native USDC"
        addr, label = stablecoin_target("binance-smart-chain")
        assert label == "wrapped native (no verified USDC on this chain)"
        assert SOLANA_USDC_MINT == "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZ8nMYp7F"

    def test_emergency_real_mode_targets_usdc(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        import memicoin_core.onchain as oc
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has",
                            lambda k: k == "onchain_private_key")
        c0 = load_constraints(db)
        c0.paper_trading = False
        db.set_setting("constraints", c0.to_json())
        monkeypatch.setattr(oc, "read_wallet_balances", lambda s, chains=None: {
            "evm": {"address": "0xabc", "chains": {
                "ethereum": {"native_symbol": "ETH", "native": 0.0,
                             "tokens": [{"symbol": "PEPE", "contract": "0x698250814c", "balance": 100.0}]}
            }}, "solana": None})
        class _FakeW:
            def __init__(self, s, c): self.cfg = {"router": "0xr"}
            def swap_exact_tokens_for_tokens(self, a, b, amt):
                return oc.TxResult(True, tx_id="0xtx", amount=amt)
        monkeypatch.setattr(oc, "EVMWallet", _FakeW)
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        c = TestClient(srv.app)
        r = c.post("/api/emergency-to-stablecoin")
        res = r.json()["results"][0]
        assert res["ok"] is True and res["target"] == "USDC"

    def test_mistral_key_stored_never_returned(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        monkeypatch.setattr(srv.secrets, "has",
                            lambda k: k == "mistral_api_key")
        c = TestClient(srv.app)
        r = c.post("/api/mistral-key", json={"api_key": "secret-key-123"})
        assert r.status_code == 200 and r.json()["configured"] is True
        assert "secret-key-123" not in r.text
        assert c.get("/api/setup-status").json()["mistral_key"] is True

    def test_mistral_key_empty_removes(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        c = TestClient(srv.app)
        r = c.post("/api/mistral-key", json={"api_key": "  "})
        assert r.json()["configured"] is False

    def test_engine_reads_key_from_store(self, db, monkeypatch):
        c = Constraints()
        class _Store:
            def get(self, k): return "stored-key" if k == "mistral_api_key" else None
            def has(self, k): return k == "mistral_api_key"
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews(),
                                onchain_secrets=_Store())
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        # patch at the source the method imports from
        import mistralai.client as msk
        captured = {}
        class _FakeMistral:
            def __init__(self, api_key=None):
                captured["key"] = api_key
                raise RuntimeError("stop here")
        monkeypatch.setattr(msk, "Mistral", _FakeMistral)
        try:
            engine._call_mistral("prompt")
        except RuntimeError as e:
            assert "stop here" in str(e)
        assert captured.get("key") == "stored-key"


class TestPercentAmountBug:
    """Regression: validator and executor must resolve 'N%' identically.
    Bug found in user testing: validator read '50%' as 50 UNITS, executor
    read it as 50% OF PORTFOLIO -> position limits bypassed."""

    def test_percent_buy_rejected_over_position_limit(self, db):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        # AI proposes to spend 100% of the portfolio on DOGE
        r = v.validate_action({"type": "buy", "asset": "DOGE",
                               "amount_or_percent": "100%"}, PRICES, ASSETS)
        assert not r.ok, "buy of 100% of portfolio must be rejected at 25% limit"
        assert "Max position size" in r.reason

    def test_percent_buy_within_limit_passes(self, db):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        r = v.validate_action({"type": "buy", "asset": "DOGE",
                               "amount_or_percent": "20%"}, PRICES, ASSETS)
        assert r.ok, r.reason

    def test_validator_and_executor_resolve_identically(self, db):
        from memicoin_core.constraints import resolve_action_amount
        from memicoin_core.market import MarketDataError
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        e = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": "25%"}
        # validator path
        v_amt = v._resolve_amount(action, PRICES)
        # executor path
        e_amt = e._resolve_amount(action, PRICES)
        assert v_amt == e_amt
        # 25% of $100 portfolio at $0.10 = 250 units
        assert v_amt == pytest.approx(250.0)

    def test_full_cycle_enforces_position_limit_on_percent_buy(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        ai_out = json.dumps({
            "analysis_summary": "All-in on DOGE.",
            "actions": [{"type": "buy", "asset": "DOGE",
                         "amount_or_percent": "100%", "reason": "moon"}],
            "confidence": "high",
        })
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert len(result["rejected"]) == 1
        assert len(result["executed"]) == 0
        assert db.get_cash() == pytest.approx(100.0)  # untouched
        assert db.get_positions() == []
        d = db.list_decisions()[0]
        assert d["status"] == "rejected" and "Max position size" in d["reason"]

    def test_absolute_amount_still_validated(self, db):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        # 900 units * $0.10 = $90 > $25 limit
        r = v.validate_action({"type": "buy", "asset": "DOGE",
                               "amount_or_percent": 900}, PRICES, ASSETS)
        assert not r.ok and "Max position size" in r.reason


class TestDiversifyAndReplan:
    def test_replan_after_full_rejection(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        first = json.dumps({
            "analysis_summary": "all-in",
            "actions": [{"type": "buy", "asset": "DOGE",
                         "amount_or_percent": "100%", "reason": "moon"}],
            "confidence": "high",
        })
        second = json.dumps({
            "analysis_summary": "replan",
            "actions": [{"type": "buy", "asset": "SHIB",
                         "amount_or_percent": "20%", "reason": "fits limits"}],
            "confidence": "medium",
        })
        calls = []
        def fake_call(prompt):
            calls.append(prompt)
            return first if len(calls) == 1 else second
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        result = engine.run_cycle()
        assert len(calls) == 2                      # replan happened
        assert "rejected by the user's constraints" in calls[1]
        assert len(result["executed"]) == 1
        assert result["executed"][0]["asset"] == "SHIB"
        assert db.get_position("SHIB") is not None

    def test_no_replan_when_something_executed(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        out = json.dumps({
            "analysis_summary": "mixed",
            "actions": [
                {"type": "buy", "asset": "SHIB", "amount_or_percent": "20%", "reason": "ok"},
                {"type": "buy", "asset": "DOGE", "amount_or_percent": "100%", "reason": "big"},
            ],
            "confidence": "high",
        })
        calls = []
        monkeypatch.setattr(engine, "_call_mistral", lambda p: calls.append(p) or out)
        result = engine.run_cycle()
        assert len(calls) == 1                      # no replan round
        assert len(result["executed"]) == 1 and len(result["rejected"]) == 1

    def test_replan_failure_is_safe(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), _FakeNews())
        first = json.dumps({
            "analysis_summary": "all-in",
            "actions": [{"type": "buy", "asset": "DOGE",
                         "amount_or_percent": "100%", "reason": "moon"}],
            "confidence": "high",
        })
        def fake_call(prompt):
            # the replan prompt contains the rejection feedback marker
            return first if "were ALL rejected" not in prompt else "garbage not json"
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert result["executed"] == []
        assert db.get_cash() == pytest.approx(100.0)  # untouched either way

    def test_prompt_teaches_diversification(self):
        from memicoin_core.ai_engine import BASE_INSTRUCTIONS
        assert "DIVERSIFY" in BASE_INSTRUCTIONS
        assert "MULTIPLE buys" in BASE_INSTRUCTIONS


class TestPerPositionStopLoss:
    def _engine(self, db, stop=20.0):
        c = Constraints(max_position_pct=25.0, per_position_stop_pct=stop)
        return DecisionEngine(db, c, _FakeMarket(), _FakeNews())

    def test_losing_position_auto_sold_at_cycle_start(self, db, monkeypatch):
        db.set_cash(100.0)
        engine = self._engine(db, stop=20.0)
        # position bought at $0.10, now worth $0.07 -> -30%
        db.upsert_position("DOGE", "binance-smart-chain", 500, 50)
        db.set_cash(50.0)
        prices = {"DOGE": 0.07, "SHIB": 0.00001, "PEPE": 0.0000005}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: dict(ASSETS))
        ai_out = json.dumps({"analysis_summary": "hold", "actions": [],
                             "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        # DOGE sold: position gone, cash got the proceeds
        assert db.get_position("DOGE") is None
        assert db.get_cash() == pytest.approx(50.0 + 500 * 0.07)
        d = db.list_decisions()[0]
        assert d["action"] == "sell" and d["status"] == "executed"
        assert "automatic stop-loss" in d["reason"]
        assert "-30.0%" in d["reason"]

    def test_healthy_position_not_sold(self, db, monkeypatch):
        db.set_cash(100.0)
        engine = self._engine(db, stop=20.0)
        db.upsert_position("DOGE", "binance-smart-chain", 500, 50)  # +40% at 0.14
        db.set_cash(50.0)
        prices = {"DOGE": 0.14, "SHIB": 0.00001, "PEPE": 0.0000005}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: dict(ASSETS))
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: json.dumps({"analysis_summary": "h",
                                                  "actions": [], "confidence": "low"}))
        engine.run_cycle()
        assert db.get_position("DOGE") is not None

    def test_stop_disabled_at_zero(self, db, monkeypatch):
        db.set_cash(100.0)
        engine = self._engine(db, stop=0)
        db.upsert_position("DOGE", "binance-smart-chain", 500, 50)
        db.set_cash(50.0)
        prices = {"DOGE": 0.01, "SHIB": 0.00001, "PEPE": 0.0000005}  # -80%
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: dict(ASSETS))
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: json.dumps({"analysis_summary": "h",
                                                  "actions": [], "confidence": "low"}))
        engine.run_cycle()
        assert db.get_position("DOGE") is not None  # stop disabled: kept

    def test_boundary_exactly_at_threshold_sells(self, db, monkeypatch):
        db.set_cash(100.0)
        engine = self._engine(db, stop=20.0)
        db.upsert_position("DOGE", "binance-smart-chain", 500, 50)
        db.set_cash(50.0)
        prices = {"DOGE": 0.08, "SHIB": 0.00001, "PEPE": 0.0000005}  # exactly -20%
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: dict(ASSETS))
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: json.dumps({"analysis_summary": "h",
                                                  "actions": [], "confidence": "low"}))
        engine.run_cycle()
        assert db.get_position("DOGE") is None


class TestPairPickerQuality:
    def _pair(self, chain, liq, vol, base="BONK", quote=None):
        return {"chainId": chain, "liquidity": {"usd": liq}, "volume": {"h24": vol},
                "quoteTokenSymbol": quote,
                "baseToken": {"symbol": base, "name": base, "address": "0x1"},
                "priceUsd": "1.0", "priceChange": {"h24": 1, "h7": 2}}

    def test_fake_chain_rejected(self):
        from memicoin_core.market import MarketDataModule
        assert MarketDataModule._best_pair([self._pair("robinhood", 999, 999)]) is None

    def test_zero_volume_imposter_rejected(self):
        from memicoin_core.market import MarketDataModule
        assert MarketDataModule._best_pair([self._pair("ethereum", 10**9, 0)]) is None

    def test_symbol_mismatch_imposter_rejected(self):
        from memicoin_core.market import MarketDataModule
        # an imposter token named "SI" showing up in a WIF search
        assert MarketDataModule._best_pair(
            [self._pair("solana", 5470, 110410, base="SI")], symbol="WIF") is None

    def test_prefers_allowed_chain_over_higher_volume(self):
        from memicoin_core.market import MarketDataModule
        eth = self._pair("ethereum", 10**6, 500)
        sol = self._pair("solana", 10**5, 400)
        best = MarketDataModule._best_pair([eth, sol], wanted_chains={"solana"})
        assert best["chainId"] == "solana"

    def test_ranks_by_volume_not_liquidity(self):
        from memicoin_core.market import MarketDataModule
        deep = self._pair("ethereum", 10**9, 10)      # fake deep book
        active = self._pair("ethereum", 10**4, 10**6)  # real activity
        assert MarketDataModule._best_pair([deep, active])["liquidity"]["usd"] == 10**4


class TestSeedPhraseImport:
    def test_seed_phrase_derives_known_address(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/wallet/import", json={
            "chain": "evm",
            "seed_phrase": "test test test test test test test test test test test junk"})
        assert r.status_code == 200
        assert r.json()["address"] == "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"
        assert "seed" in r.json()["derived_from"]
        st = c.get("/api/wallet/status").json()
        assert st["evm_address"] == "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"

    def test_bad_word_count_rejected(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/wallet/import", json={
            "chain": "evm", "seed_phrase": "one two three four five six"})
        assert r.status_code == 400
        assert "12/15/18/21/24" in r.json()["detail"]

    def test_invalid_seed_phrase_rejected(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/wallet/import", json={
            "chain": "evm",
            "seed_phrase": " ".join(["notbip39word"] * 12)})
        assert r.status_code == 400

    def test_seed_for_solana_rejected_honestly(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/wallet/import", json={
            "chain": "solana",
            "seed_phrase": "test test test test test test test test test test test junk"})
        assert r.status_code == 400
        assert "Solana" in r.json()["detail"]

    def test_seed_never_echoed(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        seed = "test test test test test test test test test test test junk"
        r = c.post("/api/wallet/import", json={"chain": "evm", "seed_phrase": seed})
        assert seed not in r.text
        assert "test test" not in r.text
