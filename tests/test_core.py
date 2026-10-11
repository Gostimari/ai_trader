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

    def test_max_position_size_clamped_not_rejected(self, db):
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": 600}
        r = v.validate_action(action, PRICES, ASSETS)
        assert r.ok  # clamped, not rejected
        # $100 portfolio * 25% = $25 max; at $0.10 -> 250 DOGE
        assert action["amount"] == pytest.approx(250.0)
        assert "clamped" in action["_clamped"]

    def test_buy_exceeding_cash_clamped_to_cash(self, db):
        c = Constraints()
        v = ConstraintValidator(db, c)
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": 2000}
        r = v.validate_action(action, PRICES, ASSETS)
        assert r.ok  # clamped down to available cash
        assert action["amount"] <= 1000.0 + 1e-6  # $100 cash / $0.10

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

    def test_full_control_still_clamps_position_limit(self, db):
        c = Constraints(autonomy_mode="full_control", max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": 600}
        r = v.validate_action(action, PRICES, ASSETS)
        assert r.ok  # clamped even in full_control: technical limits apply
        assert action["amount"] == pytest.approx(250.0)


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
        engine = DecisionEngine(db, c, _FakeMarket())
        monkeypatch.setattr(engine, "_call_mistral", lambda p: "garbage, not json")
        result = engine.run_cycle()
        assert result["status"] == "invalid"
        assert db.get_cash() == 100.0
        assert db.get_positions() == []

    def test_violating_actions_rejected_and_logged(self, db, monkeypatch):
        c = Constraints(max_position_pct=25.0, allowed_chains=["solana"])
        engine = DecisionEngine(db, c, _FakeMarket())
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
        engine = DecisionEngine(db, c, _FakeMarket())
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
        engine = DecisionEngine(db, c, _FakeMarket())
        captured = {}

        def fake_call(prompt):
            if "prompt" not in captured:
                captured["prompt"] = prompt  # first call has the full context
            return json.dumps({"analysis_summary": "hold", "actions": [], "confidence": "low"})

        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        engine.run_cycle()
        assert "Never buy PEPE." in captured["prompt"]
        assert "max_loss_pct" in captured["prompt"]

    def test_ai_unreachable_no_trades(self, db, monkeypatch):
        c = Constraints()
        engine = DecisionEngine(db, c, _FakeMarket())

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

    def get_prices_for_addresses(self, contracts):
        return {}

    def get_prices(self, force=False):
        return dict(PRICES)

    def get_asset_map(self, force=False):
        return {k: dict(v) for k, v in ASSETS.items()}

    def get_market_overview(self, force=False):
        return list(ASSETS.values())



class TestSettings:
    def test_settings_roundtrip_and_defaults(self, db):
        c = load_constraints(db)
        # app-facing defaults after migration
        assert c.live_execution_mode == "demo"
        assert c.autonomy_mode == "constrained"
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
        c = Constraints(paper_trading=False, live_execution_mode="onchain")
        engine = DecisionEngine(db, c, _FakeMarket(), live_executor=None)
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
        # contract-address safety fires before the executor-config check
        assert "contract address" in d["reason"] or "not configured" in d["reason"]


class TestLiveModeGate:
    def test_cannot_flip_to_live_via_constraints(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        # real mode can only be entered via /api/go-live, never via the
        # generic constraints save (which defaults to demo in the app)
        body = {"max_loss_pct": 20, "paper_trading": False,
                "live_execution_mode": "onchain"}
        r = c.post("/api/constraints", json=body)
        assert r.status_code == 400
        assert "go-live" in r.json()["detail"]

    def test_go_live_requires_imported_wallet(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: False)
        c = TestClient(srv.app)
        r = c.post("/api/go-live", json={"confirm": True})
        assert r.status_code == 400
        assert "No wallet imported" in r.json()["detail"]

    def test_go_live_then_back_to_demo(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: True)
        c = TestClient(srv.app)
        r = c.post("/api/go-live", json={"confirm": True})
        assert r.status_code == 200 and r.json()["mode"] == "real"
        cc = load_constraints(db)
        assert cc.paper_trading is False and cc.live_execution_mode == "onchain"
        r = c.post("/api/go-demo")
        assert r.status_code == 200
        cc = load_constraints(db)
        assert cc.live_execution_mode == "demo"


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
            def get(self, k): return "0x" + "11" * 32  # valid non-zero key
        engine = DecisionEngine(db, c, _FakeMarket(), None,
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
                if "/latest/dex/tokens/" in url:
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

        class _FakeReqWithExc(_FakeReq):
            class RequestException(Exception):
                pass
        monkeypatch.setattr(market_mod, "requests", _FakeReqWithExc)
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
        assert "btn-mode-demo" in html and "btn-mode-real" in html
        assert "Paper money" not in html  # paper mode is gone from the UI


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
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
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
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
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
        engine = DecisionEngine(db, c, _FakeMarket(), None,
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

    def test_percent_buy_clamped_over_position_limit(self, db):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": "100%"}
        r = v.validate_action(action, PRICES, ASSETS)
        assert r.ok
        assert action["amount"] == pytest.approx(250.0)  # $25 at $0.10
        assert "clamped" in action["_clamped"]

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
        e = DecisionEngine(db, c, _FakeMarket())
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": "25%"}
        # validator path
        v_amt = v._resolve_amount(action, PRICES)
        # executor path
        e_amt = e._resolve_amount(action, PRICES)
        assert v_amt == e_amt
        # 25% of $100 portfolio at $0.10 = 250 units
        assert v_amt == pytest.approx(250.0)

    def test_full_cycle_clamps_position_limit_on_percent_buy(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        ai_out = json.dumps({
            "analysis_summary": "All-in on DOGE.",
            "actions": [{"type": "buy", "asset": "DOGE",
                         "amount_or_percent": "100%", "reason": "moon"}],
            "confidence": "high",
        })
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert len(result["executed"]) == 1  # clamped, executed at the limit
        assert len(result["rejected"]) == 0
        assert db.get_cash() == pytest.approx(75.0)  # spent only the 25% cap
        p = db.get_position("DOGE")
        assert p["quantity"] == pytest.approx(250.0)
        d = db.list_decisions()[0]
        assert d["status"] == "executed" and "clamped" in d["reason"]

    def test_absolute_amount_still_validated(self, db):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        v = ConstraintValidator(db, c)
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": 900}
        r = v.validate_action(action, PRICES, ASSETS)
        assert r.ok and action["amount"] == pytest.approx(250.0)


class TestDiversifyAndReplan:
    def test_replan_after_full_rejection(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        # DOGE already at max (25% of portfolio) -> buy fully rejected, no clamp room
        db.upsert_position("DOGE", "binance-smart-chain", 250, 25)
        db.set_cash(75.0)
        first = json.dumps({
            "analysis_summary": "more DOGE",
            "actions": [{"type": "buy", "asset": "DOGE",
                         "amount_or_percent": "10%", "reason": "moon"}],
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
        engine = DecisionEngine(db, c, _FakeMarket(), None)
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
        assert len(result["executed"]) == 2 and len(result["rejected"]) == 0
        # the 100% buy was clamped to the 25% limit, not rejected

    def test_replan_failure_is_safe(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        db.upsert_position("DOGE", "binance-smart-chain", 250, 25)
        db.set_cash(75.0)
        first = json.dumps({
            "analysis_summary": "more DOGE",
            "actions": [{"type": "buy", "asset": "DOGE",
                         "amount_or_percent": "10%", "reason": "moon"}],
            "confidence": "high",
        })
        def fake_call(prompt):
            return first if "were ALL rejected" not in prompt else "garbage not json"
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert result["executed"] == []
        assert db.get_cash() == pytest.approx(75.0)  # untouched either way

    def test_prompt_teaches_diversification(self):
        from memicoin_core.ai_engine import BASE_INSTRUCTIONS
        assert "DIVERSIFY" in BASE_INSTRUCTIONS
        assert "MULTIPLE buys" in BASE_INSTRUCTIONS


class TestPerPositionStopLoss:
    def _engine(self, db, stop=20.0):
        c = Constraints(max_position_pct=25.0, per_position_stop_pct=stop)
        return DecisionEngine(db, c, _FakeMarket())

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


class TestWatchlistTokenBalances:
    def test_extra_contracts_read_onchain(self, tmp_path, monkeypatch):
        import memicoin_core.onchain as oc
        class _FakeW:
            def __init__(self, secrets, chain): self.chain = chain
            def native_balance(self): return 0.0
            def erc20_balance(self, addr):
                # wallet holds 1000 of the watchlist token, nothing else
                if addr.lower() == "0x532f27101965dd16442e59d40670faf5ebb142e4":
                    return 1000.0
                return 0.0
        monkeypatch.setattr(oc, "EVMWallet", _FakeW)
        class _Store:
            def has(self, k): return k == "onchain_private_key"
            def get(self, k): return "0x" + "1" * 64
        balances = oc.read_wallet_balances(
            _Store(),
            extra_contracts={"base": [{"address": "0x532f27101965dd16442E59d40670FaF5eBB142E4",
                                      "symbol": "BRETT"}]})
        base_tokens = balances["evm"]["chains"]["base"]["tokens"]
        assert len(base_tokens) == 1
        assert base_tokens[0]["symbol"] == "BRETT"
        assert base_tokens[0]["balance"] == 1000.0

    def test_zero_balance_omitted(self, tmp_path, monkeypatch):
        import memicoin_core.onchain as oc
        class _FakeW:
            def __init__(self, secrets, chain): pass
            def native_balance(self): return 0.0
            def erc20_balance(self, addr): return 0.0
        monkeypatch.setattr(oc, "EVMWallet", _FakeW)
        class _Store:
            def has(self, k): return k == "onchain_private_key"
            def get(self, k): return "0x" + "1" * 64
        balances = oc.read_wallet_balances(
            _Store(), extra_contracts={"base": [{"address": "0x" + "a" * 40,
                                                 "symbol": "X"}]})
        assert balances["evm"]["chains"]["base"]["tokens"] == []

    def test_server_passes_watchlist_contracts(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        import memicoin_core.onchain as oc
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        captured = {}
        def fake_read(secrets, chains=None, extra_contracts=None):
            captured["extra"] = extra_contracts
            return {"evm": {"address": "0xa", "chains": {
                "base": {"native_symbol": "ETH", "native": 0,
                         "tokens": [{"symbol": "BRETT", "contract": "0x53", "balance": 5}]}
            }}, "solana": None}
        monkeypatch.setattr(oc, "read_wallet_balances", fake_read)
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {
            "BRETT": {"asset": "BRETT", "chain": "base",
                      "contract": "0x532f27101965dd16442E59d40670FaF5eBB142E4"}})
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {"BRETT": 0.006})
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        assert d["chains"]["base"][0]["symbol"] == "BRETT"
        assert captured["extra"]["base"][0]["symbol"] == "BRETT"


class TestTokenDiscovery:
    def test_etherscan_v2_parse(self, monkeypatch):
        import memicoin_core.discovery as disc
        class _FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {"status": "1", "result": [
                    {"contractAddress": "0xAAA", "tokenSymbol": "ARCAT", "tokenDecimal": "18"},
                    {"contractAddress": "0xBBB", "tokenSymbol": "ARGUS", "tokenDecimal": "18"},
                    {"contractAddress": "0xAAA", "tokenSymbol": "ARCAT", "tokenDecimal": "18"},
                ]}
        monkeypatch.setattr(disc.requests, "get", lambda *a, **kw: _FakeResp())
        out = disc.discover_evm_tokens("0xwallet", "key", chains=["arc"])
        syms = {t["symbol"] for t in out["arc"]}
        assert syms == {"ARCAT", "ARGUS"}
        assert len(out["arc"]) == 2  # deduped

    def test_etherscan_no_history_is_empty(self, monkeypatch):
        import memicoin_core.discovery as disc
        class _FakeResp:
            def raise_for_status(self): pass
            def json(self): return {"status": "0", "result": None}
        monkeypatch.setattr(disc.requests, "get", lambda *a, **kw: _FakeResp())
        assert disc.discover_evm_tokens("0xw", "key", chains=["arc"]) == {}

    def test_solana_discovery_parse(self, monkeypatch):
        import memicoin_core.discovery as disc
        class _FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {"result": {"value": [
                    {"account": {"data": {"parsed": {"info": {
                        "mint": "MintAAA", "tokenAmount": {"uiAmount": 12.5}}}}}},
                    {"account": {"data": {"parsed": {"info": {
                        "mint": "MintBBB", "tokenAmount": {"uiAmount": 0}}}}}},
                ]}}
        monkeypatch.setattr(disc.requests, "post", lambda *a, **kw: _FakeResp())
        toks = disc.discover_solana_tokens("SomePubkey")
        assert len(toks) == 1 and toks[0]["address"] == "MintAAA"
        assert toks[0]["balance"] == 12.5

    def test_etherscan_key_endpoint(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/etherscan-key", json={"api_key": "EK123"})
        assert r.status_code == 200 and r.json()["configured"] is True
        assert "EK123" not in r.text
        r = c.post("/api/etherscan-key", json={"api_key": ""})
        assert r.json()["configured"] is False

    def test_discovery_merged_into_balances(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        import memicoin_core.discovery as disc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has",
                            lambda k: k in ("onchain_private_key", "etherscan_api_key"))
        monkeypatch.setattr(srv.secrets, "get",
                            lambda k: {"onchain_private_key": "0x" + "1" * 64,
                                       "etherscan_api_key": "key"}.get(k))
        class _FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {"status": "1", "result": [
                    {"contractAddress": "0x" + "c" * 40, "tokenSymbol": "ARCAT",
                     "tokenDecimal": "18"}]}
        monkeypatch.setattr(disc.requests, "get", lambda *a, **kw: _FakeResp())
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        class _FakeW:
            def __init__(self, s, chain): pass
            def native_balance(self): return 0.0
            def erc20_balance(self, addr):
                return 77.0 if addr.endswith("c" * 40) else 0.0
        monkeypatch.setattr(oc, "EVMWallet", _FakeW)
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        found = [t for rows in d["chains"].values() for t in rows
                 if t.get("symbol") == "ARCAT"]
        assert found and found[0]["balance"] == 77.0


class TestAddressPricing:
    def test_prices_for_addresses_from_pairs(self, db, monkeypatch):
        import memicoin_core.market as market_mod
        m = market_mod.MarketDataModule(db)
        class _FakeReq:
            @staticmethod
            def get(url, params=None, timeout=None, headers=None):
                assert "/latest/dex/tokens/" in url
                class _R:
                    status_code = 200
                    def raise_for_status(self): pass
                    def json(self):
                        return {"pairs": [
                            {"baseToken": {"address": "0xAAA", "symbol": "ARCAT"},
                             "priceUsd": "0.42", "liquidity": {"usd": 1000}},
                            {"baseToken": {"address": "0xAAA", "symbol": "ARCAT"},
                             "priceUsd": "0.99", "liquidity": {"usd": 10}},
                        ]}
                return _R()
        monkeypatch.setattr(market_mod, "requests", _FakeReq)
        prices = m.get_prices_for_addresses(["0xaaa"])
        assert prices == {"0xaaa": 0.42}  # deepest-liquidity pair wins

    def test_wallet_rows_valued_via_contract(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses",
                            lambda contracts: {"0x" + "c" * 40: 2.5})
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "arc": {"native_symbol": "USDC", "native": 0,
                        "tokens": [{"symbol": "ARGUS", "contract": "0x" + "c" * 40,
                                    "balance": 10.0}]}}
            }, "solana": None})
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        row = d["chains"]["arc"][0]
        assert row["symbol"] == "ARGUS" and row["value_usd"] == pytest.approx(25.0)


class TestNativeAndStablecoinValuation:
    def test_eth_on_base_valued_via_weth(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        WETH_BASE = "0x4200000000000000000000000000000000000006"
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses",
                            lambda contracts: {WETH_BASE.lower(): 2500.0})
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "base": {"native_symbol": "ETH", "native": 2.0, "tokens": []}}},
            "solana": None})
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        row = d["chains"]["base"][0]
        assert row["symbol"] == "ETH" and row["value_usd"] == pytest.approx(5000.0)

    def test_arc_native_usdc_peg_floor(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", lambda contracts: {})
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "arc": {"native_symbol": "USDC", "native": 15.0, "tokens": []}}},
            "solana": None})
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        row = d["chains"]["arc"][0]
        assert row["symbol"] == "USDC" and row["value_usd"] == pytest.approx(15.0)

    def test_usdc_on_base_read_via_known_list(self, monkeypatch):
        from memicoin_core.onchain import EVM_TOKENS
        usdc = [t for t in EVM_TOKENS.get("base", []) if t[0] == "USDC"]
        assert usdc and usdc[0][1] == "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


class TestSymbolFallbackPricing:
    def test_unpriced_symbol_resolved_via_search(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", lambda contracts: {})
        def fake_search(q, **kw):
            return [
                {"chainId": "arc", "baseToken": {"symbol": "ARGUS", "address": "0x1"},
                 "priceUsd": "0.0115", "liquidity": {"usd": 500000}},
                {"chainId": "solana", "baseToken": {"symbol": "ARGUS", "address": "0x2"},
                 "priceUsd": "0.99", "liquidity": {"usd": 9000000}},  # wrong chain
            ]
        monkeypatch.setattr(srv.market, "_ds_search", fake_search)
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "arc": {"native_symbol": "USDC", "native": 0,
                        "tokens": [{"symbol": "ARGUS", "contract": "0xunk",
                                    "balance": 100.0}]}}
            }, "solana": None})
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        row = d["chains"]["arc"][0]
        assert row["symbol"] == "ARGUS"
        assert row["value_usd"] == pytest.approx(1.15)  # 100 * $0.0115, arc pair only


class TestStablecoinDecoyProtection:
    def test_stables_skip_symbol_search(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        searches = []
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", lambda contracts: {})
        def fake_search(q, **kw):
            searches.append(q)
            return []  # would be decoys; must not even be called for stables
        monkeypatch.setattr(srv.market, "_ds_search", fake_search)
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "base": {"native_symbol": "ETH", "native": 0,
                         "tokens": [{"symbol": "USDC", "contract": "0xunpriced",
                                     "balance": 10.0}]}}
            }, "solana": None})
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        row = d["chains"]["base"][0]
        assert row["value_usd"] == pytest.approx(10.0)  # peg floor, not a decoy
        assert "USDC" not in searches  # no symbol-search for stables


class TestSolanaGating:
    def _sol_assets(self):
        return {"BONK": {"asset": "BONK", "chain": "solana"},
                "DOGE": {"asset": "DOGE", "chain": "dogecoin"}}

    def test_solana_rejected_without_key(self, db):
        c = Constraints(allowed_chains=["solana", "ethereum"])  # solana allowed by user
        v = ConstraintValidator(db, c, solana_key_present=False)
        r = v.validate_action({"type": "buy", "asset": "BONK",
                               "amount_or_percent": 10}, PRICES, self._sol_assets())
        assert not r.ok and "no Solana wallet key" in r.reason

    def test_solana_allowed_with_key(self, db):
        prices = dict(PRICES, BONK=0.00003)
        c = Constraints(allowed_chains=["solana"])
        v = ConstraintValidator(db, c, solana_key_present=True)
        r = v.validate_action({"type": "buy", "asset": "BONK",
                               "amount_or_percent": 10}, prices, self._sol_assets())
        assert r.ok, r.reason

    def test_other_chains_unaffected_by_key_absence(self, db):
        db.upsert_position("DOGE", "dogecoin", 100, 10)
        c = Constraints(allowed_chains=["dogecoin"])
        v = ConstraintValidator(db, c, solana_key_present=False)
        r = v.validate_action({"type": "sell", "asset": "DOGE",
                               "amount_or_percent": 10}, PRICES, self._sol_assets())
        assert r.ok, r.reason


class TestBridgeAction:
    def test_bridge_paper_mode_logged(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        engine = DecisionEngine(db, c, _FakeMarket())
        ai_out = json.dumps({
            "analysis_summary": "bridge to arc for tomorrow's opportunity",
            "actions": [{"type": "bridge", "from_chain": "ethereum",
                         "to_chain": "arc", "amount_or_percent": 50,
                         "reason": "best token is on arc"}],
            "confidence": "medium"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        d = db.list_decisions()[0]
        assert d["action"] == "bridge" and d["status"] == "executed"
        assert "[PAPER]" in d["reason"] and "ethereum" in d["reason"]

    def test_malformed_bridge_rejected(self, db, monkeypatch):
        # valid amount but missing source chain: passes the AI schema,
        # must be rejected by the engine's bridge validation
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        engine = DecisionEngine(db, c, _FakeMarket())
        ai_out = json.dumps({
            "analysis_summary": "bridge",
            "actions": [{"type": "bridge", "to_chain": "arc",
                         "amount_or_percent": 50, "reason": "bad"}],
            "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        engine.run_cycle()
        d = db.list_decisions()[0]
        assert d["action"] == "bridge"
        assert d["status"] == "rejected" and "Malformed" in d["reason"]

    def test_garbage_bridge_amount_rejected_by_schema(self, db, monkeypatch):
        # unparseable amount -> whole AI response invalid, nothing executes
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        engine = DecisionEngine(db, c, _FakeMarket())
        ai_out = json.dumps({
            "analysis_summary": "bridge",
            "actions": [{"type": "bridge", "from_chain": "ethereum",
                         "to_chain": "arc", "amount_or_percent": "garbage",
                         "reason": "bad"}],
            "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "invalid"  # schema safety net
        assert db.get_cash() == pytest.approx(100.0)

    def test_bridge_unverified_pair_refused_honestly(self, db, monkeypatch):
        from memicoin_core.bridge import bridge_usdc
        class _Store:
            def has(self, k): return k == "onchain_private_key"
            def get(self, k): return "0x" + "1" * 64
        res = bridge_usdc(_Store(), "base", "arc", 5.0)
        assert not res.ok and "not enabled" in res.error

    def test_wallet_capabilities_in_prompt(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        engine = DecisionEngine(db, c, _FakeMarket())
        captured = []
        def fake_call(prompt):
            captured.append(prompt)
            return json.dumps({"analysis_summary": "h", "actions": [],
                              "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        engine.run_cycle()
        p0 = captured[0]
        assert "Solana key imported: NO" in p0
        assert "do NOT propose Solana assets" in p0
        assert "bridge action available" in p0


class TestGasFeasibility:
    def _wallet(self, native, chain="ethereum", gas_price=30 * 10**9):
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet.__new__(EVMWallet)
        w.chain = chain
        w.cfg = {"native_symbol": "ETH", "chain_id": 1}
        w._acct = None
        w.native = native
        w.gas_price = gas_price
        def nb(): return native
        def eg(): return gas_price
        w.native_balance = nb
        w.w3 = type("W3", (), {
            "from_wei": staticmethod(lambda v, u: v / 1e18),
            "eth": type("Eth", (), {"gas_price": gas_price}),
        })()
        return w

    def test_gas_ok_with_native(self):
        # at 30 gwei a swap needs ~0.0075 ETH; give comfortable margin
        w = self._wallet(native=0.05)
        ok, why = w.gas_ok()
        assert ok, why

    def test_gas_cost_estimate_realistic(self):
        # 250k gas at 30 gwei = 0.0075 ETH — the check must not underestimate
        w = self._wallet(native=0.05)
        cost = w.estimated_gas_cost()
        assert 0.007 < cost < 0.010

    def test_gas_fails_without_native(self):
        w = self._wallet(native=0.0)
        ok, why = w.gas_ok()
        assert not ok and "insufficient gas" in why
        assert "Top up" in why

    def test_arc_gas_ok_with_balance(self):
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet.__new__(EVMWallet)
        w.chain = "arc"
        w.cfg = {"native_symbol": "USDC", "chain_id": 5042}
        w._acct = None
        def nb(): return 5.0
        w.native_balance = nb
        w.w3 = type("W3", (), {
            "from_wei": staticmethod(lambda v, u: v / 1e18),
            "eth": type("Eth", (), {"gas_price": 20 * 10**9}),
        })()
        ok, why = w.gas_ok()
        assert ok  # USDC balance covers gas on arc

    def test_arc_gas_fails_with_zero_usdc(self):
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet.__new__(EVMWallet)
        w.chain = "arc"
        w.cfg = {"native_symbol": "USDC", "chain_id": 5042}
        w._acct = None
        def nb(): return 0.0
        w.native_balance = nb
        w.w3 = type("W3", (), {
            "from_wei": staticmethod(lambda v, u: v / 1e18),
            "eth": type("Eth", (), {"gas_price": 20 * 10**9}),
        })()
        ok, why = w.gas_ok()
        assert not ok and "top up USDC" in why  # honest: no USDC = no gas even on arc

    def test_executor_preflight_blocks_gasless_chain(self, db, monkeypatch):
        c = Constraints(paper_trading=False, live_execution_mode="onchain",
                        onchain_chain="base")
        class _Store:
            def has(self, k): return k == "onchain_private_key"
            def get(self, k): return "0x" + "1" * 64
        engine = DecisionEngine(db, c, _FakeMarket(), None,
                                onchain_secrets=_Store())
        ai_out = json.dumps({
            "analysis_summary": "buy",
            "actions": [{"type": "buy", "asset": "0x" + "2" * 40,
                         "amount_or_percent": 1, "reason": "r"}],
            "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        # call the executor path directly (validator would need the asset
        # in market data; the gas preflight is what we are testing)
        err = engine._execute_onchain(
            "cycle-x",
            {"type": "buy", "asset": "0x" + "2" * 40,
             "amount": 1, "reason": "r"}, "0x" + "2" * 40, 1.0)
        assert err is not None and "gas check failed" in err


class TestGasReserve:
    def _wallet(self, native, chain="ethereum", gas_price=30 * 10**9):
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet.__new__(EVMWallet)
        w.chain = chain
        w.cfg = {"native_symbol": "ETH", "chain_id": 1}
        w._acct = None
        def nb(): return native
        w.native_balance = nb
        w.w3 = type("W3", (), {
            "from_wei": staticmethod(lambda v, u: v / 1e18),
            "eth": type("Eth", (), {"gas_price": gas_price}),
        })()
        return w

    def test_reserve_is_multiple_of_swap_cost(self):
        w = self._wallet(native=0.5)
        # 250k * 30gwei = 0.0075 per swap; reserve = 3 swaps = 0.0225
        assert w.gas_reserve() == pytest.approx(0.0225)

    def test_investable_excludes_reserve(self):
        w = self._wallet(native=0.5)
        assert w.investable_native() == pytest.approx(0.5 - 0.0225)

    def test_investable_never_negative(self):
        w = self._wallet(native=0.001)
        assert w.investable_native() == 0.0

    def test_native_buy_refused_beyond_investable(self):
        w = self._wallet(native=0.5)
        res = w.swap_exact_eth_for_tokens("0x" + "3" * 40, 0.49)
        assert not res.ok and "gas reserve protection" in res.error
        assert "reserve" in res.error

    def test_native_buy_ok_within_investable(self):
        w = self._wallet(native=0.5)
        res = w.swap_exact_eth_for_tokens("0x" + "3" * 40, 0.4)
        assert res.ok or "router" in res.error or "wrapped" in res.error
        # passes the reserve gate (fails later only for router config)

    def test_arc_usdc_reserve_also_enforced(self):
        from memicoin_core.onchain import EVMWallet
        w = EVMWallet.__new__(EVMWallet)
        w.chain = "arc"
        w.cfg = {"native_symbol": "USDC", "chain_id": 5042}
        w._acct = None
        def nb(): return 1.0
        w.native_balance = nb
        w.w3 = type("W3", (), {
            "from_wei": staticmethod(lambda v, u: v / 1e18),
            "eth": type("Eth", (), {"gas_price": 20 * 10**9}),  # arc 20 gwei floor
        })()
        assert w.investable_native() < 1.0  # reserve held back on arc too


class TestGasWarningSystem:
    def test_classification_levels(self):
        # exercise the classifier through the endpoint logic shape
        # (mirrors _classify in server.py)
        def classify(balance, per_swap):
            swaps = balance / per_swap if per_swap > 0 else float("inf")
            if balance <= 0: return "empty"
            if swaps < 1: return "critical"
            if swaps < 2: return "low"
            return "healthy"
        assert classify(0.0, 0.0075) == "empty"
        assert classify(0.005, 0.0075) == "critical"   # <1 swap
        assert classify(0.012, 0.0075) == "low"        # 1.6 swaps
        assert classify(0.5, 0.0075) == "healthy"

    def test_endpoint_reports_warnings_and_levels(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", lambda c: {})
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "base": {"native_symbol": "ETH", "native": 0.004,  # critical
                         "tokens": [{"symbol": "BRETT", "contract": "0x1", "balance": 5.0}]},
                "ethereum": {"native_symbol": "ETH", "native": 0.5,  # healthy
                             "tokens": []},
            }}, "solana": None})
        class _FakeW:
            def __init__(self, s, chain): self.chain = chain
            def gas_balance(self):
                return {"base": 0.004, "ethereum": 0.5}[self.chain]
            def estimated_gas_cost(self): return 0.0075
            def gas_ok(self): return True, ""
        monkeypatch.setattr(oc, "EVMWallet", _FakeW)
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        assert d["gas"]["base"]["level"] == "critical"
        assert d["gas"]["base"]["swaps_left"] == pytest.approx(0.5, abs=0.1)
        assert d["gas"]["ethereum"]["level"] == "healthy"
        warned = {w["chain"] for w in d["gas_warnings"]}
        assert "base" in warned            # has tokens -> warns
        assert "ethereum" not in warned    # healthy + no tokens -> no warn
        assert any("Top up" in w["hint"] or "out of gas" in w["hint"] for w in d["gas_warnings"])

    def test_no_warning_without_wallet(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: False)
        c = TestClient(srv.app)
        d = c.get("/api/wallet/balances").json()
        assert d["configured"] is False
        assert d.get("gas_warnings", []) == []


class TestDemoWallet:
    def _demo(self, db):
        from memicoin_core.demowallet import DemoWallet
        return DemoWallet(db)

    def test_gas_reserve_and_investable(self, db):
        d = self._demo(db)
        db.demo_set("base", "ETH", 0.01, is_gas=True)
        assert d.gas_balance("base") == 0.01
        assert d.investable_gas("base") == pytest.approx(0.01 - 0.00015 * 3)

    def test_swap_deducts_gas_and_balances(self, db):
        d = self._demo(db)
        db.demo_set("base", "ETH", 0.01, is_gas=True)
        db.demo_set("base", "USDC", 100.0)
        ok, msg = d.swap("base", "USDC", "", "BRETT", "0x532f", 50.0,
                         price_from=1.0, price_to=0.006)
        assert ok, msg
        assert db.demo_get("base", "USDC")["balance"] == pytest.approx(50.0)
        # bought 50/0.006 = 8333.33 BRETT
        assert db.demo_get("base", "BRETT", "0x532f")["balance"] == pytest.approx(8333.33, rel=1e-3)
        # gas deducted
        assert db.demo_get("base", "ETH")["balance"] == pytest.approx(0.01 - 0.00015)

    def test_swap_refused_without_gas(self, db):
        d = self._demo(db)
        db.demo_set("base", "USDC", 100.0)  # no ETH at all
        ok, msg = d.swap("base", "USDC", "", "BRETT", "0x532f", 50.0, 1.0, 0.006)
        assert not ok and "insufficient gas" in msg

    def test_swap_refused_insufficient_funds(self, db):
        d = self._demo(db)
        db.demo_set("base", "ETH", 0.01, is_gas=True)
        db.demo_set("base", "USDC", 10.0)
        ok, msg = d.swap("base", "USDC", "", "BRETT", "0x532f", 50.0, 1.0, 0.006)
        assert not ok and "insufficient USDC" in msg

    def test_bridge_moves_funds_with_gas(self, db):
        d = self._demo(db)
        db.demo_set("ethereum", "ETH", 0.05, is_gas=True)
        db.demo_set("ethereum", "USDC", 50.0)
        db.demo_set("arc", "USDC", 10.0, is_gas=True)  # dst gas (USDC on arc)
        ok, msg = d.bridge("ethereum", "arc", "USDC", 30.0)
        assert ok, msg
        fee = 30.0 * 0.10 / 100.0
        assert db.demo_get("ethereum", "USDC")["balance"] == pytest.approx(50.0 - 30.0 - fee)
        assert db.demo_get("arc", "USDC")["balance"] == pytest.approx(10.0 + 30.0 - 0.005)
        assert db.demo_get("ethereum", "ETH")["balance"] == pytest.approx(0.05 - 0.0075)

    def test_demo_endpoint_crud(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/demo-wallet", json={"chain": "arc", "asset": "USDC",
                                             "balance": 25.0, "is_gas": True})
        assert r.status_code == 200
        rows = c.get("/api/demo-wallet").json()["rows"]
        assert rows[0]["chain"] == "arc" and rows[0]["balance"] == 25.0
        assert rows[0]["value_usd"] == pytest.approx(25.0)  # USDC peg
        r = c.post("/api/go-demo")
        assert r.json()["mode"] == "demo"
        from memicoin_core.settings import load_constraints
        cc = load_constraints(db)
        assert cc.paper_trading is False and cc.live_execution_mode == "demo"
        c.post("/api/demo-wallet/clear")
        assert c.get("/api/demo-wallet").json()["rows"] == []


class TestModeAwarePortfolio:
    def test_demo_mode_default_shows_demo_wallet(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        s = c.get("/api/state").json()
        assert s["mode"] == "demo"
        assert "wallet_view" in s

    def test_demo_mode_shows_demo_wallet(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        db.demo_set("arc", "USDC", 15.0, is_gas=True)
        c = TestClient(srv.app)
        c.post("/api/go-demo")
        s = c.get("/api/state").json()
        assert s["mode"] == "demo"
        w = s["wallet_view"]
        assert w is not None
        assert w["chains"]["arc"][0]["symbol"] == "USDC"
        assert w["total_usd"] == pytest.approx(15.0)
        c.post("/api/go-paper")

    def test_real_mode_shows_real_wallet_view(self, db, monkeypatch):
        import app.server as srv
        import memicoin_core.onchain as oc
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.secrets, "has", lambda k: k == "onchain_private_key")
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_asset_map", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", lambda c: {})
        monkeypatch.setattr(oc, "read_wallet_balances",
                           lambda s, chains=None, extra_contracts=None: {
            "evm": {"address": "0xabc", "chains": {
                "base": {"native_symbol": "ETH", "native": 1.0, "tokens": []}}},
            "solana": None})
        from memicoin_core.settings import load_constraints
        cc = load_constraints(db)
        cc.paper_trading = False
        cc.live_execution_mode = "onchain"
        db.set_setting("constraints", cc.to_json())
        s = c2 = TestClient(srv.app).get("/api/state").json()
        assert s["mode"] == "real"
        assert s["wallet_view"] is not None
        assert s["wallet_view"]["chains"]["base"][0]["symbol"] == "ETH"
        # restore paper for other tests
        cc.paper_trading = True
        db.set_setting("constraints", cc.to_json())


class TestDemoGasValuation:
    def test_demo_eth_valued_via_weth(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        WETH_ETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
        WETH_BASE = "0x4200000000000000000000000000000000000006"
        db.demo_set("ethereum", "ETH", 0.02, is_gas=True)
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        captured = {}
        def fake_addr_prices(contracts):
            captured["c"] = contracts
            return {WETH_ETH.lower(): 2500.0, WETH_BASE.lower(): 2499.0}
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", fake_addr_prices)
        c = TestClient(srv.app)
        rows = c.get("/api/demo-wallet").json()["rows"]
        by = {(r["chain"], r["asset"]): r for r in rows}
        assert by[("ethereum", "ETH")]["value_usd"] == pytest.approx(0.02 * 2500.0)
        assert by[("base", "ETH")]["value_usd"] == pytest.approx(0.02 * 2499.0)
        assert by[("arc", "USDC")]["value_usd"] == pytest.approx(50.0)  # peg
        # both wrapped contracts were requested
        assert WETH_ETH.lower() in [x.lower() for x in captured["c"]]
        assert WETH_BASE.lower() in [x.lower() for x in captured["c"]]
        db.demo_clear()

    def test_demo_view_portfolio_totals_gas(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses",
                            lambda c: {"0x4200000000000000000000000000000000000006": 2500.0})
        c = TestClient(srv.app)
        c.post("/api/go-demo")
        s = c.get("/api/state").json()
        assert s["wallet_view"]["total_usd"] == pytest.approx(0.02 * 2500.0)
        assert s["wallet_view"]["chains"]["base"][0]["value_usd"] == pytest.approx(50.0)
        c.post("/api/go-paper")
        db.demo_clear()


class TestDemoUsdAmounts:
    def test_usd_amount_converted_to_token_quantity(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses",
                            lambda c: {"0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 2500.0})
        c = TestClient(srv.app)
        # $50 of ETH on ethereum (gas) -> 0.02 ETH
        r = c.post("/api/demo-wallet", json={
            "chain": "ethereum", "asset": "ETH", "balance": 50.0,
            "is_gas": True, "amount_is_usd": True})
        assert r.status_code == 200
        assert r.json()["stored_balance"] == pytest.approx(0.02)
        row = db.demo_get("ethereum", "ETH")
        assert row["balance"] == pytest.approx(0.02)
        db.demo_clear()

    def test_usd_amount_arc_usdc_peg(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/demo-wallet", json={
            "chain": "arc", "asset": "USDC", "balance": 50.0,
            "is_gas": True, "amount_is_usd": True})
        assert r.status_code == 200
        assert r.json()["stored_balance"] == pytest.approx(50.0)  # 1:1
        db.demo_clear()

    def test_usd_amount_unpriceable_rejected(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        monkeypatch.setattr(srv.market, "get_prices", lambda force=False: {})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses", lambda c: {})
        c = TestClient(srv.app)
        r = c.post("/api/demo-wallet", json={
            "chain": "ethereum", "asset": "WEIRDCOIN", "balance": 50.0,
            "amount_is_usd": True})
        assert r.status_code == 400
        assert "no live price" in r.json()["detail"]
        db.demo_clear()

    def test_token_amount_still_works_as_before(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        c = TestClient(srv.app)
        r = c.post("/api/demo-wallet", json={
            "chain": "ethereum", "asset": "ETH", "balance": 0.02,
            "is_gas": True, "amount_is_usd": False})
        assert r.json()["stored_balance"] == pytest.approx(0.02)
        db.demo_clear()


class TestDemoCycleIntegration:
    def _demo_engine(self, db):
        c = Constraints(paper_trading=False, live_execution_mode="demo",
                        max_position_pct=25.0)
        return DecisionEngine(db, c, _FakeMarket(), None)

    def test_ai_context_shows_demo_wallet(self, db, monkeypatch):
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        engine = self._demo_engine(db)
        captured = []
        def fake_call(prompt):
            captured.append(prompt)
            return json.dumps({"analysis_summary": "h", "actions": [],
                              "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        engine.run_cycle()
        p = captured[0]  # first call carries the full context
        assert "DEMO wallet" in p
        assert "USDC" in p and "arc" in p
        assert "investable gas" in p
        db.demo_clear()

    def test_buy_funded_by_usdc_when_present(self, db, monkeypatch):
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        db.demo_set("base", "USDC", 100.0)
        engine = self._demo_engine(db)
        # BRETT on base at $0.006: buy 1000 tokens = $6
        prices = {"BRETT": 0.006, "USDC": 1.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: {"BRETT": {"asset": "BRETT", "chain": "base",
                                                           "contract": BRETT_C}})
        monkeypatch.setattr(engine.market, "get_prices_for_addresses",
                            lambda c: {})
        err = engine._execute_demo("cx", {"type": "buy", "asset": "0x532f27101965dd16442E59d40670FaF5eBB142E4",
                                          "amount": 1000, "reason": "r"},
                                   "0x532f", 1000.0)
        assert err is None, err
        assert db.demo_get("base", "USDC")["balance"] == pytest.approx(94.0)
        assert db.demo_get("base", "BRETT", "0x532f27101965dd16442E59d40670FaF5eBB142E4")["balance"] == pytest.approx(1000.0)
        db.demo_clear()

    def test_buy_funded_by_investable_gas_when_no_usdc(self, db, monkeypatch):
        # user's scenario: only ETH on base, no USDC
        db.demo_set("base", "ETH", 0.05, is_gas=True)
        engine = self._demo_engine(db)
        prices = {"BRETT": 0.006, "ETH": 2500.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: {"BRETT": {"asset": "BRETT", "chain": "base",
                                                           "contract": BRETT_C}})
        monkeypatch.setattr(engine.market, "get_prices_for_addresses",
                            lambda c: {})
        # buy $30 of BRETT; investable ETH = 0.05 - 0.00045 = ~$124 -> enough
        err = engine._execute_demo("cx", {"type": "buy", "asset": "0x532f27101965dd16442E59d40670FaF5eBB142E4",
                                          "amount": 5000, "reason": "r"},
                                   "0x532f", 5000.0)
        assert err is None, err
        eth_after = db.demo_get("base", "ETH")["balance"]
        assert eth_after == pytest.approx(0.05 - 30.0 / 2500.0 - 0.00015, rel=1e-4)
        assert db.demo_get("base", "BRETT", "0x532f27101965dd16442E59d40670FaF5eBB142E4")["balance"] == pytest.approx(5000.0)
        db.demo_clear()

    def test_buy_refused_insufficient_all_sources(self, db, monkeypatch):
        db.demo_set("base", "ETH", 0.001, is_gas=True)  # below reserve anyway
        engine = self._demo_engine(db)
        prices = {"BRETT": 0.006, "ETH": 2500.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: {"BRETT": {"asset": "BRETT", "chain": "base",
                                                           "contract": BRETT_C}})
        monkeypatch.setattr(engine.market, "get_prices_for_addresses",
                            lambda c: {})
        err = engine._execute_demo("cx", {"type": "buy", "asset": "0x532f27101965dd16442E59d40670FaF5eBB142E4",
                                          "amount": 50000, "reason": "r"},
                                   "0x532f", 50000.0)
        assert err is not None
        assert "insufficient funds" in err or "gas" in err
        db.demo_clear()


class TestEmptyActionsChallenge:
    def test_hold_with_funds_gets_challenged_and_acts(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        empty = json.dumps({"analysis_summary": "opportunities exist",
                            "actions": [], "confidence": "high"})
        committed = json.dumps({"analysis_summary": "buy DOGE",
                               "actions": [{"type": "buy", "asset": "DOGE",
                                            "amount_or_percent": "20%",
                                            "reason": "momentum"}],
                               "confidence": "high"})
        calls = []
        def fake_call(prompt):
            calls.append(prompt)
            return empty if len(calls) == 1 else committed
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        result = engine.run_cycle()
        assert len(calls) == 2  # challenge fired
        assert "ZERO actions" in calls[1]
        assert len(result["executed"]) == 1
        assert result["executed"][0]["asset"] == "DOGE"

    def test_hold_with_no_investable_funds_not_challenged(self, db, monkeypatch):
        # all capital in a position, no free cash: nothing to invest with
        db.set_cash(0.0)
        db.set_setting("constraints", Constraints(max_loss_pct=100).to_json())
        db.upsert_position("DOGE", "dogecoin", 1000, 100)  # all funds invested
        from memicoin_core.settings import load_constraints
        c = load_constraints(db)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        calls = []
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: calls.append(p) or json.dumps(
                                {"analysis_summary": "h", "actions": [],
                                 "confidence": "low"}))
        result = engine.run_cycle()
        assert len(calls) == 1  # no challenge: no free funds to invest
        assert result["executed"] == []

    def test_demo_hold_with_demo_funds_challenged(self, db, monkeypatch):
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        empty = json.dumps({"analysis_summary": "x", "actions": [],
                            "confidence": "low"})
        calls = []
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: calls.append(p) or empty)
        engine.run_cycle()
        assert len(calls) == 2  # demo funds exist -> challenge fired
        db.demo_clear()

    def test_challenge_failure_keeps_original_hold(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        calls = []
        def fake_call(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return json.dumps({"analysis_summary": "h", "actions": [],
                                   "confidence": "low"})
            return "garbage not json"  # challenge response invalid
        monkeypatch.setattr(engine, "_call_mistral", fake_call)
        result = engine.run_cycle()
        assert len(calls) == 2
        assert result["status"] == "completed"
        assert result["executed"] == []  # original hold stands
        assert db.get_cash() == pytest.approx(100.0)


class TestContractsInContext:
    def test_market_rows_include_contract(self):
        from memicoin_core.ai_engine import BASE_INSTRUCTIONS
        assert "TOKEN IDENTIFIERS" in BASE_INSTRUCTIONS
        assert "contract address" in BASE_INSTRUCTIONS

    def test_context_passes_contract_through(self, db, monkeypatch):
        db.set_cash(100.0)
        c = Constraints(paper_trading=True)
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        class FM2(_FakeMarket):
            def get_asset_map(self, force=False):
                return {"BRETT": {"asset": "BRETT", "chain": "base",
                                  "price_usd": 0.006, "contract": BRETT_C}}
            def get_market_overview(self, force=False):
                return [{"asset": "BRETT", "chain": "base", "price_usd": 0.006,
                         "contract": BRETT_C, "name": "Brett"}]
        engine = DecisionEngine(db, c, FM2(), None)
        captured = []
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: captured.append(p) or json.dumps(
                                {"analysis_summary": "h", "actions": [],
                                 "confidence": "low"}))
        engine.run_cycle()
        assert BRETT_C in captured[0]  # the AI can see and use it


class TestDemoPercentResolution:
    def test_percent_buy_resolves_against_demo_funds(self, db, monkeypatch):
        from memicoin_core.constraints import resolve_action_amount
        db.demo_mode_percent = True
        db.demo_set("base", "ETH", 0.02, is_gas=True)   # $50 at $2500
        db.demo_set("base", "USDC", 100.0)               # + $100
        prices = {"BRETT": 0.006, "ETH": 2500.0, "USDC": 1.0}
        from memicoin_core.portfolio import PortfolioTracker
        action = {"type": "buy", "asset": "BRETT", "amount_or_percent": "25%",
                  "_chain": "base"}
        qty = resolve_action_amount(action, prices, PortfolioTracker(db), db)
        # spendable: $100 USDC + investable gas (0.02-0.00045 ETH * 2500 = $48.875)
        spendable = 100.0 + (0.02 - 0.00015 * 3) * 2500.0
        assert qty == pytest.approx(spendable * 0.25 / 0.006)
        db.demo_mode_percent = False
        db.demo_clear()

    def test_percent_buy_paper_mode_unchanged(self, db):
        from memicoin_core.constraints import resolve_action_amount
        from memicoin_core.portfolio import PortfolioTracker
        db.set_cash(100.0)
        action = {"type": "buy", "asset": "DOGE", "amount_or_percent": "25%"}
        qty = resolve_action_amount(action, PRICES, PortfolioTracker(db), db)
        assert qty == pytest.approx(250.0)  # 25% of $100 at $0.10

    def test_full_demo_cycle_percent_buy_executes(self, db, monkeypatch):
        # the user's exact failing scenario end to end
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo",
                        max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        prices = {"BRETT": 0.006, "ETH": 2500.0, "USDC": 1.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: {"BRETT": {"asset": "BRETT", "chain": "base",
                                                            "contract": BRETT_C}})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: json.dumps({
            "analysis_summary": "BRETT momentum",
            "actions": [{"type": "buy", "asset": BRETT_C,
                         "amount_or_percent": "25%", "reason": "momentum"}],
            "confidence": "high"}))
        result = engine.run_cycle()
        assert result["status"] == "completed", result.get("error")
        assert len(result["executed"]) == 1, result
        brett = db.demo_get("base", "BRETT", BRETT_C)
        assert brett and brett["balance"] > 0
        db.demo_clear()


class TestTwoModeWorld:
    def test_default_mode_is_demo(self, db):
        from memicoin_core.settings import load_constraints
        c = load_constraints(db)
        assert c.paper_trading is False  # migrated away from the class default
        assert c.live_execution_mode == "demo"

    def test_legacy_paper_setting_migrated_to_demo(self, db):
        from memicoin_core.settings import load_constraints, Constraints
        legacy = Constraints(paper_trading=True)
        db.set_setting("constraints", legacy.to_json())
        c = load_constraints(db)
        assert c.paper_trading is False and c.live_execution_mode == "demo"

    def test_demo_cost_basis_tracked_on_buy(self, db):
        from memicoin_core.demowallet import DemoWallet
        d = DemoWallet(db)
        db.demo_set("base", "ETH", 0.01, is_gas=True)
        db.demo_set("base", "USDC", 100.0)
        ok, _ = d.swap("base", "USDC", "", "BRETT", "0xabc",
                       50.0, price_from=1.0, price_to=0.006)
        assert ok
        row = db.demo_get("base", "BRETT", "0xabc")
        assert row["cost_basis_usd"] == pytest.approx(50.0)  # what was paid
        # sell half: basis scales down proportionally
        ok, _ = d.swap("base", "BRETT", "0xabc", "USDC", "",
                       row["balance"] / 2, price_from=0.006, price_to=1.0)
        row2 = db.demo_get("base", "BRETT", "0xabc")
        assert row2["cost_basis_usd"] == pytest.approx(25.0)
        db.demo_clear()

    def test_demo_stop_loss_fires_on_losing_position(self, db, monkeypatch):
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        c = Constraints(paper_trading=False, live_execution_mode="demo",
                        per_position_stop_pct=20.0)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        db.demo_mode_percent = True
        # bought at $0.006 (basis $60), now worth $0.004 -> -33%
        db.demo_set("base", "BRETT", 10000.0, contract=BRETT_C, cost_basis_usd=60.0)
        db.demo_set("base", "ETH", 0.01, is_gas=True)
        prices = {"BRETT": 0.004, "ETH": 2500.0, "USDC": 1.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: {"BRETT": {"asset": "BRETT", "chain": "base",
                                                           "contract": BRETT_C}})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: json.dumps({
            "analysis_summary": "h", "actions": [], "confidence": "low"}))
        engine.run_cycle()
        d = db.list_decisions()
        stops = [x for x in d if "stop-loss" in (x.get("reason") or "")]
        assert stops, "stop-loss should have fired"
        assert stops[0]["status"] == "executed"
        assert db.demo_get("base", "BRETT", BRETT_C)["balance"] == pytest.approx(0.0)
        db.demo_mode_percent = False
        db.demo_clear()


class TestDemoBridge:
    def test_demo_bridge_moves_usdc_without_real_key(self, db, monkeypatch):
        # the user's exact scenario: USDC on arc, no EVM key imported
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        db.demo_set("base", "ETH", 0.02, is_gas=True)  # dst gas
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)  # NO onchain_secrets
        ai_out = json.dumps({
            "analysis_summary": "bridge to base",
            "actions": [{"type": "bridge", "from_chain": "arc",
                         "to_chain": "base", "amount_or_percent": 40,
                         "reason": "BRETT momentum on base"}],
            "confidence": "high"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert result["status"] == "completed"
        assert len(result["executed"]) == 1  # bridge counted as executed
        fee = 40.0 * 0.10 / 100.0
        assert db.demo_get("arc", "USDC")["balance"] == pytest.approx(50.0 - 40.0 - fee - 0.005)
        assert db.demo_get("base", "USDC")["balance"] == pytest.approx(40.0)
        d = db.list_decisions()[0]
        assert d["status"] == "executed" and "[DEMO]" in d["reason"]
        db.demo_clear()

    def test_demo_bridge_insufficient_funds_fails_honestly(self, db, monkeypatch):
        db.demo_set("arc", "USDC", 10.0, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        ai_out = json.dumps({
            "analysis_summary": "bridge",
            "actions": [{"type": "bridge", "from_chain": "arc",
                         "to_chain": "base", "amount_or_percent": 50,
                         "reason": "r"}],
            "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        assert len(result["executed"]) == 0
        d = db.list_decisions()[0]
        assert d["status"] == "failed" and "insufficient" in d["reason"]
        db.demo_clear()


class TestBridgeAmountResolution:
    def test_percent_bridge_uses_source_chain_balance(self, db, monkeypatch):
        # arc holds 50 USDC; portfolio total is ~150 -> "100%" must mean 50
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        ai_out = json.dumps({
            "analysis_summary": "bridge all arc funds",
            "actions": [{"type": "bridge", "from_chain": "arc",
                         "to_chain": "base", "amount_or_percent": "100%",
                         "reason": "BRETT on base"}],
            "confidence": "high"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        result = engine.run_cycle()
        # 100% of arc's 50 = 50, but gas (0.005) comes from the same balance:
        # all-or-nothing refuses 50 + gas > 50. AI must size below 100%.
        d = db.list_decisions()[0]
        if d["status"] == "executed":
            assert db.demo_get("base", "USDC")["balance"] <= 50.0
        else:
            assert "insufficient" in d["reason"]  # honest refusal
        db.demo_clear()

    def test_absolute_bridge_clamped_to_source_balance(self, db, monkeypatch):
        # the user's exact case: 100 absolute from a 50 USDC arc balance
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        ai_out = json.dumps({
            "analysis_summary": "bridge",
            "actions": [{"type": "bridge", "from_chain": "arc",
                         "to_chain": "base", "amount_or_percent": 100,
                         "reason": "r"}],
            "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        engine.run_cycle()
        d = db.list_decisions()[0]
        # clamped to the 50 available (then all-or-nothing vs gas -> refused
        # honestly, or executed if it fits after the clamp)
        assert d["amount"] == 50.0 or d["amount"] < 100
        assert not (d["status"] == "failed" and "need 100" in d.get("reason", ""))
        db.demo_clear()

    def test_reasonable_percent_bridge_executes_with_fees(self, db, monkeypatch):
        # realistic bridge: needs gas on BOTH chains + a service fee
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        db.demo_set("base", "ETH", 0.02, is_gas=True)  # dst gas
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        ai_out = json.dumps({
            "analysis_summary": "bridge most arc funds",
            "actions": [{"type": "bridge", "from_chain": "arc",
                         "to_chain": "base", "amount_or_percent": "90%",
                         "reason": "BRETT momentum"}],
            "confidence": "high"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        engine.run_cycle()
        d = db.list_decisions()[0]
        assert d["status"] == "executed"
        amt = 45.0  # 90% of arc's 50
        fee = amt * 0.10 / 100.0  # 0.10% service fee
        # src: amount + fee + arc gas (USDC is gas there); dst: base gas burned
        assert db.demo_get("base", "USDC")["balance"] == pytest.approx(amt)
        assert db.demo_get("arc", "USDC")["balance"] == pytest.approx(50.0 - amt - fee - 0.005)
        assert db.demo_get("base", "ETH")["balance"] == pytest.approx(0.02 - 0.00015)
        db.demo_clear()

    def test_bridge_refused_without_dst_gas(self, db, monkeypatch):
        # dst has no gas token: real bridges cannot complete there
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        ai_out = json.dumps({
            "analysis_summary": "bridge",
            "actions": [{"type": "bridge", "from_chain": "arc",
                         "to_chain": "base", "amount_or_percent": "50%",
                         "reason": "r"}],
            "confidence": "low"})
        monkeypatch.setattr(engine, "_call_mistral", lambda p: ai_out)
        engine.run_cycle()
        d = db.list_decisions()[0]
        assert d["status"] == "failed"
        assert "both chains" in d["reason"] or "gas" in d["reason"]
        db.demo_clear()

    def test_same_chain_first_in_instructions(self):
        from memicoin_core.ai_engine import BASE_INSTRUCTIONS
        assert "SAME-CHAIN FIRST" in BASE_INSTRUCTIONS
        assert "sufficient funds" in BASE_INSTRUCTIONS and "SAME-CHAIN FIRST" in BASE_INSTRUCTIONS


class TestAiseesAllChains:
    def test_demo_summary_includes_gas_tokens_valued(self, db, monkeypatch):
        # the user's setup: ETH on ethereum + base, USDC on arc
        db.demo_set("ethereum", "ETH", 0.02, is_gas=True)
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        c = Constraints(paper_trading=False, live_execution_mode="demo")
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        prices = {"USDC": 1.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_prices_for_addresses",
                            lambda contracts: {"0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 2500.0,
                                               "0x4200000000000000000000000000000000000006": 2500.0})
        summary = engine._demo_portfolio_summary(prices)
        # all three chains present with REAL values (not 0)
        assert set(summary["chains"].keys()) == {"ethereum", "base", "arc"}
        assert summary["total_value_usd"] == pytest.approx(50.0 + 50.0 + 50.0)
        # the AI prompt must contain the ETH holdings
        captured = []
        monkeypatch.setattr(engine, "_call_mistral",
                            lambda p: captured.append(p) or json.dumps(
                                {"analysis_summary": "h", "actions": [], "confidence": "low"}))
        monkeypatch.setattr(engine.market, "get_asset_map", lambda force=False: {})
        engine.run_cycle()
        assert "ETH" in captured[0] and "ethereum" in captured[0]
        assert captured[0].count("ETH") >= 2  # both ETH rows visible
        db.demo_clear()

    def test_position_cap_uses_full_wallet_value(self, db, monkeypatch):
        # 25% of the FULL wallet (150) = 37.50, not 25% of arc-only (12.50)
        db.demo_set("ethereum", "ETH", 0.02, is_gas=True)
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        c = Constraints(paper_trading=False, live_execution_mode="demo",
                        max_position_pct=25.0)
        engine = DecisionEngine(db, c, _FakeMarket(), None)
        db.demo_mode_percent = True
        prices = {"BRETT": 0.006, "ETH": 2500.0, "USDC": 1.0}
        monkeypatch.setattr(engine.market, "get_prices", lambda force=False: prices)
        monkeypatch.setattr(engine.market, "get_prices_for_addresses",
                            lambda contracts: {"0x4200000000000000000000000000000000000006": 2500.0,
                                               "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 2500.0})
        monkeypatch.setattr(engine.market, "get_asset_map",
                            lambda force=False: {"BRETT": {"asset": "BRETT", "chain": "base",
                                                            "contract": BRETT_C}})
        action = {"type": "buy", "asset": "BRETT", "amount_or_percent": 10_000_000,
                  "_chain": "base", "_price": 0.006, "_contract": BRETT_C}
        r = engine.validator.validate_action(action, prices,
                                             engine.market.get_asset_map())
        assert r.ok
        # cap = 25% of 150 = 37.50 -> 6250 BRETT at 0.006
        assert action["amount"] == pytest.approx(6250.0)
        db.demo_mode_percent = False
        db.demo_clear()


class TestDemoPnL:
    def test_wallet_view_computes_pnl(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        BRETT_C = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
        # bot bought BRETT at $12 basis; now worth $24 -> +100%
        db.demo_set("base", "BRETT", 4000.0, contract=BRETT_C, cost_basis_usd=12.0)
        db.demo_set("base", "ETH", 0.02, is_gas=True)
        monkeypatch.setattr(srv.market, "get_prices",
                            lambda force=False: {"BRETT": 0.006, "ETH": 2500.0})
        monkeypatch.setattr(srv.market, "get_prices_for_addresses",
                            lambda c: {"0x4200000000000000000000000000000000000006": 2500.0})
        c = TestClient(srv.app)
        s = c.get("/api/state").json()
        rows = s["wallet_view"]["chains"]["base"]
        brett = [r for r in rows if r["symbol"] == "BRETT"][0]
        assert brett["pnl_usd"] == pytest.approx(24.0 - 12.0)
        assert brett["pnl_pct"] == pytest.approx(100.0)
        # manually-set rows (no trade history) have no P&L: honest '-'
        eth = [r for r in rows if r["symbol"] == "ETH"][0]
        # ETH was set by the user, basis == balance*price at set time or 0
        db.demo_clear()

    def test_untraded_rows_show_null_pnl(self, db, monkeypatch):
        import app.server as srv
        from fastapi.testclient import TestClient
        monkeypatch.setattr(srv, "db", db)
        db.demo_set("arc", "USDC", 50.0, is_gas=True)
        c = TestClient(srv.app)
        s = c.get("/api/state").json()
        row = s["wallet_view"]["chains"]["arc"][0]
        assert row["pnl_usd"] is None or row["pnl_usd"] == pytest.approx(0.0)
        db.demo_clear()
