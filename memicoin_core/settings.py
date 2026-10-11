"""User goals & constraints (§5 of the brief) — versioned JSON, enforced in code."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskProfile = Literal["safe", "balanced", "aggressive"]
AutonomyMode = Literal["constrained", "full_control"]
# exchange = Mode A (CCXT), onchain = Mode B (imported hot wallet + DEX)
LiveExecutionMode = Literal["exchange", "onchain", "demo"]

SETTINGS_VERSION = 1

MAX_POSITION_DEFAULTS = {"safe": 10.0, "balanced": 25.0, "aggressive": 50.0}


class Constraints(BaseModel):
    settings_version: int = SETTINGS_VERSION
    profit_target_pct: Optional[float] = Field(
        default=None, ge=0, description="Stop when total portfolio is up this % (e.g. 50)."
    )
    max_loss_pct: float = Field(
        default=20.0, ge=0, le=100,
        description="Hard cap on total drawdown in % of starting capital. Required.",
    )
    per_position_stop_pct: Optional[float] = Field(
        default=20.0, ge=0,
        description="Per-position stop-loss: when a single position is down this %, "
                    "it is sold to cash/USDC automatically at the next cycle. "
                    "Set 0 to disable.",
    )
    risk_profile: RiskProfile = "balanced"
    allowed_chains: list[str] = Field(
        default_factory=lambda: ["solana", "ethereum", "binance-smart-chain", "base", "polygon", "arc"]
    )
    max_position_pct: float = Field(
        default=25.0, gt=0, le=100,
        description="Max % of portfolio value in a single asset.",
    )
    autonomy_mode: AutonomyMode = "constrained"
    custom_prompt: str = Field(
        default="", description="Behavioral instructions the AI reads before every decision."
    )
    decision_time_utc: str = "00:00"
    # Class default is the internal test substrate (paper). The APP default
    # is demo mode - enforced by load_constraints() migration below, so a
    # fresh install and legacy paper settings both land on demo.
    paper_trading: bool = True
    live_execution_mode: LiveExecutionMode = "demo"
    # chain used for Mode B on-chain swaps (must be in onchain.SUPPORTED_CHAINS)
    onchain_chain: str = "ethereum"
    starting_capital_usd: float = 100.0
    trading_paused: bool = False
    emergency_stopped: bool = False

    def effective_max_position_pct(self) -> float:
        return self.max_position_pct

    def to_json(self) -> dict:
        return self.model_dump()


def load_constraints(db) -> Constraints:
    raw = db.get_setting("constraints")
    if raw is None:
        c = Constraints()
        # fresh install: land on the user-facing default (demo), not the
        # internal paper test substrate
        c.paper_trading = False
        c.live_execution_mode = "demo"
        db.set_setting("constraints", c.to_json())
        return c
    c = Constraints(**{k: v for k, v in raw.items() if k in Constraints.model_fields})
    # two-mode world: the app is demo or real. Paper/exchange are internal
    # legacy states -> migrate on load.
    if c.paper_trading or c.live_execution_mode == "exchange":
        c.paper_trading = False
        c.live_execution_mode = "demo"
        db.set_setting("constraints", c.to_json())
    return c
