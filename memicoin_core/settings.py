"""User goals & constraints (§5 of the brief) — versioned JSON, enforced in code."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskProfile = Literal["safe", "balanced", "aggressive"]
AutonomyMode = Literal["constrained", "full_control"]
# exchange = Mode A (CCXT), onchain = Mode B (imported hot wallet + DEX)
LiveExecutionMode = Literal["exchange", "onchain"]

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
    risk_profile: RiskProfile = "balanced"
    allowed_chains: list[str] = Field(
        default_factory=lambda: ["solana", "ethereum", "binance-smart-chain", "base", "polygon"]
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
    paper_trading: bool = True
    live_execution_mode: LiveExecutionMode = "exchange"
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
        db.set_setting("constraints", c.to_json())
        return c
    return Constraints(**{k: v for k, v in raw.items() if k in Constraints.model_fields})
