"""
schemas.py — Typed Request and Response Models for TRUSTGRAPH Risk Decision API
================================================================================
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
PolicyDecision = Literal["ALLOW", "VERIFY", "THROTTLE", "BLOCK"]


class TransactionRiskRequest(BaseModel):
    """
    Incoming transaction risk evaluation request payload.

    Supports both explicit top-level entity fields (card1, addr1, P_emaildomain, DeviceInfo)
    and optional arbitrary tabular features (C1-C14, D1-D15, V1-V339, etc.).
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    transaction_id: Union[int, str] = Field(
        ...,
        description="Unique identifier for the transaction (payment gateway reference or merchant order ID)."
    )
    transaction_dt: Optional[float] = Field(
        None,
        description="Transaction timestamp in seconds (TransactionDT). If None, defaults to 0.0 or current time."
    )
    amount: float = Field(
        ...,
        gt=0.0,
        description="Transaction monetary amount (must be positive)."
    )
    card1: Optional[Union[int, str]] = Field(
        None,
        description="Primary card identifier (BIN/IIN representation)."
    )
    card2: Optional[Union[int, str]] = Field(
        None,
        description="Secondary card attribute (e.g. card sub-type / issuer routing)."
    )
    addr1: Optional[Union[int, str]] = Field(
        None,
        description="Billing / shipping region address code."
    )
    email_domain: Optional[str] = Field(
        None,
        alias="P_emaildomain",
        description="Purchaser email domain (e.g. gmail.com, yahoo.com)."
    )
    device_info: Optional[str] = Field(
        None,
        alias="DeviceInfo",
        description="Client device fingerprint or user agent string."
    )
    features: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional dictionary of additional raw IEEE-CIS tabular features (e.g. C1-C14, D1-D15, V1-V339)."
    )

    @field_validator("transaction_id")
    @classmethod
    def validate_transaction_id(cls, v: Union[int, str]) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("transaction_id cannot be empty")
        return s

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError("amount must be strictly positive")
        return float(v)

    def to_raw_feature_dict(self) -> Dict[str, Any]:
        """Flatten into the dictionary format expected by RuntimeScorer."""
        raw: Dict[str, Any] = {}
        if self.features:
            raw.update(self.features)

        # Convert string transaction IDs to stable integers for underlying graph engine
        try:
            raw["TransactionID"] = int(str(self.transaction_id))
        except ValueError:
            raw["TransactionID"] = abs(hash(str(self.transaction_id))) % 1_000_000_000

        raw["TransactionAmt"] = self.amount
        if self.transaction_dt is not None:
            raw["TransactionDT"] = self.transaction_dt

        if self.card1 is not None:
            raw["card1"] = self.card1
        if self.card2 is not None:
            raw["card2"] = self.card2
        if self.addr1 is not None:
            raw["addr1"] = self.addr1
        if self.email_domain is not None:
            raw["P_emaildomain"] = self.email_domain
        if self.device_info is not None:
            raw["DeviceInfo"] = self.device_info

        # Also capture any extra attributes passed in root
        extra = getattr(self, "__pydantic_extra__", None)
        if extra:
            for k, v in extra.items():
                if k not in raw:
                    raw[k] = v

        return raw


class SignalBreakdown(BaseModel):
    """Component risk signals derived causally from the pipeline."""
    baseline_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Instantaneous point-wise LightGBM risk score (A_t)."
    )
    temporal_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Entity-scoped temporal memory risk score (P_t)."
    )
    graph_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Bipartite relational device sharing risk score (G_t)."
    )
    fusion_risk: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Combined fused risk score under frozen M0 formulation (R_t)."
    )


class TransactionRiskResponse(BaseModel):
    """Authoritative transaction risk evaluation response."""
    transaction_id: str = Field(
        ...,
        description="Transaction identifier corresponding to the request."
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Primary continuous risk score R_t in [0.0, 1.0] from M0 fusion."
    )
    risk_level: RiskLevel = Field(
        ...,
        description="Categorical risk tier (LOW: <0.60, MEDIUM: 0.60-<0.65, HIGH: 0.65-<0.80, CRITICAL: >=0.80)."
    )
    decision: PolicyDecision = Field(
        ...,
        description="Action recommended by frozen progressive policy (ALLOW, VERIFY, THROTTLE, BLOCK)."
    )
    signals: SignalBreakdown = Field(
        ...,
        description="Transparent breakdown of all underlying model and contextual risk components."
    )
    explanation: List[str] = Field(
        ...,
        description="Auditable, human-readable reasons mapping to the actual features/signals that drove the decision."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Operational context, entity identifiers, and audit metadata."
    )


class HealthResponse(BaseModel):
    """Engine health and model readiness status."""
    status: str = Field(..., description="Overall service status (healthy / degraded).")
    engine: str = Field("TRUSTGRAPH Unified Risk Decision Engine", description="Engine identifier.")
    version: str = Field("1.0.0", description="API version.")
    model_readiness: Dict[str, bool] = Field(
        ...,
        description="Readiness flags for individual pipeline components."
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Frozen threshold parameters and fusion rule configuration."
    )
    stored_transactions: int = Field(
        ...,
        description="Total evaluated transactions retained in the state store."
    )


class ErrorResponse(BaseModel):
    """Standardized error envelope."""
    error: str = Field(..., description="Error category code.")
    message: str = Field(..., description="Descriptive error explanation.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of error occurrence.")
