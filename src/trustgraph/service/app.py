"""
app.py — FastAPI Application for TRUSTGRAPH Transaction Risk Decision API
==========================================================================

Endpoints:
  POST /api/v1/risk/evaluate                   Evaluate transaction risk
  GET  /api/v1/risk/transactions/{txn_id}      Retrieve latest decision for transaction
  GET  /api/v1/health                          Engine health & model readiness
"""

import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from trustgraph.service.schemas import (
    ErrorResponse,
    HealthResponse,
    TransactionRiskRequest,
    TransactionRiskResponse,
)
from trustgraph.service.engine_service import RiskEngineService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trustgraph.api")

# Initialize FastAPI app
app = FastAPI(
    title="TRUSTGRAPH Transaction Risk Decision API",
    description="Production-grade real-time payment risk decision API powered by causal temporal and relational risk fusion.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend / dashboard integration (e.g. Razorpay Buildathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance (lazy or eagerly loaded)
_engine_service: Optional[RiskEngineService] = None


def get_engine_service() -> RiskEngineService:
    """Retrieve or initialize singleton RiskEngineService."""
    global _engine_service
    if _engine_service is None:
        logger.info("Initializing RiskEngineService singleton...")
        _engine_service = RiskEngineService.create()
    return _engine_service


def set_engine_service(service: RiskEngineService) -> None:
    """Explicitly set service instance (used in tests for isolation)."""
    global _engine_service
    _engine_service = service


# ---------------------------------------------------------------------------
# Middleware: Request Timing and Structured Logging
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Add processing latency header
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"

    if not request.url.path.endswith("/health"):
        logger.info(
            "%s %s -> status=%d latency=%.2fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
    return response


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("Validation error on %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "VALIDATION_ERROR",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred during risk evaluation.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# API Routes: /api/v1
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/risk/evaluate",
    response_model=TransactionRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate Transaction Risk",
    description="Evaluate combined risk score R_t and policy action for an incoming transaction.",
    responses={
        200: {"description": "Risk decision evaluated successfully."},
        422: {"model": ErrorResponse, "description": "Validation error in request payload."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)
async def evaluate_transaction(request: TransactionRiskRequest) -> TransactionRiskResponse:
    service = get_engine_service()
    try:
        response = service.evaluate_transaction(request)
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error("Error evaluating transaction %s: %s", request.transaction_id, str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}",
        )


@app.get(
    "/api/v1/risk/transactions/{transaction_id}",
    response_model=TransactionRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Transaction Decision",
    description="Retrieve the latest known risk evaluation and explanation for a given transaction ID.",
    responses={
        200: {"description": "Transaction found."},
        404: {"model": ErrorResponse, "description": "Transaction ID not found in state store."},
    },
)
async def get_transaction(transaction_id: str) -> TransactionRiskResponse:
    service = get_engine_service()
    result = service.get_transaction(str(transaction_id))
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction '{transaction_id}' not found in risk engine state store.",
        )
    return result


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Engine Health Check",
    description="Returns service status, component readiness flags, and active frozen parameter configuration.",
)
async def health_check() -> HealthResponse:
    service = get_engine_service()
    status_data = service.get_health_status()
    return HealthResponse(**status_data)


# ---------------------------------------------------------------------------
# Phase 4 Routes: GraphRAG AI Investigator & Interactive Graph Visualization
# ---------------------------------------------------------------------------

from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from trustgraph.investigator.service import get_investigator_service


class AskQuestionRequest(BaseModel):
    question: str


@app.get(
    "/api/v1/investigation/demo-transactions",
    summary="List Demonstration Cases",
    description="Retrieve representative demonstration transactions for GraphRAG investigation.",
)
async def list_demo_transactions():
    service = get_investigator_service()
    return service.get_demo_transactions()


@app.get(
    "/api/v1/overview/stats",
    summary="Portfolio Overview KPI Statistics",
    description="Retrieve dynamic portfolio-wide transaction volumes, high-risk counts, loss avoided, and action breakdown.",
)
async def get_overview_statistics():
    service = get_investigator_service()
    return service.get_overview_stats()


@app.get(
    "/api/v1/risk/{transaction_id}",
    summary="Get Transaction Risk Summary",
    description="Retrieve base risk A_t, graph risk G_t, final risk R_t, and cost decision.",
)
async def get_transaction_risk_summary(transaction_id: int):
    service = get_investigator_service()
    return service.get_transaction_risk_record(transaction_id)


@app.get(
    "/api/v1/risk/{transaction_id}/graph",
    summary="Get Force-Directed Graph Neighborhood",
    description="Retrieve 1-hop and 2-hop graph neighborhood view for force-directed D3 visualization.",
)
async def get_transaction_graph(transaction_id: int, max_hops: int = 2):
    service = get_investigator_service()
    view = service.get_graph_view(transaction_id, max_hops=max_hops)
    return view.to_dict()


@app.get(
    "/api/v1/risk/{transaction_id}/evidence",
    summary="Get Retrieved Graph Evidence",
    description="Retrieve ranked, provenance-backed evidence items for a transaction.",
)
async def get_transaction_evidence_items(transaction_id: int):
    service = get_investigator_service()
    items = service.get_evidence(transaction_id)
    return [item.to_dict() for item in items]


@app.get(
    "/api/v1/risk/{transaction_id}/investigate",
    summary="Investigate Transaction (GET)",
    description="Run GraphRAG AI Risk Investigation producing grounded findings and evidence citations.",
)
@app.post(
    "/api/v1/risk/{transaction_id}/investigate",
    summary="Investigate Transaction (POST)",
    description="Run GraphRAG AI Risk Investigation producing grounded findings and evidence citations.",
)
async def run_investigation(transaction_id: int, scenario: str = "balanced"):
    service = get_investigator_service()
    report, _ = service.investigate(transaction_id, scenario_name=scenario)
    return report.to_dict()


@app.post(
    "/api/v1/risk/{transaction_id}/ask",
    summary="Ask the AI Risk Investigator",
    description="Ask an investigative question grounded strictly in retrieved graph evidence.",
)
async def ask_investigator(transaction_id: int, req: AskQuestionRequest):
    service = get_investigator_service()
    ans = service.ask(transaction_id, question=req.question)
    return ans.to_dict()


# ---------------------------------------------------------------------------
# Phase 8: Multi-Analyst Risk Council Endpoint
# ---------------------------------------------------------------------------

from trustgraph.council import get_risk_council


@app.get(
    "/api/v1/risk/{transaction_id}/council",
    summary="Multi-Analyst Risk Council Evaluation",
    description="Evaluate transaction through independent Transaction Risk Analyst and Slow-Burn Analyst with AI Risk Officer synthesis.",
)
async def get_risk_council_case(transaction_id: int, scenario: str = "balanced"):
    council = get_risk_council()
    return council.evaluate(transaction_id, scenario_name=scenario)



FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"

if FRONTEND_DIR.exists():
    public_dir = FRONTEND_DIR / "public"
    if public_dir.exists():
        app.mount("/public", StaticFiles(directory=str(public_dir)), name="public")
    js_dir = FRONTEND_DIR / "js"
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="SENTRY Frontend Dashboard",
    include_in_schema=False,
)
@app.get(
    "/",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def serve_sentinel_frontend():
    """Serve the unified SENTRY frontend (frontend/index.html)."""
    frontend_index = FRONTEND_DIR / "index.html"
    if frontend_index.exists():
        with open(frontend_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SENTRY frontend not found.</h1>", status_code=404)


@app.get(
    "/{page_name}.html",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def serve_html_page(page_name: str):
    """Serve direct HTML pages like transactions.html, investigations.html, risk-engine.html."""
    target_file = FRONTEND_DIR / f"{page_name}.html"
    if target_file.exists():
        with open(target_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    raise HTTPException(status_code=404, detail="Page Not Found")


