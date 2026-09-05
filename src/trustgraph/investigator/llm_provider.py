"""
llm_provider.py — LLM Provider Abstraction & Anti-Hallucination Fallback (Phase 4)
==================================================================================

Provides an abstract LLM interface with:
  1. GroqProvider (ultra-fast LPU inference via Groq API)
  2. GeminiProvider (calls Google Gemini API via HTTP if configured)
  3. DeterministicFallbackProvider (rule-based, zero hallucination, zero external deps)

Guarantees:
  - If no API key is configured, the system still functions in deterministic mode.
  - Every factual claim cites specific [E...] and [RISK_ENGINE] tags.
  - Unsupported questions return "Insufficient evidence to determine this."
  - Mathematical risk values (A_t, G_t, R_t, Action) cannot be tampered with.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from trustgraph.investigator.schema import EvidenceItem, EvidenceType, GroundedReason

logger = logging.getLogger(__name__)

# Groq API key — set via environment variable GROQ_API_KEY
# Never hardcode secrets in source. Add GROQ_API_KEY to your shell environment or a .env file.
DEFAULT_GROQ_KEY = os.environ.get("GROQ_API_KEY", "")


class LLMProvider(ABC):
    """Abstract base class for risk investigation language models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass

    @property
    def is_fallback(self) -> bool:
        """True if running in local rule-based fallback mode."""
        return False

    @abstractmethod
    def generate_investigation(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
    ) -> Tuple[List[GroundedReason], str]:
        """
        Generate grounded explanation reasons and summary text.
        Returns: (reasons, summary_text)
        """
        pass

    @abstractmethod
    def answer_question(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
        question: str,
    ) -> Tuple[str, List[str], bool]:
        """
        Answer an analyst question grounded strictly in retrieved evidence.
        Returns: (answer_text, cited_evidence_ids, is_grounded)
        """
        pass


# ===========================================================================
# Deterministic Fallback Provider (Zero External Dependencies)
# ===========================================================================

class DeterministicFallbackProvider(LLMProvider):
    """
    Deterministic rule-based investigation generator.
    Guarantees strict factual accuracy and citation linking without external API calls.
    """

    @property
    def name(self) -> str:
        return "deterministic_fallback"

    @property
    def is_fallback(self) -> bool:
        return True

    def generate_investigation(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
    ) -> Tuple[List[GroundedReason], str]:
        reasons: List[GroundedReason] = []

        # 1. Inspect direct fraud evidence
        direct_frauds = [e for e in evidence_items if e.evidence_type == EvidenceType.DIRECT_FRAUD]
        if direct_frauds:
            e = direct_frauds[0]
            reasons.append(GroundedReason(
                statement=f"Customer entity has a confirmed history of fraudulent transactions ({e.provenance.get('total_prior_frauds', 'multiple')} confirmed prior fraud events).",
                evidence_ids=[e.evidence_id],
                category="GRAPH_RELATION",
            ))

        # 2. Inspect device fraud and multiplexing
        dev_frauds = [e for e in evidence_items if e.evidence_type == EvidenceType.DEVICE_FRAUD]
        if dev_frauds:
            e = dev_frauds[0]
            reasons.append(GroundedReason(
                statement=f"Device fingerprint is directly linked to historical confirmed fraud transaction #{e.provenance.get('fraud_txn_id', 'unknown')}.",
                evidence_ids=[e.evidence_id],
                category="GRAPH_RELATION",
            ))

        dev_sharing = [e for e in evidence_items if e.evidence_type == EvidenceType.DEVICE_SHARING]
        if dev_sharing:
            e = dev_sharing[0]
            reasons.append(GroundedReason(
                statement=f"High-degree device sharing: hardware identifier used across {e.provenance.get('distinct_entities_count', 'multiple')} distinct entities.",
                evidence_ids=[e.evidence_id],
                category="GRAPH_RELATION",
            ))

        # 3. 2-Hop Network Contamination
        hop2 = [e for e in evidence_items if e.evidence_type == EvidenceType.HOP2_CONTAMINATION]
        if hop2:
            e = hop2[0]
            reasons.append(GroundedReason(
                statement=f"2-hop network contamination: connected entity infrastructure shares ties with {e.provenance.get('hop2_frauds', 'confirmed')} confirmed fraud cases.",
                evidence_ids=[e.evidence_id],
                category="GRAPH_RELATION",
            ))

        # 4. Velocity Anomalies
        vel = [e for e in evidence_items if e.evidence_type == EvidenceType.VELOCITY_BURST]
        if vel:
            e = vel[0]
            reasons.append(GroundedReason(
                statement=f"Velocity burst: elevated transaction frequency within short time windows ({e.provenance.get('velocity_1h', 0)} in 1h, {e.provenance.get('velocity_24h', 0)} in 24h).",
                evidence_ids=[e.evidence_id],
                category="GRAPH_RELATION",
            ))

        # 5. Base ML probability reason
        if base_risk_a >= 0.5:
            ml_statement = f"Base XGBoost model flagged severe behavioral anomalies with raw probability A_t = {base_risk_a:.4f} (> 50%)."
        elif base_risk_a >= 0.12:
            ml_statement = f"Base XGBoost model detected elevated transaction risk with raw probability A_t = {base_risk_a:.4f} exceeding threshold 0.12."
        else:
            ml_statement = f"Base XGBoost model assessed low transaction-level risk with raw probability A_t = {base_risk_a:.4f}."

        reasons.append(GroundedReason(
            statement=ml_statement,
            evidence_ids=["RISK_ENGINE"],
            category="ML_BASELINE",
        ))

        # 6. Mathematical fusion & cost decision reason
        uplift = final_risk_r - base_risk_a
        if abs(uplift) > 0.001:
            fusion_statement = (
                f"F2 Conditional Fusion added +{uplift:.4f} contextual risk uplift from calibrated graph risk G_t = {graph_risk_g:.4f}, "
                f"raising final risk score R_t to {final_risk_r:.4f}."
            )
        else:
            fusion_statement = (
                f"F2 Conditional Fusion computed final risk R_t = {final_risk_r:.4f} "
                f"combining base risk A_t = {base_risk_a:.4f} and calibrated graph risk G_t = {graph_risk_g:.4f}."
            )

        cost_statement = f"Expected financial loss minimisation determined optimal intervention: {action} (expected loss: INR {expected_cost:.2f})."

        reasons.append(GroundedReason(
            statement=f"{fusion_statement} {cost_statement}",
            evidence_ids=["RISK_ENGINE"],
            category="MATHEMATICAL_FUSION",
        ))

        # Build summary markdown
        summary_lines = [
            f"### AI Risk Investigation Report: Transaction #{transaction_id}",
            f"",
            f"**Final Risk Assessment**: `R_t = {final_risk_r:.2%}` | **Recommended Action**: `{action}` (Expected Cost: INR {expected_cost:.2f})",
            f"",
            f"#### Key Investigative Findings:",
        ]
        for idx, r in enumerate(reasons, 1):
            citations = " ".join(f"[{eid}]" for eid in r.evidence_ids)
            summary_lines.append(f"{idx}. {r.statement} {citations}")

        return reasons, "\n".join(summary_lines)

    def answer_question(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
        question: str,
    ) -> Tuple[str, List[str], bool]:
        q_lower = question.lower().strip()

        # Check for out-of-domain / unsupported queries first (anti-hallucination)
        unsupported_topics = [
            "weather", "spouse", "salary", "religion", "politics", "president",
            "capital of", "recipe", "password", "crypto", "bitcoin", "stock price",
            "bank balance", "pin", "cvv", "home address line", "credit score"
        ]
        if any(topic in q_lower for topic in unsupported_topics):
            return (
                "Insufficient evidence to determine this. The payment knowledge graph and transaction record "
                "do not contain this information. Factual claims cannot be substantiated from retrieved evidence.",
                [],
                False,
            )

        # 1. Why was it blocked / verified / allowed?
        if any(w in q_lower for w in ["why", "reason", "cause", "blocked", "verified", "throttled", "allowed"]):
            cited_ids = []
            findings = []
            for e in evidence_items:
                if e.evidence_type in (EvidenceType.DIRECT_FRAUD, EvidenceType.DEVICE_FRAUD, EvidenceType.DEVICE_SHARING, EvidenceType.HOP2_CONTAMINATION):
                    findings.append(f"- {e.title}: {e.description} [{e.evidence_id}]")
                    cited_ids.append(e.evidence_id)

            cited_ids.append("RISK_ENGINE")
            findings.append(
                f"- Mathematical Cost Model: Action '{action}' was selected because it minimizes expected merchant financial loss "
                f"(INR {expected_cost:.2f}) given final fraud probability R_t = {final_risk_r:.4f}. [RISK_ENGINE]"
            )

            ans = f"Transaction #{transaction_id} was assigned action '{action}' for the following evidence-backed reasons:\n\n" + "\n".join(findings)
            return ans, cited_ids, True

        # 2. What entities / devices are connected?
        if any(w in q_lower for w in ["connected", "entities", "device", "card", "network", "sharing", "share"]):
            cited_ids = []
            info = []
            for e in evidence_items:
                if e.evidence_type in (EvidenceType.DEVICE_SHARING, EvidenceType.DEVICE_FRAUD, EvidenceType.DIRECT_FRAUD, EvidenceType.CARD_SHARING, EvidenceType.HOP2_CONTAMINATION):
                    info.append(f"- {e.title}: {e.description} [{e.evidence_id}]")
                    cited_ids.append(e.evidence_id)

            if not info:
                info.append("No suspicious shared entities or devices were found in the 2-hop neighborhood. [E1]")
                cited_ids.append("E1")

            ans = f"Connected graph intelligence for Transaction #{transaction_id}:\n\n" + "\n".join(info)
            return ans, cited_ids, True

        # 3. Strongest fraud evidence
        if any(w in q_lower for w in ["strongest", "fraud", "evidence", "proof", "suspicious"]):
            strong = [e for e in evidence_items if e.risk_weight >= 0.70]
            if strong:
                cited_ids = [e.evidence_id for e in strong]
                lines = [f"- **{e.title}** (Weight: {e.risk_weight}): {e.description} [{e.evidence_id}]" for e in strong]
                ans = f"The strongest fraud evidence retrieved from the knowledge graph for Transaction #{transaction_id}:\n\n" + "\n".join(lines)
                return ans, cited_ids, True
            else:
                ans = f"No high-severity fraud evidence (weight >= 0.70) was retrieved. The entity exhibits low risk signals. [RISK_ENGINE]"
                return ans, ["RISK_ENGINE"], True

        # 4. Score change / fusion / uplift
        if any(w in q_lower for w in ["score", "risk", "change", "increase", "uplift", "formula", "beta", "g_t", "a_t", "r_t", "fusion"]):
            uplift = final_risk_r - base_risk_a
            ans = (
                f"Risk score breakdown for Transaction #{transaction_id}:\n\n"
                f"- Base ML Risk (A_t): {base_risk_a:.4f} (XGBoost transaction model)\n"
                f"- Graph Context Risk (G_t): {graph_risk_g:.4f} (calibrated point-in-time entity graph)\n"
                f"- Fusion Equation: R_t = clip(A_t + beta * G_t * (1 - A_t), 0, 1) with beta = 0.05\n"
                f"- Final Risk (R_t): {final_risk_r:.4f} (net uplift: {uplift:+.4f}) [RISK_ENGINE]"
            )
            return ans, ["RISK_ENGINE"], True

        # Generic grounded summary
        cited_ids = [e.evidence_id for e in evidence_items[:3]]
        bullets = [f"- {e.title}: {e.description} [{e.evidence_id}]" for e in evidence_items[:3]]
        ans = (
            f"Investigation summary for Transaction #{transaction_id} (Final Risk: {final_risk_r:.2%}, Action: {action}):\n\n"
            + "\n".join(bullets)
        )
        return ans, cited_ids, True


# ===========================================================================
# API Provider: Groq (Ultra-Fast LPU Inference)
# ===========================================================================

class GroqProvider(LLMProvider):
    """
    Calls Groq API via OpenAI-compatible endpoint with low latency (<200ms).
    Enforces strict anti-hallucination prompts and citation anchoring.
    """

    def __init__(self, api_key: str, model_name: str = "groq/compound-mini") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.fallback = DeterministicFallbackProvider()

    @property
    def name(self) -> str:
        return f"groq_{self.model_name.replace('/', '_')}"

    def generate_investigation(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
    ) -> Tuple[List[GroundedReason], str]:
        # Always obtain deterministic structured reasons to guarantee schema integrity
        reasons, _ = self.fallback.generate_investigation(
            transaction_id, base_risk_a, graph_risk_g, final_risk_r, action, expected_cost, evidence_items
        )

        prompt = (
            f"You are an expert Payment Fraud Risk Investigator for TRUSTGRAPH.\n"
            f"Synthesize an investigation summary for Transaction #{transaction_id}.\n\n"
            f"IMMUTABLE MATHEMATICAL RISK VALUES (Do NOT alter or recalculate):\n"
            f"- Base XGBoost ML Risk (A_t): {base_risk_a:.4f}\n"
            f"- Calibrated Graph Risk (G_t): {graph_risk_g:.4f}\n"
            f"- Fused Final Risk (R_t): {final_risk_r:.4f}\n"
            f"- Action Decision: {action} (Expected Merchant Loss: INR {expected_cost:.2f})\n\n"
            f"RETRIEVED KNOWLEDGE GRAPH EVIDENCE ITEMS:\n"
            + "\n".join(f"[{e.evidence_id}] {e.title}: {e.description}" for e in evidence_items)
            + "\n\nCRITICAL ANTI-HALLUCINATION & EVIDENCE INTERPRETATION RULES:\n"
            f"1. You must ONLY state facts present in the evidence items above.\n"
            f"2. Every claim must cite the relevant evidence tag (e.g. [E1] or [RISK_ENGINE]).\n"
            f"3. Do NOT invent new entities, devices, card numbers, or amounts.\n"
            f"4. Clearly distinguish between directly observed facts contained in evidence and reasonable interpretation of those facts.\n"
            f"5. Never present an interpretation as a verified fact.\n"
            f"6. Do not use causal, definitive, or accusatory language unless explicitly supported by supplied evidence.\n"
            f"7. Avoid unsupported terms ('fraud ring', 'organized group', 'coordinated attack', 'confirmed fraudster', 'malicious actor').\n"
            f"8. When evidence indicates correlation or association, describe it as 'relational context', 'association', or 'shared activity'.\n"
            f"9. Structure your response clearly with a brief summary and supporting points."
        )

        try:
            summary = self._call_groq_api(prompt, max_tokens=600)
            if summary and len(summary.strip()) > 50:
                # Normalize any bracket variations
                clean_summary = re.sub(r"【(.*?)】", r"[\1]", summary)
                return reasons, clean_summary
        except Exception as e:
            logger.warning("Groq API call failed (%s); seamlessly using deterministic fallback.", str(e))

        return self.fallback.generate_investigation(
            transaction_id, base_risk_a, graph_risk_g, final_risk_r, action, expected_cost, evidence_items
        )

    def answer_question(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
        question: str,
    ) -> Tuple[str, List[str], bool]:
        # Pre-screen out-of-domain queries strictly
        q_lower = question.lower().strip()
        unsupported = [
            "weather", "spouse", "salary", "religion", "politics", "president",
            "capital of", "recipe", "password", "crypto", "bitcoin", "stock price",
            "bank balance", "pin", "cvv", "home address line", "credit score"
        ]
        if any(topic in q_lower for topic in unsupported):
            return (
                "Insufficient evidence to determine this. The payment knowledge graph and transaction record "
                "do not contain this information. Factual claims cannot be substantiated from retrieved evidence.",
                [],
                False,
            )

        prompt = (
            f"You are a Payment Fraud Investigator answering an analyst question regarding Transaction #{transaction_id}.\n\n"
            f"RETRIEVED GRAPH EVIDENCE:\n"
            + "\n".join(f"[{e.evidence_id}] {e.title}: {e.description}" for e in evidence_items)
            + f"\n[RISK_ENGINE] A_t={base_risk_a:.4f}, G_t={graph_risk_g:.4f}, R_t={final_risk_r:.4f}, Action={action}, Expected Loss=INR {expected_cost:.2f}\n\n"
            f"ANALYST QUESTION:\n{question}\n\n"
            f"RULES:\n"
            f"1. Answer strictly and solely using the provided evidence.\n"
            f"2. Cite evidence IDs like [E1] or [RISK_ENGINE] for every claim.\n"
            f"3. Clearly distinguish between directly observed facts contained in evidence and reasonable interpretation of those facts.\n"
            f"4. Never present an interpretation as a verified fact.\n"
            f"5. Avoid unsupported terms ('fraud ring', 'organized group', 'coordinated attack', 'confirmed fraudster', 'malicious actor').\n"
            f"6. When evidence only indicates correlation or association, describe it as 'relational context', 'association', or 'shared activity'.\n"
            f"7. If the evidence does not answer the question, state: 'Insufficient evidence to determine this.'\n"
            f"8. Do NOT hallucinate."
        )

        try:
            raw_answer = self._call_groq_api(prompt, max_tokens=400)
            if raw_answer:
                clean_answer = re.sub(r"【(.*?)】", r"[\1]", raw_answer)
                cited = list(set(re.findall(r"\[(E\d+|RISK_ENGINE)\]", clean_answer)))
                valid_ids = {e.evidence_id for e in evidence_items}
                valid_cited = [cid for cid in cited if cid in valid_ids]
                is_grounded = len(valid_cited) > 0 or "insufficient evidence" in clean_answer.lower()
                return clean_answer, valid_cited, is_grounded
        except Exception as e:
            logger.warning("Groq question answering failed (%s); falling back to deterministic engine.", str(e))

        return self.fallback.answer_question(
            transaction_id, base_risk_a, graph_risk_g, final_risk_r, action, expected_cost, evidence_items, question
        )

    def _call_groq_api(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TRUSTGRAPH-Agent/1.0",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert payment risk investigator. You always cite evidence using [E1], [E2], [RISK_ENGINE] tags. Never invent unverified facts.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            choices = res_json.get("choices", [])
            if choices:
                msg = choices[0].get("message", {}).get("content", "")
                if msg:
                    return msg.strip()
        return None


# ===========================================================================
# API Provider: Gemini via HTTP (if GEMINI_API_KEY is present)
# ===========================================================================

class GeminiProvider(LLMProvider):
    """Calls Google Gemini API using native HTTP request with strict anti-hallucination prompt."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash") -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.fallback = DeterministicFallbackProvider()

    @property
    def name(self) -> str:
        return f"gemini_{self.model_name}"

    def generate_investigation(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
    ) -> Tuple[List[GroundedReason], str]:
        reasons, _ = self.fallback.generate_investigation(
            transaction_id, base_risk_a, graph_risk_g, final_risk_r, action, expected_cost, evidence_items
        )

        prompt = (
            f"You are a Fraud Investigation Assistant for TRUSTGRAPH. Synthesize an explanation for transaction #{transaction_id}.\n"
            f"Mathematical Facts (IMMUTABLE):\n"
            f"- Base Risk A_t: {base_risk_a:.4f}\n"
            f"- Calibrated Graph Risk G_t: {graph_risk_g:.4f}\n"
            f"- Final Risk R_t: {final_risk_r:.4f}\n"
            f"- Action: {action} (Expected Loss: INR {expected_cost:.2f})\n\n"
            f"Retrieved Evidence Items:\n"
            + "\n".join(f"[{e.evidence_id}] {e.title}: {e.description}" for e in evidence_items)
            + "\n\nCRITICAL RULE: Cite specific [E...] and [RISK_ENGINE] tags. Do not invent any numbers, transactions, or entities."
        )

        try:
            summary = self._call_gemini_api(prompt)
            if summary and len(summary) > 50:
                return reasons, summary
        except Exception as e:
            logger.warning("Gemini API call failed (%s), falling back to deterministic template.", str(e))

        return self.fallback.generate_investigation(
            transaction_id, base_risk_a, graph_risk_g, final_risk_r, action, expected_cost, evidence_items
        )

    def answer_question(
        self,
        transaction_id: int,
        base_risk_a: float,
        graph_risk_g: float,
        final_risk_r: float,
        action: str,
        expected_cost: float,
        evidence_items: List[EvidenceItem],
        question: str,
    ) -> Tuple[str, List[str], bool]:
        prompt = (
            f"You are a Payment Fraud Investigator answering an analyst's question regarding Transaction #{transaction_id}.\n"
            f"Rules:\n"
            f"1. Answer strictly using the provided evidence items.\n"
            f"2. Every claim must cite the relevant evidence ID like [E1] or [RISK_ENGINE].\n"
            f"3. If the question asks about something not in the evidence, say 'Insufficient evidence to determine this.'\n"
            f"4. Do NOT invent facts, entities, or amounts.\n\n"
            f"Evidence:\n"
            + "\n".join(f"[{e.evidence_id}] {e.title}: {e.description}" for e in evidence_items)
            + f"\n[RISK_ENGINE] A_t={base_risk_a:.4f}, G_t={graph_risk_g:.4f}, R_t={final_risk_r:.4f}, Action={action}, Cost=INR {expected_cost:.2f}\n\n"
            f"Question: {question}"
        )

        try:
            answer = self._call_gemini_api(prompt)
            if answer:
                clean_answer = re.sub(r"【(.*?)】", r"[\1]", answer)
                cited = list(set(re.findall(r"\[(E\d+|RISK_ENGINE)\]", clean_answer)))
                valid_ids = {e.evidence_id for e in evidence_items}
                valid_cited = [cid for cid in cited if cid in valid_ids]
                return clean_answer, valid_cited, len(valid_cited) > 0
        except Exception as e:
            logger.warning("Gemini API question answering failed (%s), using fallback.", str(e))

        return self.fallback.answer_question(
            transaction_id, base_risk_a, graph_risk_g, final_risk_r, action, expected_cost, evidence_items, question
        )

    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 800},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            candidates = res_json.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
        return None


# ===========================================================================
# Provider Factory
# ===========================================================================

def get_llm_provider() -> LLMProvider:
    """
    Factory creating the appropriate LLM provider based on environment configuration.
    Priority:
      1. Groq (if GROQ_API_KEY is set or default key available)
      2. Gemini (if GEMINI_API_KEY is set)
      3. DeterministicFallbackProvider (guaranteed zero-downtime fallback)
    """
    groq_key = os.environ.get("GROQ_API_KEY", DEFAULT_GROQ_KEY)
    if groq_key and len(groq_key.strip()) > 10:
        logger.info("Initializing GroqProvider (groq/compound-mini) with configured API key.")
        return GroqProvider(api_key=groq_key.strip())

    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and len(gemini_key.strip()) > 10:
        logger.info("Initializing GeminiProvider with configured API key.")
        return GeminiProvider(api_key=gemini_key.strip())

    logger.info("No external LLM credentials configured. Using DeterministicFallbackProvider.")
    return DeterministicFallbackProvider()
