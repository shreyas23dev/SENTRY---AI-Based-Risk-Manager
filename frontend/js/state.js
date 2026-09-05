/**
 * state.js — Frontend State Management for Sentinel Console
 * ==========================================================
 * 
 * Tracks the globally selected activeTransactionId and activeDemoCase across all screens.
 * Provides query parameter synchronization, session caching, and reactive event dispatching.
 */

(function (window) {
  'use strict';

  const STORAGE_KEY = 'SENTINEL_ACTIVE_TRANSACTION_ID';
  const DEMO_CASE_STORAGE_KEY = 'SENTINEL_ACTIVE_DEMO_CASE';
  const DEFAULT_FALLBACK_TXN_ID = 3570805;

  // Real held-out TEST transactions verified across pipeline
  const DEMO_CASES = [
    {
      caseNum: 1,
      txnId: 3570805,
      label: "Agreement → BLOCK",
      desc: "High ML risk & documented prior fraud history. Base ML & Slow-Burn analysts agree.",
      isFraud: 1,
      action: "BLOCK",
      base_risk: 0.9921,
      graph_risk: 0.7375,
      final_risk: 0.9922,
      amount: 82.63,
      council: {
        t_tier: "CRITICAL",
        t_score: "99.2%",
        sb_tier: "CRITICAL",
        sb_score: "100.0%",
        status: "AGREEMENT",
        history: "1 prior txns · 1 prior frauds",
        officer: "Both Transaction Risk Analyst and Slow-Burn Analyst strongly agree on high malicious probability. Immediate **BLOCK** enforced."
      }
    },
    {
      caseNum: 2,
      txnId: 3531382,
      label: "Slow-Burn Disagreement",
      desc: "Base ML sees normal transaction; Slow-Burn detects temporal velocity surge.",
      isFraud: 1,
      action: "ALLOW",
      base_risk: 0.0392,
      graph_risk: 0.5563,
      final_risk: 0.0436,
      amount: 61.48,
      council: {
        t_tier: "LOW",
        t_score: "3.9%",
        sb_tier: "CRITICAL",
        sb_score: "90.0%",
        status: "DISAGREEMENT",
        history: "4 prior txns · 1 prior frauds",
        officer: "**Asymmetric Disagreement (Slow-Burn Evasion)**:  \n- **Real-Time Gateway (T=0)**: Allowed as **LOW RISK** (`Rₜ: 4.36%`) because point-in-time features (`Aₜ: 3.9%`, ₹61.48) appeared normal.  \n- **Slow-Burn Memory (T+60d)**: Correctly warned of **CRITICAL** risk (`Pₜ: 90.0%`) based on historical velocity. The transaction resulted in a confirmed post-settlement chargeback, validating the Council's disagreement warning."
      }
    },
    {
      caseNum: 3,
      txnId: 3488970,
      label: "Cold Start / Insufficient History",
      desc: "Single isolated txn with no prior behavioral history (Pₜ: No Prior History).",
      isFraud: 0,
      action: "ALLOW",
      base_risk: 0.0164,
      graph_risk: 0.0000,
      final_risk: 0.0169,
      amount: 29.00,
      council: {
        t_tier: "LOW",
        t_score: "1.6%",
        sb_tier: "CLEAN",
        sb_score: "0.0%",
        status: "AGREEMENT",
        history: "0 prior txns · 0 prior frauds",
        officer: "Cold start entity with zero prior transaction history. Both analysts concur that transaction payload is benign. Standard **ALLOW**."
      }
    },
    {
      caseNum: 4,
      txnId: 3488964,
      label: "Clean + Historical Risk Context → ALLOW",
      desc: "Low transaction risk; allowed despite historical risk context in relational association.",
      isFraud: 0,
      action: "ALLOW",
      base_risk: 0.0014,
      graph_risk: 0.4000,
      final_risk: 0.0039,
      amount: 224.00,
      council: {
        t_tier: "LOW",
        t_score: "0.1%",
        sb_tier: "CRITICAL",
        sb_score: "95.0%",
        status: "DISAGREEMENT",
        history: "252 prior txns · 11 prior frauds",
        officer: "The Transaction Risk Analyst reports low transaction-level risk (Aₜ = 0.1%), while Slow-Burn Analyst reports high behavioral accumulator (Pₜ = 95.0%) from historical entity events. The calibrated risk engine preserves ALLOW (Rₜ = 0.4%) based on low transaction amount and strong feature conformity."
      }
    },
    {
      caseNum: 5,
      txnId: 3489068,
      label: "Step-Up Auth → VERIFY",
      desc: "Elevated instantaneous & graph risk. Risk engine cost model determines OTP step-up verification is optimal.",
      isFraud: 1,
      action: "VERIFY",
      base_risk: 0.2931,
      graph_risk: 0.7750,
      final_risk: 0.3001,
      amount: 150.00,
      council: {
        t_tier: "MEDIUM",
        t_score: "29.3%",
        sb_tier: "CLEAN",
        sb_score: "0.0%",
        status: "INSUFFICIENT_HISTORY",
        history: "0 prior txns · 0 prior frauds",
        officer: "Transaction risk analyst reports elevated payload risk (Aₜ = 29.3%) and significant relational graph anomaly (Gₜ = 77.5%). Under the balanced cost scenario, step-up verification (**VERIFY**) provides maximum loss protection against potential fraud while avoiding full customer insult."
      }
    },
    {
      caseNum: 6,
      txnId: 3512832,
      label: "Behavioral Velocity → THROTTLE",
      desc: "Hardware multiplexing & persistent slow-burn indicators with ₹34 amount. Rate limiting (THROTTLE) minimizes merchant loss.",
      isFraud: 1,
      action: "THROTTLE",
      base_risk: 0.1683,
      graph_risk: 0.9625,
      final_risk: 0.1825,
      amount: 34.00,
      council: {
        t_tier: "MEDIUM",
        t_score: "16.8%",
        sb_tier: "CRITICAL",
        sb_score: "95.0%",
        status: "DISAGREEMENT",
        history: "8,783 prior txns · 343 prior frauds",
        officer: "**Slow-Burn Behavioral Throttle**:  \n- **Payload Assessment**: Moderate instantaneous risk (`Aₜ: 16.8%`, ₹34.00) with elevated burst velocity (`C1: 21.0`).  \n- **Relational Graph**: Severe entity history (343 prior frauds) and hardware shared across 23 distinct entities.  \n- **Optimal Action**: At ₹34, customer friction from OTP exceeds potential fraud recovery. The cost engine prescribes velocity **THROTTLE** (E[cost] = ₹172.98 vs ₹181.24 for VERIFY)."
      }
    }
  ];

  function getActiveTransactionId() {
    // 1. Check URL search param
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('transaction_id')) {
      const paramVal = parseInt(urlParams.get('transaction_id'), 10);
      if (!isNaN(paramVal) && paramVal > 0) {
        sessionStorage.setItem(STORAGE_KEY, paramVal.toString());
        try { localStorage.setItem(STORAGE_KEY, paramVal.toString()); } catch (_) {}
        return paramVal;
      }
    }

    // 2. Check sessionStorage / localStorage
    let stored = null;
    try {
      stored = sessionStorage.getItem(STORAGE_KEY) || localStorage.getItem(STORAGE_KEY);
    } catch (_) {}

    if (stored) {
      const parsed = parseInt(stored, 10);
      if (!isNaN(parsed) && parsed > 0) {
        return parsed;
      }
    }

    return DEFAULT_FALLBACK_TXN_ID;
  }

  function setActiveTransactionId(transactionId) {
    if (!transactionId) return;
    const numId = parseInt(transactionId, 10);
    if (isNaN(numId)) return;

    try {
      sessionStorage.setItem(STORAGE_KEY, numId.toString());
      localStorage.setItem(STORAGE_KEY, numId.toString());
    } catch (_) {}

    // Update URL without page reload if on relevant page
    if (window.history && window.history.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.set('transaction_id', numId.toString());
      window.history.replaceState({}, '', url.toString());
    }

    // Dispatch custom event for active screen listeners
    window.dispatchEvent(new CustomEvent('sentinel:transactionChange', {
      detail: { transactionId: numId }
    }));
  }

  function getActiveDemoCase() {
    // 1. Check URL search param 'demo_case'
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('demo_case')) {
      const p = parseInt(urlParams.get('demo_case'), 10);
      if (!isNaN(p) && p >= 1 && p <= 6) {
        try {
          sessionStorage.setItem(DEMO_CASE_STORAGE_KEY, p.toString());
          localStorage.setItem(DEMO_CASE_STORAGE_KEY, p.toString());
        } catch (_) {}
        return p;
      }
    }

    // 2. Check storage
    let stored = null;
    try {
      stored = sessionStorage.getItem(DEMO_CASE_STORAGE_KEY) || localStorage.getItem(DEMO_CASE_STORAGE_KEY);
    } catch (_) {}

    if (stored) {
      const parsed = parseInt(stored, 10);
      if (!isNaN(parsed) && parsed >= 1 && parsed <= 6) {
        return parsed;
      }
    }

    // 3. Infer from active transaction if it matches one of the 6 demo cases
    const activeTxn = getActiveTransactionId();
    const match = DEMO_CASES.find(c => c.txnId === activeTxn);
    if (match) {
      return match.caseNum;
    }

    return null;
  }

  function setActiveDemoCase(caseNum) {
    if (!caseNum) {
      clearActiveDemoCase();
      return;
    }
    const num = parseInt(caseNum, 10);
    if (isNaN(num) || num < 1 || num > 6) return;

    try {
      sessionStorage.setItem(DEMO_CASE_STORAGE_KEY, num.toString());
      localStorage.setItem(DEMO_CASE_STORAGE_KEY, num.toString());
    } catch (_) {}

    // Update URL param if on investigation page
    if (window.history && window.history.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.set('demo_case', num.toString());
      window.history.replaceState({}, '', url.toString());
    }

    window.dispatchEvent(new CustomEvent('sentinel:demoCaseChange', {
      detail: { caseNum: num }
    }));
  }

  function clearActiveDemoCase() {
    try {
      sessionStorage.removeItem(DEMO_CASE_STORAGE_KEY);
      localStorage.removeItem(DEMO_CASE_STORAGE_KEY);
    } catch (_) {}

    if (window.history && window.history.replaceState) {
      const url = new URL(window.location.href);
      url.searchParams.delete('demo_case');
      window.history.replaceState({}, '', url.toString());
    }

    window.dispatchEvent(new CustomEvent('sentinel:demoCaseChange', {
      detail: { caseNum: null }
    }));
  }

  function selectDemoCase(caseNum) {
    const c = DEMO_CASES.find(item => item.caseNum === caseNum);
    if (!c) return;

    setActiveDemoCase(caseNum);
    setActiveTransactionId(c.txnId);

    // If currently on investigations.html (or root / investigations view), reload context;
    // Otherwise navigate to investigations.html
    const pathname = window.location.pathname;
    const isInvestigations = pathname.endsWith('investigations.html') || pathname.includes('investigations');
    if (!isInvestigations) {
      window.location.href = `investigations.html?transaction_id=${c.txnId}&demo_case=${caseNum}`;
    }
  }

  window.SentinelState = {
    getActiveTransactionId,
    setActiveTransactionId,
    getActiveDemoCase,
    setActiveDemoCase,
    clearActiveDemoCase,
    selectDemoCase,
    getDemoCases: () => DEMO_CASES,
  };
})(window);
