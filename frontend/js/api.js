/**
 * api.js — Centralized Sentinel API Client Layer
 * ========================================================
 * 
 * Production client connecting the Stitch frontend to the FastAPI Sentinel backend.
 * Provides unified request handling, error wrapping, timeout management,
 * and base URL configuration.
 */

(function (window) {
  'use strict';

  // Determine API Base URL dynamically
  function getApiBaseUrl() {
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.has('api_url')) {
      const customUrl = urlParams.get('api_url');
      localStorage.setItem('SENTINEL_API_BASE_URL', customUrl);
      return customUrl;
    }
    return localStorage.getItem('SENTINEL_API_BASE_URL') ||
           window.SENTINEL_API_BASE_URL ||
           'http://localhost:8000';
  }

  function setApiBaseUrl(url) {
    if (url) {
      localStorage.setItem('SENTINEL_API_BASE_URL', url);
    }
  }

  /**
   * Core fetch wrapper with timeout, JSON parsing, and unified error handling.
   */
  async function apiRequest(endpoint, options = {}) {
    const baseUrl = getApiBaseUrl().replace(/\/+$/, '');
    const url = `${baseUrl}${endpoint.startsWith('/') ? endpoint : '/' + endpoint}`;
    const timeoutMs = options.timeoutMs || 25000;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const defaultHeaders = {
      'Accept': 'application/json',
      'Content-Type': 'application/json',
    };

    const fetchConfig = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...(options.headers || {}),
      },
      signal: controller.signal,
    };

    try {
      const response = await fetch(url, fetchConfig);
      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorDetail = `HTTP ${response.status} ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            errorDetail = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail);
          } else if (errData && errData.message) {
            errorDetail = errData.message;
          }
        } catch (_) {}
        const error = new Error(errorDetail);
        error.status = response.status;
        throw error;
      }

      return await response.json();
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') {
        const timeoutError = new Error(`Request timed out after ${timeoutMs / 1000}s. Please verify Sentinel backend is running.`);
        timeoutError.isTimeout = true;
        throw timeoutError;
      }
      if (err instanceof TypeError && err.message.includes('fetch')) {
        const netError = new Error(`Unable to connect to Sentinel backend at ${baseUrl}. Please ensure the backend service is running on port 8000.`);
        netError.isNetworkError = true;
        throw netError;
      }
      throw err;
    }
  }

  // Exposed API Methods
  const SentinelAPI = {
    getBaseUrl: getApiBaseUrl,
    setBaseUrl: setApiBaseUrl,

    /** Health & readiness status */
    async getHealth() {
      return apiRequest('/api/v1/health');
    },

    /** List representative demonstration transactions */
    async getDemoTransactions() {
      return apiRequest('/api/v1/investigation/demo-transactions');
    },

    /** Fetch portfolio-level dynamic overview statistics and KPI metrics */
    async getOverviewStats() {
      return apiRequest('/api/v1/overview/stats');
    },

    /** Fetch risk summary (A_t, G_t, R_t, decision, amount) for a transaction */
    async getRiskSummary(transactionId) {
      return apiRequest(`/api/v1/risk/${encodeURIComponent(transactionId)}`);
    },

    /** Fetch 1-hop or 2-hop graph neighborhood view for D3 force-directed visualization */
    async getGraph(transactionId, maxHops = 2) {
      return apiRequest(`/api/v1/risk/${encodeURIComponent(transactionId)}/graph?max_hops=${maxHops}`);
    },

    /** Fetch ranked, provenance-backed evidence items for a transaction */
    async getEvidence(transactionId) {
      return apiRequest(`/api/v1/risk/${encodeURIComponent(transactionId)}/evidence`);
    },

    /** Trigger GraphRAG AI investigation report generation */
    async investigateTransaction(transactionId, scenario = 'balanced') {
      return apiRequest(`/api/v1/risk/${encodeURIComponent(transactionId)}/investigate?scenario=${encodeURIComponent(scenario)}`, {
        method: 'POST',
      });
    },

    /** Fetch multi-analyst risk council evaluation including Slow-Burn analyst */
    async getCouncil(transactionId, scenario = 'balanced') {
      return apiRequest(`/api/v1/risk/${encodeURIComponent(transactionId)}/council?scenario=${encodeURIComponent(scenario)}`);
    },

    /** Ask question grounded in retrieved knowledge graph evidence */
    async askInvestigator(transactionId, question) {
      return apiRequest(`/api/v1/risk/${encodeURIComponent(transactionId)}/ask`, {
        method: 'POST',
        body: JSON.stringify({ question }),
      });
    },

    /** Evaluate raw transaction request */
    async evaluateRisk(payload) {
      return apiRequest('/api/v1/risk/evaluate', {
        method: 'POST',
        body: JSON.stringify(payload),
      });
    },
  };

  window.SentinelAPI = SentinelAPI;
})(window);
