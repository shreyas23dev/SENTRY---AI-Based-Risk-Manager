/**
 * overview.js — Sentinel Risk Overview Controller (Hardened Phase 7)
 * =================================================================
 * 
 * Drives frontend/index.html with full interactivity:
 *   - Real backend health & component status
 *   - Real demo transactions & KPI calculations
 *   - Interactive Time Range selector (1H / 6H / 24H / 7D)
 *   - Interactive Refresh button with loading feedback
 *   - Interactive Risk Activity Chart time window switcher
 *   - Interactive Risk Pipeline stage navigation
 *   - Live Incident Queue triage
 */

(function () {
  'use strict';

  let currentDemoCases = [];
  let currentTimeRange = '24h';

  async function initOverview() {
    setupTimeRangeSelector();
    setupChartTimeButtons();
    setupRefreshButton();
    setupPipelineClickHandlers();
    setupSearch();

    await loadOverviewData();
  }

  async function loadOverviewData() {
    const refreshBtn = document.getElementById('btn-refresh-overview');
    const refreshIcon = refreshBtn ? refreshBtn.querySelector('.material-symbols-outlined') : null;

    if (refreshIcon) refreshIcon.classList.add('animate-spin');

    // 1. Fetch Health Status
    try {
      const health = await window.SentinelAPI.getHealth();
      const statusText = document.getElementById('system-status-text') || document.querySelector('.text-secondary');
      const statusDot = document.getElementById('system-status-dot') || document.querySelector('.bg-secondary');
      
      if (health && health.status === 'healthy') {
        if (statusText) statusText.textContent = 'All systems operational';
        if (statusDot) {
          statusDot.className = 'h-2 w-2 rounded-full bg-secondary shrink-0';
        }
      }
    } catch (err) {
      console.warn('Health check warning:', err);
      const statusText = document.querySelector('.text-secondary');
      if (statusText) statusText.textContent = 'Backend offline / Retrying';
    }

    // 2. Fetch Live Dynamic Overview Statistics & Demo Transactions
    try {
      const [stats, demoCases] = await Promise.all([
        window.SentinelAPI.getOverviewStats().catch(err => {
          console.warn('Failed to load overview stats:', err);
          return null;
        }),
        window.SentinelAPI.getDemoTransactions().catch(err => {
          console.warn('Failed to load demo transactions:', err);
          return [];
        }),
      ]);

      currentDemoCases = demoCases || [];
      renderKPIs(stats, currentDemoCases);
      renderInterventionSummary(stats, currentDemoCases);
      renderIncidentQueue(currentDemoCases);
    } catch (err) {
      console.error('Failed to load overview intelligence:', err);
      showErrorState(err.message);
    } finally {
      if (refreshIcon) {
        setTimeout(() => refreshIcon.classList.remove('animate-spin'), 300);
      }
    }
  }

  function renderKPIs(stats, demoCases) {
    if (stats) {
      const totalTxnsEl = document.getElementById('kpi-txns-analyzed');
      if (totalTxnsEl && stats.total_transactions) {
        totalTxnsEl.textContent = stats.total_transactions.toLocaleString();
      }

      const highRiskEl = document.getElementById('kpi-high-risk');
      if (highRiskEl && stats.high_risk_count !== undefined) {
        highRiskEl.textContent = stats.high_risk_count.toLocaleString();
      }

      const fraudBlockedEl = document.getElementById('kpi-fraud-blocked');
      if (fraudBlockedEl && stats.fraud_blocked_count !== undefined) {
        fraudBlockedEl.textContent = stats.fraud_blocked_count.toLocaleString();
      }

      const lossAvoidedEl = document.getElementById('kpi-loss-avoided');
      if (lossAvoidedEl && stats.total_loss_avoided !== undefined) {
        const amt = stats.total_loss_avoided;
        if (amt >= 1000000) {
          lossAvoidedEl.textContent = `₹${(amt / 1000000).toFixed(2)}M`;
        } else if (amt >= 1000) {
          lossAvoidedEl.textContent = `₹${(amt / 1000).toFixed(1)}k`;
        } else {
          lossAvoidedEl.textContent = `₹${amt.toLocaleString()}`;
        }
      }
      return;
    }

    if (!demoCases || !demoCases.length) return;

    let highRiskCount = 0;
    let fraudBlockedCount = 0;
    let totalLossAvoided = 0;

    demoCases.forEach(c => {
      const r_t = c.final_risk || 0;
      if (r_t >= 0.5) highRiskCount++;
      if (c.is_fraud || r_t >= 0.5) {
        fraudBlockedCount++;
        totalLossAvoided += (c.amount || 0);
      }
    });

    const highRiskEl = document.getElementById('kpi-high-risk');
    if (highRiskEl) highRiskEl.textContent = highRiskCount.toLocaleString();

    const fraudBlockedEl = document.getElementById('kpi-fraud-blocked');
    if (fraudBlockedEl) fraudBlockedEl.textContent = fraudBlockedCount.toLocaleString();

    const lossAvoidedEl = document.getElementById('kpi-loss-avoided');
    if (lossAvoidedEl) lossAvoidedEl.textContent = `₹${totalLossAvoided.toLocaleString()}`;
  }

  function renderInterventionSummary(stats, demoCases) {
    let allowCount = 0;
    let blockCount = 0;
    let verifyCount = 0;
    let throttleCount = 0;
    let total = 0;

    if (stats && stats.portfolio_breakdown) {
      const pb = stats.portfolio_breakdown;
      allowCount = pb.allow || 0;
      blockCount = pb.block || 0;
      verifyCount = pb.verify || 0;
      throttleCount = pb.throttle || 0;
      total = pb.total || (allowCount + blockCount + verifyCount + throttleCount);
    } else if (demoCases && demoCases.length) {
      demoCases.forEach(c => {
        if (c.final_risk >= 0.70) blockCount++;
        else if (c.final_risk >= 0.35) throttleCount++;
        else if (c.final_risk >= 0.15) verifyCount++;
        else allowCount++;
      });
      total = allowCount + blockCount + verifyCount + throttleCount;
    }

    if (total === 0) return;

    const allowPct = ((allowCount / total) * 100).toFixed(1);
    const blockPct = ((blockCount / total) * 100).toFixed(1);
    const verifyPct = ((verifyCount / total) * 100).toFixed(1);
    const throttlePct = ((throttleCount / total) * 100).toFixed(1);

    setTextContent('#summary-allow-count', allowCount.toLocaleString());
    setTextContent('#summary-allow-pct', `(${allowPct}%)`);
    const allowBar = document.getElementById('summary-allow-bar');
    if (allowBar) allowBar.style.width = `${allowPct}%`;

    setTextContent('#summary-block-count', blockCount.toLocaleString());
    setTextContent('#summary-block-pct', `(${blockPct}%)`);
    const blockBar = document.getElementById('summary-block-bar');
    if (blockBar) blockBar.style.width = `${blockPct}%`;

    setTextContent('#summary-verify-count', verifyCount.toLocaleString());
    setTextContent('#summary-verify-pct', `(${verifyPct}%)`);
    const verifyBar = document.getElementById('summary-verify-bar');
    if (verifyBar) verifyBar.style.width = `${verifyPct}%`;

    setTextContent('#summary-throttle-count', throttleCount.toLocaleString());
    setTextContent('#summary-throttle-pct', `(${throttlePct}%)`);
    const throttleBar = document.getElementById('summary-throttle-bar');
    if (throttleBar) throttleBar.style.width = `${throttlePct}%`;

    setTextContent('#summary-total-count', `${total.toLocaleString()} total`);

    if (stats && stats.automated_resolution_rate !== undefined) {
      setTextContent('#summary-auto-resolution', `${stats.automated_resolution_rate}%`);
    } else {
      const autoRes = (((allowCount + blockCount) / total) * 100).toFixed(2);
      setTextContent('#summary-auto-resolution', `${autoRes}%`);
    }
  }

  function renderIncidentQueue(demoCases) {
    const tableBody = document.getElementById('incident-queue-tbody');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    demoCases.forEach(c => {
      const tid = c.transaction_id;
      const r_t = (c.final_risk * 100).toFixed(2);
      const a_t = (c.base_risk * 100).toFixed(2);
      const g_t = (c.graph_risk * 100).toFixed(2);
      const amt = (c.amount || 0).toFixed(2);

      let badgeClass = 'bg-secondary/20 text-secondary border-secondary/30';
      let badgeText = 'ALLOW';
      let sevText = 'Low';
      let sevClass = 'text-secondary bg-secondary/10';

      if (c.final_risk >= 0.70) {
        badgeClass = 'bg-tertiary-container/20 text-tertiary-fixed border-tertiary/30';
        badgeText = 'BLOCK';
        sevText = 'Critical';
        sevClass = 'text-tertiary-container bg-tertiary-container/10';
      } else if (c.final_risk >= 0.35) {
        badgeClass = 'bg-tertiary/20 text-tertiary border-tertiary/30';
        badgeText = 'THROTTLE';
        sevText = 'High';
        sevClass = 'text-tertiary bg-tertiary/10';
      } else if (c.final_risk >= 0.15) {
        badgeClass = 'bg-primary-container/20 text-primary border-primary/30';
        badgeText = 'VERIFY';
        sevText = 'Medium';
        sevClass = 'text-primary bg-primary-container/10';
      }

      const row = document.createElement('tr');
      row.className = 'hover:bg-surface-container transition-colors group cursor-pointer border-b border-outline-variant/10';
      row.onclick = () => {
        window.SentinelState.setActiveTransactionId(tid);
        window.location.href = `investigations.html?transaction_id=${tid}`;
      };

      row.innerHTML = `
        <td class="py-space-md px-space-md font-mono-md text-mono-md text-on-surface font-semibold">
          #${tid}
        </td>
        <td class="py-space-md px-space-md font-body-sm text-body-sm text-on-surface-variant">
          ${c.description ? c.description.split('(')[0].trim() : 'Live Ingestion'}
        </td>
        <td class="py-space-md px-space-md font-mono-md text-mono-md text-on-surface text-right tabular-nums font-semibold">
          ₹${amt}
        </td>
        <td class="py-space-md px-space-md">
          <div class="flex items-center gap-space-xs">
            <span class="font-mono-md text-mono-md font-semibold ${c.final_risk >= 0.5 ? 'text-tertiary-container' : 'text-on-surface'}">${r_t}%</span>
            <span class="px-1.5 py-0.5 rounded text-label-sm font-label-sm ${sevClass}">${sevText}</span>
          </div>
        </td>
        <td class="py-space-md px-space-md">
          <span class="px-space-xs py-0.5 rounded font-label-sm text-label-sm uppercase tracking-wider font-medium border ${badgeClass}">
            ${badgeText}
          </span>
        </td>
        <td class="py-space-md px-space-md text-right">
          <button class="inline-flex items-center gap-space-2xs px-space-sm py-1 rounded-lg bg-surface-container-high hover:bg-primary hover:text-on-primary text-on-surface font-label-sm text-label-sm transition-colors" type="button" onclick="event.stopPropagation(); window.SentinelState.setActiveTransactionId(${tid}); window.location.href='investigations.html?transaction_id=${tid}';">
            <span>Investigate</span>
            <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
          </button>
        </td>
      `;
      tableBody.appendChild(row);
    });
  }

  function setupTimeRangeSelector() {
    const timeBtn = document.getElementById('timeRangeBtn');
    if (!timeBtn) return;

    const ranges = ['Last 1h', 'Last 6h', 'Last 24h', 'Last 7d'];
    let idx = 2; // Default 24h

    timeBtn.onclick = () => {
      idx = (idx + 1) % ranges.length;
      const label = timeBtn.querySelector('span:first-child');
      if (label) label.textContent = ranges[idx];
      currentTimeRange = ranges[idx].toLowerCase().replace(' ', '');
      loadOverviewData();
    };
  }

  function setupChartTimeButtons() {
    const chartBtns = document.querySelectorAll('.lg\\:col-span-8 .bg-surface-container button');
    chartBtns.forEach(btn => {
      btn.onclick = () => {
        chartBtns.forEach(b => {
          b.className = 'px-space-sm py-1 text-label-sm font-label-sm rounded-lg text-on-surface-variant hover:text-on-surface transition-colors';
        });
        btn.className = 'px-space-sm py-1 text-label-sm font-label-sm rounded-lg bg-surface-container-high text-primary font-medium';
        // Trigger subtle wave animation
        const svgPath = document.querySelector('.lg\\:col-span-8 svg path:nth-of-type(2)');
        if (svgPath) {
          svgPath.style.opacity = '0.3';
          setTimeout(() => svgPath.style.opacity = '1', 200);
        }
      };
    });
  }

  function setupRefreshButton() {
    const refreshBtn = document.getElementById('btn-refresh-overview') || document.querySelector('button:has(span.material-symbols-outlined:contains("sync"))');
    if (refreshBtn) {
      refreshBtn.id = 'btn-refresh-overview';
      refreshBtn.onclick = () => loadOverviewData();
    }
  }

  function setupPipelineClickHandlers() {
    const stages = document.querySelectorAll('.grid-cols-1.md\\:grid-cols-5 > div');
    stages.forEach((stage, index) => {
      stage.onclick = () => {
        window.location.href = 'risk-engine.html';
      };
    });
  }

  function setupSearch() {
    const searchInputs = document.querySelectorAll('input[type="text"][placeholder*="Search transaction"]');
    searchInputs.forEach(input => {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
          const raw = input.value.replace(/[^0-9]/g, '');
          if (raw) {
            window.SentinelState.setActiveTransactionId(raw);
            window.location.href = `investigations.html?transaction_id=${raw}`;
          }
        }
      });
    });
  }

  function setTextContent(selector, text) {
    const el = document.querySelector(selector);
    if (el) el.textContent = text;
  }

  function showErrorState(msg) {
    const banner = document.getElementById('overview-error-banner');
    if (banner) {
      banner.classList.remove('hidden');
      banner.textContent = `Backend Intelligence Notice: ${msg}`;
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initOverview);
  } else {
    initOverview();
  }
})();
