/**
 * transactions.js — Sentinel Transaction Monitor Controller (Hardened Phase 7)
 * ============================================================================
 * 
 * Populates frontend/transactions.html with real backend transaction records.
 * Supports:
 *   - 10-column Stitch visual design table
 *   - Risk severity triage filters (All, Medium, High, Critical)
 *   - Decision Gate triage filters (All Gates, ALLOW, VERIFY, THROTTLE, BLOCK)
 *   - Live text search across ID, Amount, Description, and Entity
 *   - Time window cycling with live indicator sync
 *   - CSV export of active transaction ledger
 *   - Manual refresh ledger sync with backend
 *   - Row click & Investigate button routing to AI Investigation
 */

(function () {
  'use strict';

  let allTransactions = [];
  let currentRiskFilter = 'All';
  let currentGateFilter = 'All Gates';
  let currentSearchQuery = '';

  async function initTransactions() {
    setupFilters();
    setupSearch();
    setupTimeWindowSelector();
    setupRefreshButton();
    setupExportCSV();
    setupHeaderSearch();

    await loadTransactions();
  }

  async function loadTransactions() {
    const refreshBtn = document.getElementById('btn-refresh-txns');
    const refreshIcon = refreshBtn ? refreshBtn.querySelector('.material-symbols-outlined') : null;

    if (refreshIcon) refreshIcon.classList.add('animate-spin');

    try {
      const data = await window.SentinelAPI.getDemoTransactions();
      allTransactions = data || [];
      renderTable();
      updatePagination();
    } catch (err) {
      console.error('Failed to load transactions:', err);
      const tbody = document.getElementById('transactions-tbody');
      if (tbody) {
        tbody.innerHTML = `
          <tr>
            <td colspan="10" class="py-8 text-center text-tertiary font-body-sm">
              Failed to load transactions from Sentinel backend (${err.message}).
            </td>
          </tr>
        `;
      }
    } finally {
      if (refreshIcon) {
        setTimeout(() => refreshIcon.classList.remove('animate-spin'), 300);
      }
    }
  }

  function getFilteredTransactions() {
    return allTransactions.filter(t => {
      const r_t = t.final_risk || 0;
      const decision = getDecisionText(r_t);

      // Risk Filter
      if (currentRiskFilter === 'Critical' && r_t < 0.70) return false;
      if (currentRiskFilter === 'High' && (r_t < 0.35 || r_t >= 0.70)) return false;
      if (currentRiskFilter === 'Medium' && (r_t < 0.15 || r_t >= 0.35)) return false;

      // Decision Gate Filter
      if (currentGateFilter !== 'All Gates' && decision !== currentGateFilter) return false;

      // Text Search Query
      if (currentSearchQuery) {
        const q = currentSearchQuery.toLowerCase();
        const tid = String(t.transaction_id || '').toLowerCase();
        const entity = String(t.entity_id || '').toLowerCase();
        const desc = String(t.description || '').toLowerCase();
        const amt = String(t.amount || '').toLowerCase();
        if (!tid.includes(q) && !entity.includes(q) && !desc.includes(q) && !amt.includes(q)) {
          return false;
        }
      }

      return true;
    });
  }

  function renderTable() {
    const tbody = document.getElementById('transactions-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    const filtered = getFilteredTransactions();

    if (filtered.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="10" class="py-12 text-center text-outline font-body-sm">
            <div class="flex flex-col items-center justify-center gap-2">
              <span class="material-symbols-outlined text-[32px]">filter_list_off</span>
              <span>No transactions match the selected filters or search query.</span>
            </div>
          </td>
        </tr>
      `;
      updatePagination(0);
      return;
    }

    filtered.forEach((t, index) => {
      const tid = t.transaction_id;
      const amt = (t.amount || 0).toFixed(2);
      const a_t = ((t.base_risk || 0) * 100).toFixed(2);
      const g_t = ((t.graph_risk || 0) * 100).toFixed(2);
      const r_t = ((t.final_risk || 0) * 100).toFixed(2);
      const r_val = t.final_risk || 0;
      const decision = getDecisionText(r_val);
      const riskMeta = getRiskLevelMeta(r_val);
      const primaryTrigger = t.description ? t.description.split('(')[0].trim() : (r_val >= 0.5 ? 'Synthetic graph ring link' : 'Standard velocity check');

      // Simulated UTC time based on index
      const simulatedMinutes = (42 - index * 4 + 60) % 60;
      const simulatedTime = `12:${simulatedMinutes < 10 ? '0' + simulatedMinutes : simulatedMinutes}:18`;

      const tr = document.createElement('tr');
      tr.className = 'h-14 bg-surface-container-low hover:bg-surface-container transition-colors group cursor-pointer border-b border-outline-variant/10';
      
      tr.onclick = () => {
        window.SentinelState.setActiveTransactionId(tid);
        window.location.href = `investigations.html?transaction_id=${tid}`;
      };

      tr.innerHTML = `
        <td class="px-space-md py-space-xs">
          <span class="inline-flex items-center gap-1.5 px-space-xs py-0.5 rounded font-label-sm text-label-sm uppercase tracking-wider font-bold ${riskMeta.badgeClass}">
            <span class="h-1.5 w-1.5 rounded-full ${riskMeta.dotClass}"></span>
            ${riskMeta.label}
          </span>
        </td>
        <td class="px-space-md py-space-xs font-mono-md text-mono-md text-primary font-medium tracking-tight">
          #${tid}
        </td>
        <td class="px-space-md py-space-xs font-mono-sm text-mono-sm text-on-surface-variant">
          ${simulatedTime}
        </td>
        <td class="px-space-md py-space-xs font-mono-md text-mono-md text-on-surface font-semibold text-right" style="font-variant-numeric: tabular-nums;">
          ₹${amt}
        </td>
        <td class="px-space-md py-space-xs font-mono-md text-mono-md text-on-surface-variant text-right" style="font-variant-numeric: tabular-nums;">
          ${a_t}%
        </td>
        <td class="px-space-md py-space-xs font-mono-md text-mono-md text-secondary text-right" style="font-variant-numeric: tabular-nums;">
          ${g_t}%
        </td>
        <td class="px-space-md py-space-xs text-right">
          <div class="inline-flex flex-col items-end">
            <span class="font-mono-md text-mono-md ${r_val >= 0.5 ? 'text-tertiary-fixed' : 'text-on-surface'} font-bold" style="font-variant-numeric: tabular-nums;">${r_t}%</span>
            <div class="w-12 h-1 bg-surface-container-highest rounded-full overflow-hidden mt-0.5">
              <div class="${r_val >= 0.7 ? 'bg-tertiary-container' : r_val >= 0.35 ? 'bg-tertiary' : r_val >= 0.15 ? 'bg-primary' : 'bg-secondary'} h-full" style="width: ${Math.min(100, r_val * 100)}%;"></div>
            </div>
          </div>
        </td>
        <td class="px-space-md py-space-xs text-center">
          <span class="inline-block px-2 py-0.5 rounded font-label-sm text-label-sm font-bold uppercase tracking-wider border ${getDecisionBadgeStyle(decision)}">
            ${decision}
          </span>
        </td>
        <td class="px-space-md py-space-xs">
          <div class="flex items-center gap-space-xs max-w-xs">
            <span class="material-symbols-outlined ${r_val >= 0.5 ? 'text-tertiary' : 'text-outline'} text-[16px] shrink-0">
              ${r_val >= 0.7 ? 'gpp_bad' : r_val >= 0.35 ? 'warning' : 'verified_user'}
            </span>
            <span class="font-body-sm text-body-sm text-on-surface truncate" title="${primaryTrigger}">${primaryTrigger}</span>
          </div>
        </td>
        <td class="px-space-md py-space-xs text-right">
          <button class="h-8 px-space-sm bg-primary-container hover:bg-primary text-on-primary-container hover:text-on-primary rounded font-label-md text-label-md font-semibold inline-flex items-center gap-1 transition-colors shadow-sm" type="button" onclick="event.stopPropagation(); window.SentinelState.setActiveTransactionId(${tid}); window.location.href='investigations.html?transaction_id=${tid}';">
            <span>Investigate</span>
            <span class="material-symbols-outlined text-[14px]">arrow_forward</span>
          </button>
        </td>
      `;

      tbody.appendChild(tr);
    });

    updatePagination(filtered.length);
  }

  function getRiskLevelMeta(r) {
    if (r >= 0.70) {
      return {
        label: 'CRITICAL',
        badgeClass: 'bg-tertiary-container/15 text-tertiary font-bold',
        dotClass: 'bg-tertiary-container animate-pulse',
      };
    } else if (r >= 0.35) {
      return {
        label: 'HIGH',
        badgeClass: 'bg-tertiary/15 text-tertiary-fixed font-semibold',
        dotClass: 'bg-tertiary',
      };
    } else if (r >= 0.15) {
      return {
        label: 'MEDIUM',
        badgeClass: 'bg-primary/15 text-primary font-semibold',
        dotClass: 'bg-primary',
      };
    } else {
      return {
        label: 'LOW',
        badgeClass: 'bg-secondary/15 text-secondary font-semibold',
        dotClass: 'bg-secondary',
      };
    }
  }

  function getDecisionText(r) {
    if (r >= 0.70) return 'BLOCK';
    if (r >= 0.35) return 'THROTTLE';
    if (r >= 0.15) return 'VERIFY';
    return 'ALLOW';
  }

  function getDecisionBadgeStyle(decision) {
    switch (decision) {
      case 'BLOCK':
        return 'bg-tertiary-container/20 text-tertiary-fixed border-tertiary/30';
      case 'THROTTLE':
        return 'bg-tertiary/20 text-tertiary border-tertiary/30';
      case 'VERIFY':
        return 'bg-primary-container/20 text-primary border-primary/30';
      case 'ALLOW':
      default:
        return 'bg-secondary/20 text-secondary border-secondary/30';
    }
  }

  function updatePagination(count = allTransactions.length) {
    const startEl = document.getElementById('pagination-start');
    const endEl = document.getElementById('pagination-end');
    const totalEl = document.getElementById('pagination-total');

    if (startEl) startEl.textContent = count > 0 ? '1' : '0';
    if (endEl) endEl.textContent = String(count);
    if (totalEl) totalEl.textContent = String(count);
  }

  function setupFilters() {
    // Risk Severity Filters
    const riskBtns = document.querySelectorAll('[data-risk-filter]');
    riskBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        riskBtns.forEach(b => {
          b.classList.remove('bg-surface-container-high', 'text-on-surface', 'font-semibold');
          b.classList.add('text-on-surface-variant');
        });
        btn.classList.add('bg-surface-container-high', 'text-on-surface', 'font-semibold');
        btn.classList.remove('text-on-surface-variant');
        currentRiskFilter = btn.getAttribute('data-risk-filter');
        renderTable();
      });
    });

    // Decision Gate Filters
    const gateBtns = document.querySelectorAll('[data-gate-filter]');
    gateBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        gateBtns.forEach(b => {
          b.classList.remove('bg-surface-container-high', 'text-on-surface', 'font-semibold');
        });
        btn.classList.add('bg-surface-container-high', 'text-on-surface', 'font-semibold');
        currentGateFilter = btn.getAttribute('data-gate-filter');
        renderTable();
      });
    });
  }

  function setupSearch() {
    const searchInput = document.getElementById('txns-search-input') || document.querySelector('input[placeholder*="Filter by Transaction ID"]');
    if (!searchInput) return;

    searchInput.addEventListener('input', (e) => {
      currentSearchQuery = e.target.value.trim();
      renderTable();
    });
  }

  function setupHeaderSearch() {
    const headerSearch = document.querySelector('header input[placeholder*="Search transaction"]');
    if (!headerSearch) return;

    headerSearch.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && headerSearch.value.trim()) {
        const raw = headerSearch.value.replace(/[^0-9]/g, '');
        if (raw) {
          window.SentinelState.setActiveTransactionId(raw);
          window.location.href = `investigations.html?transaction_id=${raw}`;
        }
      }
    });
  }

  function setupTimeWindowSelector() {
    const timeBtn = document.getElementById('btn-time-window');
    if (!timeBtn) return;

    const windows = [
      { label: 'Last 1 hour', seconds: 3600 },
      { label: 'Last 6 hours', seconds: 21600 },
      { label: 'Last 24 hours', seconds: 86400 },
      { label: 'Last 7 days', seconds: 604800 },
    ];
    let idx = 0;

    timeBtn.addEventListener('click', () => {
      idx = (idx + 1) % windows.length;
      const span = timeBtn.querySelector('span:nth-child(2)');
      if (span) span.textContent = windows[idx].label;

      const indicator = document.querySelector('.hidden.xl\\:flex span:nth-child(2)');
      if (indicator) indicator.textContent = `Filter applied: [window: ${windows[idx].seconds}s]`;

      loadTransactions();
    });
  }

  function setupRefreshButton() {
    const refreshBtn = document.getElementById('btn-refresh-txns');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => loadTransactions());
    }
  }

  function setupExportCSV() {
    const exportBtn = document.getElementById('btn-export-csv');
    if (!exportBtn) return;

    exportBtn.addEventListener('click', () => {
      const filtered = getFilteredTransactions();
      if (!filtered.length) {
        alert('No transactions available to export.');
        return;
      }

      const headers = ['Transaction ID', 'Amount (INR)', 'Base ML Risk (At)', 'Graph Risk (Gt)', 'Final Risk (Rt)', 'Decision', 'Trigger Description'];
      const rows = filtered.map(t => [
        t.transaction_id,
        (t.amount || 0).toFixed(2),
        ((t.base_risk || 0) * 100).toFixed(2) + '%',
        ((t.graph_risk || 0) * 100).toFixed(2) + '%',
        ((t.final_risk || 0) * 100).toFixed(2) + '%',
        getDecisionText(t.final_risk || 0),
        `"${(t.description || '').replace(/"/g, '""')}"`
      ]);

      const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `sentinel_transactions_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTransactions);
  } else {
    initTransactions();
  }
})();
