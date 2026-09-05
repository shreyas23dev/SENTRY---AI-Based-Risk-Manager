/**
 * demo-mode.js — Shared Sentinel Demo Mode Component & Controller
 * ================================================================
 * 
 * Provides unified, compact DEMO MODE popover in the top header across
 * all frontend screens (investigations.html, transactions.html, 
 * risk-engine.html, overview.html, index.html).
 *
 * Synchronizes with window.SentinelState for seamless cross-page persistence.
 */

(function () {
  'use strict';

  function getDemoPopoverHtml() {
    return `
<div class="relative inline-block" id="sentinel-demo-wrapper">
  <button id="btn-sentinel-demo" class="flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20 hover:border-primary text-xs font-semibold tracking-wide transition-all shadow-sm cursor-pointer select-none" type="button" title="Select a real held-out test case">
    <span class="material-symbols-outlined text-[15px] text-amber-400">bolt</span>
    <span id="sentinel-demo-label">DEMO MODE</span>
    <span class="material-symbols-outlined text-[14px]">arrow_drop_down</span>
  </button>

  <!-- Popover Dropdown -->
  <div id="sentinel-demo-popover" class="hidden absolute right-0 top-full mt-2 w-80 bg-surface-container-low border border-outline-variant/40 rounded-xl shadow-2xl z-50 overflow-hidden flex-col p-2 gap-1.5">
    <div class="px-3 py-2 border-b border-outline-variant/30 flex flex-col">
      <div class="flex items-center justify-between">
        <span class="font-mono-sm text-mono-sm text-primary font-bold tracking-wider uppercase flex items-center gap-1">
          <span class="material-symbols-outlined text-[16px] text-amber-400">bolt</span> DEMO MODE
        </span>
        <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-secondary/10 text-secondary border border-secondary/30">TEST SPLIT</span>
      </div>
      <span class="text-[11px] text-on-surface-variant mt-0.5">Demo Mode — Real held-out test transaction</span>
    </div>

    <div class="flex flex-col gap-1 max-h-80 overflow-y-auto" id="demo-cases-list">
      <!-- Case 1 -->
      <div class="demo-case-card p-2 rounded-lg bg-surface-container/60 hover:bg-surface-container border border-outline-variant/20 hover:border-primary/50 cursor-pointer transition-all flex flex-col gap-0.5" data-case="1">
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-[10px] font-bold text-primary">CASE 1</span>
          <span class="font-mono-sm text-[10px] text-on-surface-variant font-mono">Txn #3570805</span>
        </div>
        <div class="font-label-md text-xs font-semibold text-on-surface">Agreement → BLOCK</div>
        <div class="font-body-sm text-[11px] text-on-surface-variant leading-tight">High ML risk & documented prior fraud history. Base ML & Slow-Burn analysts agree.</div>
      </div>

      <!-- Case 2 -->
      <div class="demo-case-card p-2 rounded-lg bg-surface-container/60 hover:bg-surface-container border border-outline-variant/20 hover:border-primary/50 cursor-pointer transition-all flex flex-col gap-0.5" data-case="2">
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-[10px] font-bold text-primary">CASE 2</span>
          <span class="font-mono-sm text-[10px] text-on-surface-variant font-mono">Txn #3531382</span>
        </div>
        <div class="font-label-md text-xs font-semibold text-on-surface">Slow-Burn Disagreement</div>
        <div class="font-body-sm text-[11px] text-on-surface-variant leading-tight">Base ML sees normal transaction; Slow-Burn detects temporal velocity surge.</div>
      </div>

      <!-- Case 3 -->
      <div class="demo-case-card p-2 rounded-lg bg-surface-container/60 hover:bg-surface-container border border-outline-variant/20 hover:border-primary/50 cursor-pointer transition-all flex flex-col gap-0.5" data-case="3">
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-[10px] font-bold text-primary">CASE 3</span>
          <span class="font-mono-sm text-[10px] text-on-surface-variant font-mono">Txn #3488970</span>
        </div>
        <div class="font-label-md text-xs font-semibold text-on-surface">Cold Start / Insufficient History</div>
        <div class="font-body-sm text-[11px] text-on-surface-variant leading-tight">Single isolated txn with no prior behavioral history (Pₜ: No Prior History).</div>
      </div>

      <!-- Case 4 -->
      <div class="demo-case-card p-2 rounded-lg bg-surface-container/60 hover:bg-surface-container border border-outline-variant/20 hover:border-primary/50 cursor-pointer transition-all flex flex-col gap-0.5" data-case="4">
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-[10px] font-bold text-primary">CASE 4</span>
          <span class="font-mono-sm text-[10px] text-on-surface-variant font-mono">Txn #3488964</span>
        </div>
        <div class="font-label-md text-xs font-semibold text-on-surface">Clean + Historical Risk Context → ALLOW</div>
        <div class="font-body-sm text-[11px] text-on-surface-variant leading-tight">Low transaction risk; allowed despite historical risk context in relational association.</div>
      </div>

      <!-- Case 5 -->
      <div class="demo-case-card p-2 rounded-lg bg-surface-container/60 hover:bg-surface-container border border-outline-variant/20 hover:border-primary/50 cursor-pointer transition-all flex flex-col gap-0.5" data-case="5">
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-[10px] font-bold text-primary">CASE 5</span>
          <span class="font-mono-sm text-[10px] text-on-surface-variant font-mono">Txn #3489068</span>
        </div>
        <div class="font-label-md text-xs font-semibold text-on-surface">Step-Up Auth → VERIFY</div>
        <div class="font-body-sm text-[11px] text-on-surface-variant leading-tight">Elevated risk with ₹150 amount. Cost model triggers step-up OTP verification to minimize expected loss.</div>
      </div>

      <!-- Case 6 -->
      <div class="demo-case-card p-2 rounded-lg bg-surface-container/60 hover:bg-surface-container border border-outline-variant/20 hover:border-primary/50 cursor-pointer transition-all flex flex-col gap-0.5" data-case="6">
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-[10px] font-bold text-primary">CASE 6</span>
          <span class="font-mono-sm text-[10px] text-on-surface-variant font-mono">Txn #3512832</span>
        </div>
        <div class="font-label-md text-xs font-semibold text-on-surface">Behavioral Velocity → THROTTLE</div>
        <div class="font-body-sm text-[11px] text-on-surface-variant leading-tight">Low ₹34 amount with slow-burn device multiplexing. Cost engine selectively rate-limits velocity.</div>
      </div>
    </div>

    <div class="pt-1.5 border-t border-outline-variant/30 flex items-center justify-between px-2">
      <span class="text-[10px] text-outline font-mono">REAL HELD-OUT TEST DATA</span>
      <button id="btn-sentinel-exit-demo" class="text-[11px] text-on-surface-variant hover:text-tertiary-container hover:bg-tertiary-container/10 px-2 py-1 rounded transition-colors font-medium cursor-pointer" type="button">
        ✕ Exit Demo Mode
      </button>
    </div>
  </div>
</div>
`;
  }

  function mountDemoMode() {
    // If not already in DOM, mount it into header
    if (!document.getElementById('sentinel-demo-wrapper')) {
      const headerRight = document.querySelector('header div.flex.items-center.gap-space-md.shrink-0');
      if (headerRight) {
        const notifBtn = headerRight.querySelector('button');
        const container = document.createElement('div');
        container.innerHTML = getDemoPopoverHtml();
        const demoEl = container.firstElementChild;
        if (notifBtn) {
          headerRight.insertBefore(demoEl, notifBtn);
        } else {
          headerRight.prepend(demoEl);
        }
      }
    }

    bindEvents();
    syncUIWithState();
  }

  function bindEvents() {
    const btn = document.getElementById('btn-sentinel-demo');
    const popover = document.getElementById('sentinel-demo-popover');
    const exitBtn = document.getElementById('btn-sentinel-exit-demo');

    if (btn && popover) {
      btn.onclick = (e) => {
        e.stopPropagation();
        const isHidden = popover.classList.contains('hidden');
        if (isHidden) {
          popover.classList.remove('hidden');
          popover.classList.add('flex');
        } else {
          popover.classList.add('hidden');
          popover.classList.remove('flex');
        }
      };
    }

    // Dismiss on click outside
    window.addEventListener('click', (e) => {
      if (popover && !popover.classList.contains('hidden')) {
        const wrapper = document.getElementById('sentinel-demo-wrapper');
        if (wrapper && !wrapper.contains(e.target)) {
          popover.classList.add('hidden');
          popover.classList.remove('flex');
        }
      }
    });

    // Case selection cards
    document.querySelectorAll('.demo-case-card').forEach(card => {
      card.onclick = (e) => {
        e.stopPropagation();
        const caseNum = parseInt(card.dataset.case, 10);
        if (caseNum && window.SentinelState) {
          if (popover) {
            popover.classList.add('hidden');
            popover.classList.remove('flex');
          }
          window.SentinelState.selectDemoCase(caseNum);
        }
      };
    });

    // Exit demo button
    if (exitBtn) {
      exitBtn.onclick = (e) => {
        e.stopPropagation();
        if (popover) {
          popover.classList.add('hidden');
          popover.classList.remove('flex');
        }
        if (window.SentinelState) {
          window.SentinelState.clearActiveDemoCase();
        }
      };
    }

    // Header search integration: typing a custom transaction ID disengages Demo Mode
    const headerSearch = document.querySelector('header input[placeholder*="Search transaction"]');
    if (headerSearch) {
      headerSearch.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && headerSearch.value.trim()) {
          const rawId = parseInt(headerSearch.value.replace(/[^0-9]/g, ''), 10);
          if (rawId) {
            const demoCases = window.SentinelState ? window.SentinelState.getDemoCases() : [];
            const match = demoCases.find(c => c.txnId === rawId);
            if (match) {
              window.SentinelState.selectDemoCase(match.caseNum);
            } else {
              if (window.SentinelState) {
                window.SentinelState.clearActiveDemoCase();
                window.SentinelState.setActiveTransactionId(rawId);
              }
              const isInvestigations = window.location.pathname.includes('investigations.html');
              if (!isInvestigations) {
                window.location.href = `investigations.html?transaction_id=${rawId}`;
              }
            }
          }
        }
      });
    }

    // State change listeners
    window.addEventListener('sentinel:demoCaseChange', () => syncUIWithState());
    window.addEventListener('sentinel:transactionChange', () => syncUIWithState());
  }

  function syncUIWithState() {
    if (!window.SentinelState) return;

    const activeCase = window.SentinelState.getActiveDemoCase();
    const btn = document.getElementById('btn-sentinel-demo');
    const label = document.getElementById('sentinel-demo-label');

    if (btn && label) {
      if (activeCase && activeCase >= 1 && activeCase <= 6) {
        label.textContent = `DEMO MODE · CASE ${activeCase}`;
        btn.className = 'flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/20 border border-amber-400 text-amber-300 hover:bg-amber-500/30 hover:border-amber-300 text-xs font-semibold tracking-wide transition-all shadow-md shadow-amber-500/10 cursor-pointer select-none';
      } else {
        label.textContent = 'DEMO MODE';
        btn.className = 'flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/30 text-primary hover:bg-primary/20 hover:border-primary text-xs font-semibold tracking-wide transition-all shadow-sm cursor-pointer select-none';
      }
    }

    // Highlight active case in popover list
    document.querySelectorAll('.demo-case-card').forEach(card => {
      const cNum = parseInt(card.dataset.case, 10);
      if (cNum === activeCase) {
        card.classList.add('border-primary', 'bg-primary/20', 'ring-1', 'ring-primary/40');
        card.classList.remove('border-outline-variant/20', 'bg-surface-container/60');
      } else {
        card.classList.remove('border-primary', 'bg-primary/20', 'ring-1', 'ring-primary/40');
        card.classList.add('border-outline-variant/20', 'bg-surface-container/60');
      }
    });
  }

  function updateActiveNavRail(forcePath) {
    const nav = document.querySelector('nav[data-active-classes]');
    if (!nav) return;

    let activePath = forcePath;
    if (!activePath) {
      const pathname = window.location.pathname.toLowerCase();
      const hash = window.location.hash.toLowerCase();

      if (pathname.includes('investigations.html')) {
        activePath = 'investigations';
      } else if (pathname.includes('risk-engine.html')) {
        activePath = (hash === '#analytics-section') ? 'analytics' : 'risk-engine';
      } else if (pathname.includes('transactions.html')) {
        activePath = 'transactions';
      } else {
        // index.html, overview.html, or /
        activePath = 'overview';
      }
    }

    const activeClasses = ['bg-surface-container', 'text-primary', 'font-medium', 'border-l-2', 'border-primary-container'];
    const inactiveClasses = ['font-body-sm', 'text-body-sm', 'text-on-surface-variant'];

    nav.querySelectorAll('a[data-path]').forEach(link => {
      const path = link.getAttribute('data-path');
      if (path === activePath) {
        link.setAttribute('aria-current', 'page');
        activeClasses.forEach(cls => link.classList.add(cls));
        inactiveClasses.forEach(cls => link.classList.remove(cls));
      } else {
        link.removeAttribute('aria-current');
        activeClasses.forEach(cls => link.classList.remove(cls));
        inactiveClasses.forEach(cls => link.classList.add(cls));
      }
    });
  }

  function initActiveNavRail() {
    updateActiveNavRail();

    window.addEventListener('hashchange', () => {
      updateActiveNavRail();
    });

    window.addEventListener('popstate', () => {
      updateActiveNavRail();
    });

    // In-page anchor click handling for responsive UI & smooth scroll
    const nav = document.querySelector('nav[data-active-classes]');
    if (!nav) return;

    nav.querySelectorAll('a[data-path]').forEach(link => {
      link.addEventListener('click', (e) => {
        const path = link.getAttribute('data-path');
        const pathname = window.location.pathname.toLowerCase();

        if (pathname.includes('investigations.html')) {
          if (path === 'investigations') {
            e.preventDefault();
            window.history.pushState(null, '', 'investigations.html');
            window.scrollTo({ top: 0, behavior: 'smooth' });
            updateActiveNavRail('investigations');
          }
        } else if (pathname.includes('risk-engine.html')) {
          if (path === 'risk-engine') {
            e.preventDefault();
            window.history.pushState(null, '', 'risk-engine.html');
            window.scrollTo({ top: 0, behavior: 'smooth' });
            updateActiveNavRail('risk-engine');
          } else if (path === 'analytics') {
            e.preventDefault();
            window.history.pushState(null, '', 'risk-engine.html#analytics-section');
            const target = document.getElementById('analytics-section');
            if (target) {
              target.scrollIntoView({ behavior: 'smooth' });
            }
            updateActiveNavRail('analytics');
          }
        }
      });
    });
  }

  function mountDemoMode() {
    // If not already in DOM, mount it into header
    if (!document.getElementById('sentinel-demo-wrapper')) {
      const headerRight = document.querySelector('header div.flex.items-center.gap-space-md.shrink-0');
      if (headerRight) {
        const notifBtn = headerRight.querySelector('button');
        const container = document.createElement('div');
        container.innerHTML = getDemoPopoverHtml();
        const demoEl = container.firstElementChild;
        if (notifBtn) {
          headerRight.insertBefore(demoEl, notifBtn);
        } else {
          headerRight.prepend(demoEl);
        }
      }
    }

    bindEvents();
    syncUIWithState();
    initActiveNavRail();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountDemoMode);
  } else {
    mountDemoMode();
  }
})();