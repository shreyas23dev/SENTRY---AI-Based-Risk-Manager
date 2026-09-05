/**
 * investigations.js — Sentinel AI Risk Investigation & Knowledge Graph Controller (Hardened Phase 7)
 * ==================================================================================================
 * 
 * Drives frontend/investigations.html:
 *   - Real risk score breakdown (A_t, G_t, R_t, Decision, Cost)
 *   - Real 1-hop / 2-hop D3 Force-Directed Knowledge Graph
 *   - Dynamic Node Inspector Drawer with slide-over controls and properties inspection
 *   - Ranked, provenance-backed Evidence items with clickable citations
 *   - Grounded GraphRAG AI Investigator reports with evidence citations
 *   - Interactive SOC Copilot Q&A terminal with instant Suggested Question Chips
 *   - JSON evidence export & Technical Specs drawer
 */

(function () {
  'use strict';

  let currentTxnId = null;
  let graphRenderer = null;
  let currentMaxHops = 2;
  let evidenceCache = [];
  let currentRiskRecord = null;
  let selectedNodeData = null;

  async function initInvestigation() {
    // 1. Get active transaction ID from state or URL query parameter
    const urlParams = new URLSearchParams(window.location.search);
    const urlTxnId = urlParams.get('transaction_id');
    if (urlTxnId) {
      window.SentinelState.setActiveTransactionId(urlTxnId);
    }
    currentTxnId = window.SentinelState.getActiveTransactionId();

    // 2. Initialize D3 Knowledge Graph Visualizer
    const graphContainer = '#knowledge-graph-canvas';
    if (document.querySelector(graphContainer)) {
      graphRenderer = new window.KnowledgeGraphRenderer(graphContainer, {
        onNodeSelected: handleNodeSelected,
      });
      graphRenderer.init();
    }

    // 3. Setup UI Controls & Event Listeners
    setupGraphControls();
    setupInvestigatorActions();
    setupCopilotQA();
    setupQuestionChips();
    setupTechSpecsToggle();
    setupNodeDrawer();
    setupExportEvidence();

    // 4. Load All Real Backend Data for Active Transaction
    await loadTransactionData(currentTxnId);
  }

  async function loadTransactionData(txnId) {
    if (!txnId) return;

    // Instant pre-render if transaction matches a verified demo case
    const demoCases = window.SentinelState && window.SentinelState.getDemoCases ? window.SentinelState.getDemoCases() : [];
    const demoMatch = demoCases.find(c => c.txnId === txnId);
    if (demoMatch && demoMatch.final_risk !== undefined) {
      renderRiskSummary({
        transaction_id: txnId,
        base_risk: demoMatch.base_risk,
        graph_risk: demoMatch.graph_risk,
        final_risk: demoMatch.final_risk,
        amount: demoMatch.amount,
        is_fraud: demoMatch.isFraud,
        action: demoMatch.action,
      });
      if (demoMatch.council) {
        renderCouncilCase({
          transaction_analyst: {
            assessment: demoMatch.council.t_tier,
            risk: parseFloat(demoMatch.council.t_score.replace('%', '')) / 100,
          },
          slow_burn_analyst: {
            assessment: demoMatch.council.sb_tier,
            risk: parseFloat(demoMatch.council.sb_score.replace('%', '')) / 100,
          },
          council: {
            status: demoMatch.council.status,
            reasoning: demoMatch.council.officer,
          },
          graph_context: {
            prior_entity_txns: parseInt(demoMatch.council.history.split(' ')[0], 10) || 0,
            prior_entity_frauds: parseInt(demoMatch.council.history.split('· ')[1], 10) || 0,
          },
          officer_synthesis: demoMatch.council.officer,
        });
      }
    } else {
      // Reset UI to prevent stale state from previously loaded case
      setTextContent('#header-txn-title', `Transaction #${txnId}`);
      setTextContent('#header-fraud-badge', 'Verifying...');
      setTextContent('#score-final-risk', '--%');
      setTextContent('#score-risk-tier', '--');
      setTextContent('#score-base-ml', '--%');
      setTextContent('#score-graph-context', '--%');
      setTextContent('#score-graph-impact', '--%');
      setTextContent('#score-graph-impact-sub', 'Computing...');
      setTextContent('#score-decision', 'ANALYZING...');
      setTextContent('#score-expected-loss', 'Estimating...');
      setTextContent('#council-status-pill', 'EVALUATING');
      setTextContent('#council-banner-text', 'Evaluating multi-analyst consensus...');
      const councilNarrative = document.getElementById('council-officer-narrative');
      if (councilNarrative) councilNarrative.innerHTML = '<span class="text-outline italic text-xs">Gathering council deliberations...</span>';
    }
    const copilotHistory = document.getElementById('copilot-history-container');
    if (copilotHistory) copilotHistory.innerHTML = '';
    const reportBox = document.getElementById('ai-executive-summary');
    if (reportBox) reportBox.innerHTML = '<span class="text-outline italic">Analyzing transaction payload and 2-hop relational knowledge graph...</span>';
    const findingsList = document.getElementById('ai-key-findings');
    if (findingsList) findingsList.innerHTML = '<li class="text-body-sm text-outline italic">Synthesizing forensic findings...</li>';
    const drawer = document.getElementById('node-drawer');
    if (drawer) drawer.classList.add('translate-x-full');

    showLoadingState(true);

    try {
      // Parallel fetch: Risk Summary, Knowledge Graph, Evidence
      const [riskRecord, graphData, evidenceItems] = await Promise.all([
        window.SentinelAPI.getRiskSummary(txnId).catch(err => {
          console.warn('Risk summary fetch error:', err);
          return null;
        }),
        window.SentinelAPI.getGraph(txnId, currentMaxHops).catch(err => {
          console.warn('Graph fetch error:', err);
          return null;
        }),
        window.SentinelAPI.getEvidence(txnId).catch(err => {
          console.warn('Evidence fetch error:', err);
          return [];
        }),
      ]);

      currentRiskRecord = riskRecord;
      evidenceCache = evidenceItems || [];

      // 1. Render Risk Summary
      if (riskRecord) {
        renderRiskSummary(riskRecord);
      }

      // 2. Render Knowledge Graph
      if (graphRenderer && graphData) {
        graphRenderer.render(graphData);
      }

      // 3. Render Evidence List
      renderEvidenceDrawer(evidenceCache);

      // 4. Auto-trigger initial grounded investigation report
      runAIInvestigation(txnId);

    } catch (err) {
      console.error('Error loading investigation data:', err);
    } finally {
      showLoadingState(false);
    }
  }

  function renderRiskSummary(risk) {
    const tid = risk.transaction_id;
    const a_t = (risk.base_risk * 100).toFixed(2);
    const g_t = (risk.graph_risk * 100).toFixed(2);
    const r_t = (risk.final_risk * 100).toFixed(2);
    const finalRiskVal = risk.final_risk || 0;
    const delta = (risk.final_risk - risk.base_risk) * 100;
    const amt = (risk.amount || 0).toFixed(2);

    // Determine decision and risk tier
    let decision = risk.action;
    if (!decision) {
      if (finalRiskVal >= 0.70) decision = 'BLOCK';
      else if (finalRiskVal >= 0.35) decision = 'THROTTLE';
      else if (finalRiskVal >= 0.15) decision = 'VERIFY';
      else decision = 'ALLOW';
    }

    let tier = 'LOW';
    let badgeStyle = 'bg-secondary/20 text-secondary border-secondary/30';
    let tierColor = 'text-secondary';
    let gaugePathColor = 'text-secondary';
    let gaugeIcon = 'check_circle';
    let gaugeIconColor = 'text-secondary';
    let thresholdText = 'Policy Tier: Fast-Path ALLOW (<15.00%)';
    let isoCode = 'RC-00 (Approved / Legitimate)';
    let isoColor = 'text-secondary';

    if (decision === 'BLOCK' || finalRiskVal >= 0.70) {
      tier = 'CRITICAL';
      badgeStyle = 'bg-tertiary-container/20 text-tertiary-fixed border-tertiary/30';
      tierColor = 'text-tertiary-fixed';
      gaugePathColor = 'text-tertiary-container';
      gaugeIcon = 'warning';
      gaugeIconColor = 'text-tertiary-fixed';
      thresholdText = 'Loss threshold: >50.00%';
      isoCode = 'RC-59 (Suspected Fraud - BLOCK)';
      isoColor = 'text-tertiary-fixed';
    } else if (decision === 'THROTTLE' || finalRiskVal >= 0.35) {
      tier = 'HIGH';
      badgeStyle = 'bg-tertiary/20 text-tertiary border-tertiary/30';
      tierColor = 'text-tertiary';
      gaugePathColor = 'text-tertiary';
      gaugeIcon = 'error_outline';
      gaugeIconColor = 'text-tertiary';
      thresholdText = 'Policy Tier: Rate Limit (35.00% - 70.00%)';
      isoCode = 'RC-44 (Velocity Anomaly - THROTTLE)';
      isoColor = 'text-tertiary';
    } else if (decision === 'VERIFY' || finalRiskVal >= 0.15) {
      tier = 'MEDIUM';
      badgeStyle = 'bg-primary-container/20 text-primary border-primary/30';
      tierColor = 'text-primary';
      gaugePathColor = 'text-primary';
      gaugeIcon = 'verified_user';
      gaugeIconColor = 'text-primary';
      thresholdText = 'Policy Tier: Step-up 2FA (15.00% - 35.00%)';
      isoCode = 'RC-05 (Step-up Verification Required)';
      isoColor = 'text-primary';
    }

    // Update Header
    setTextContent('#header-txn-title', `Transaction #${tid}`);
    setTextContent('#header-amount', `₹${amt}`);
    const badgeEl = document.querySelector('#header-decision-badge');
    if (badgeEl) {
      badgeEl.textContent = decision;
      badgeEl.className = `px-2 py-0.5 rounded-sm font-label-sm text-label-sm uppercase tracking-wider font-semibold border ${badgeStyle}`;
    }

    // Resolve dataset label or ground truth status
    // Check risk object first, then SentinelState DEMO_CASES
    let isFraud = risk.is_fraud;
    if (isFraud === undefined && window.SentinelState && window.SentinelState.getDemoCases) {
      const demoMatch = window.SentinelState.getDemoCases().find(c => c.txnId === tid);
      if (demoMatch && demoMatch.isFraud !== undefined) {
        isFraud = demoMatch.isFraud;
      }
    }

    const fraudBadgeEl = document.querySelector('#header-fraud-badge');
    if (fraudBadgeEl) {
      if (isFraud === 1 || isFraud === true) {
        if (decision === 'ALLOW') {
          // Slow-burn evasion case: allowed in real-time gateway, charged back post-settlement
          fraudBadgeEl.textContent = 'Post-Settlement Chargeback';
          fraudBadgeEl.className = 'px-2 py-0.5 rounded-sm bg-amber-500/10 text-amber-400 border border-amber-500/30 font-mono-sm text-mono-sm font-medium';
          fraudBadgeEl.title = 'Allowed at checkout by real-time gateway (clean instantaneous features); cardholder charged back post-settlement.';
        } else {
          fraudBadgeEl.textContent = 'Confirmed Fraud';
          fraudBadgeEl.className = 'px-2 py-0.5 rounded-sm bg-tertiary-container/20 text-tertiary-fixed border border-tertiary/30 font-mono-sm text-mono-sm font-medium';
        }
      } else if (isFraud === 0 || isFraud === false) {
        if (risk.graph_risk > 0.15 || delta > 0.05) {
          fraudBadgeEl.textContent = 'Historical Risk Context';
          fraudBadgeEl.className = 'px-2 py-0.5 rounded-sm bg-surface-container-high text-on-surface-variant font-mono-sm text-mono-sm';
          fraudBadgeEl.title = 'Legitimate customer with historical risk associations in graph.';
        } else {
          fraudBadgeEl.textContent = 'Legitimate Activity';
          fraudBadgeEl.className = 'px-2 py-0.5 rounded-sm bg-secondary/20 text-secondary border border-secondary/30 font-mono-sm text-mono-sm font-medium';
        }
      } else {
        fraudBadgeEl.textContent = decision === 'BLOCK' ? 'Suspected Risk' : 'Standard Evaluation';
        fraudBadgeEl.className = 'px-2 py-0.5 rounded-sm bg-surface-container-high text-on-surface-variant font-mono-sm text-mono-sm';
      }
    }

    // Update Dominant Gauge & Scores
    setTextContent('#score-final-risk', `${r_t}%`);
    const tierEl = document.querySelector('#score-risk-tier');
    if (tierEl) {
      tierEl.textContent = tier;
      tierEl.className = `font-mono-sm text-mono-sm ${tierColor} font-semibold`;
    }
    setTextContent('#score-risk-threshold', thresholdText);

    const gaugeIconEl = document.querySelector('#gauge-status-icon');
    if (gaugeIconEl) {
      gaugeIconEl.textContent = gaugeIcon;
      gaugeIconEl.className = `material-symbols-outlined ${gaugeIconColor} absolute text-[20px]`;
    }

    const gaugePath = document.querySelector('#gauge-progress-path');
    if (gaugePath) {
      gaugePath.setAttribute('stroke-dasharray', `${Math.min(100, Math.max(0, r_t))}, 100`);
      gaugePath.setAttribute('class', gaugePathColor);
    }

    setTextContent('#score-base-ml', `${a_t}%`);
    setTextContent('#score-graph-context', `${g_t}%`);
    setTextContent('#score-graph-impact', `${delta >= 0 ? '+' : ''}${delta.toFixed(2)}%`);

    // Graph Impact Subtext
    const impactSubEl = document.querySelector('#score-graph-impact-sub');
    const impactValEl = document.querySelector('#score-graph-impact');
    if (impactSubEl) {
      if (decision === 'BLOCK' && delta > 0.01) {
        impactSubEl.textContent = 'Pushed to BLOCK';
        impactSubEl.className = 'font-mono-sm text-mono-sm text-tertiary-fixed';
        if (impactValEl) impactValEl.className = 'font-headline-md text-headline-md font-semibold text-tertiary-fixed mt-1 tabular-nums';
      } else if (decision === 'VERIFY' && delta > 0.01) {
        impactSubEl.textContent = 'Elevated to VERIFY';
        impactSubEl.className = 'font-mono-sm text-mono-sm text-primary';
        if (impactValEl) impactValEl.className = 'font-headline-md text-headline-md font-semibold text-primary mt-1 tabular-nums';
      } else if (decision === 'THROTTLE' && delta > 0.01) {
        impactSubEl.textContent = 'Elevated to THROTTLE';
        impactSubEl.className = 'font-mono-sm text-mono-sm text-tertiary';
        if (impactValEl) impactValEl.className = 'font-headline-md text-headline-md font-semibold text-tertiary mt-1 tabular-nums';
      } else if (delta > 0.01) {
        impactSubEl.textContent = 'Relational Context';
        impactSubEl.className = 'font-mono-sm text-mono-sm text-outline';
        if (impactValEl) impactValEl.className = 'font-headline-md text-headline-md font-semibold text-secondary-fixed-dim mt-1 tabular-nums';
      } else if (delta < -0.01) {
        impactSubEl.textContent = 'Contextual Dampening';
        impactSubEl.className = 'font-mono-sm text-mono-sm text-secondary';
        if (impactValEl) impactValEl.className = 'font-headline-md text-headline-md font-semibold text-secondary mt-1 tabular-nums';
      } else {
        impactSubEl.textContent = 'Neutral / Isolated';
        impactSubEl.className = 'font-mono-sm text-mono-sm text-outline';
        if (impactValEl) impactValEl.className = 'font-headline-md text-headline-md font-semibold text-outline mt-1 tabular-nums';
      }
    }

    // Decision Card
    const decisionEl = document.querySelector('#score-decision');
    if (decisionEl) {
      decisionEl.textContent = decision;
      decisionEl.className = `font-headline-md text-headline-md font-bold uppercase mt-1 ${tierColor}`;
    }

    const lossEl = document.querySelector('#score-expected-loss');
    if (lossEl) {
      if (decision === 'BLOCK') {
        lossEl.textContent = `Prevented loss: ₹${amt}`;
      } else {
        const expectedLoss = (finalRiskVal * (risk.amount || 0)).toFixed(2);
        lossEl.textContent = `Expected loss: ₹${expectedLoss}`;
      }
    }

    // Telemetry Specs
    const isoCodeEl = document.querySelector('#tech-iso-code');
    if (isoCodeEl) {
      isoCodeEl.textContent = isoCode;
      isoCodeEl.className = `font-mono-md text-mono-md font-semibold ${isoColor}`;
    }
  }

  function renderEvidenceDrawer(evidenceItems) {
    const container = document.getElementById('evidence-items-container');
    const badgeCount = document.getElementById('evidence-count-badge');
    if (badgeCount) badgeCount.textContent = `${evidenceItems.length} Verified`;

    if (!container) return;
    container.innerHTML = '';

    if (!evidenceItems || evidenceItems.length === 0) {
      container.innerHTML = '<p class="text-outline text-body-sm py-4">No verified graph anomaly evidence items for this transaction.</p>';
      return;
    }

    evidenceItems.forEach((item, idx) => {
      const card = document.createElement('div');
      card.className = 'p-space-sm rounded-lg bg-surface-container hover:bg-surface-container-high/80 transition-colors flex flex-col gap-1 border border-outline-variant/20 cursor-pointer';
      
      const isHighRisk = (item.risk_weight || item.weight || 0) >= 0.5 || (item.evidence_type || '').includes('FRAUD');
      const weightColor = isHighRisk ? 'text-tertiary-fixed' : 'text-secondary';
      const eid = item.evidence_id || `E${idx + 1}`;
      const title = item.title || item.evidence_type || `Evidence ${eid}`;

      card.innerHTML = `
        <div class="flex items-center justify-between">
          <span class="font-mono-sm text-mono-sm font-bold tracking-wider ${weightColor}">
            ${title.toUpperCase()}
          </span>
          <button class="evidence-pill px-1.5 py-0.5 rounded bg-surface-container-highest text-primary font-mono-sm text-mono-sm hover:bg-primary hover:text-on-primary transition-colors cursor-pointer" type="button">
            [${eid}]
          </button>
        </div>
        <p class="font-body-sm text-body-sm text-on-surface mt-0.5 leading-snug">
          ${item.description}
        </p>
        ${item.relationship_path && item.relationship_path.length ? `
          <div class="mt-1 flex items-center gap-1 font-mono-sm text-[10px] text-outline truncate">
            <span>Path:</span>
            <span class="text-primary font-semibold">${item.relationship_path.join(' → ')}</span>
          </div>
        ` : ''}
      `;

      // Clicking evidence card highlights corresponding entity in knowledge graph
      card.onclick = () => {
        if (graphRenderer && item.relationship_path && item.relationship_path.length) {
          const targetNodeId = item.relationship_path[item.relationship_path.length - 1];
          if (graphRenderer.selectNodeById) {
            graphRenderer.selectNodeById(targetNodeId);
          }
        }
        openNodeDrawerForEvidence(item);
      };

      container.appendChild(card);
    });
  }

  function openNodeDrawerForEvidence(evidence) {
    const drawer = document.getElementById('node-drawer');
    if (!drawer) return;

    drawer.classList.remove('translate-x-full');

    setTextContent('#inspector-node-type', 'EVIDENCE FINDING');
    setTextContent('#inspector-node-title', evidence.evidence_id || evidence.title || 'Verified Finding');
    setTextContent('#inspector-node-risk', `Weight: ${Math.round((evidence.risk_weight || evidence.weight || 0.5) * 100)}%`);

    const propsContainer = document.getElementById('inspector-properties-list');
    if (propsContainer) {
      propsContainer.innerHTML = `
        <div class="p-space-xs bg-surface-container-lowest/80 rounded font-mono-sm text-mono-sm text-on-surface-variant flex flex-col gap-1">
          <div><strong class="text-on-surface">Type:</strong> ${evidence.evidence_type || 'Graph Anomaly'}</div>
          <div><strong class="text-on-surface">Summary:</strong> ${evidence.description || 'Verified forensic finding.'}</div>
          ${evidence.relationship_path ? `<div><strong class="text-on-surface">Graph Chain:</strong> ${evidence.relationship_path.join(' → ')}</div>` : ''}
        </div>
      `;
    }
  }

  async function runAIInvestigation(txnId) {
    const reportBox = document.getElementById('ai-executive-summary');
    const btn = document.getElementById('btn-rerun');

    try {
      if (btn) btn.disabled = true;
      if (reportBox) reportBox.innerHTML = '<span class="text-outline italic">Analyzing transaction payload and 2-hop relational knowledge graph...</span>';

      // Parallel fetch: Investigation report + Multi-Analyst Council (including Slow-Burn)
      const [report, councilCase] = await Promise.all([
        window.SentinelAPI.investigateTransaction(txnId, 'balanced').catch(err => {
          console.warn('Investigate error:', err);
          return null;
        }),
        window.SentinelAPI.getCouncil(txnId, 'balanced').catch(err => {
          console.warn('Council error:', err);
          return null;
        }),
      ]);

      if (report) {
        renderInvestigationReport(report);
      }
      if (councilCase) {
        renderCouncilCase(councilCase);
      }
    } catch (err) {
      console.error('AI Investigation error:', err);
      if (reportBox) {
        reportBox.innerHTML = `<span class="text-on-surface-variant">Automated analysis generated using verified graph context. (Investigator notice: ${err.message})</span>`;
      }
    } finally {
      if (btn) btn.disabled = false;
    }
  }

  function renderCouncilCase(caseFile) {
    if (!caseFile) return;

    const tAnalyst = caseFile.transaction_analyst || {};
    const sbAnalyst = caseFile.slow_burn_analyst || {};
    const council = caseFile.council || {};
    const graphCtx = caseFile.graph_context || {};

    // 1. Transaction Risk Analyst
    const tTier = tAnalyst.assessment || 'LOW';
    const tScore = tAnalyst.risk !== undefined ? (tAnalyst.risk * 100).toFixed(1) + '%' : '--%';
    setTextContent('#council-t-tier', tTier);
    setTextContent('#council-t-score', `Aₜ: ${tScore}`);
    const tTierEl = document.querySelector('#council-t-tier');
    if (tTierEl) {
      if (tTier === 'HIGH' || tTier === 'CRITICAL') tTierEl.className = 'font-mono-sm text-xs font-bold text-tertiary-fixed';
      else if (tTier === 'MEDIUM' || tTier === 'ELEVATED') tTierEl.className = 'font-mono-sm text-xs font-bold text-tertiary';
      else tTierEl.className = 'font-mono-sm text-xs font-bold text-secondary';
    }

    // 2. Slow-Burn Analyst
    const sbTier = sbAnalyst.assessment || sbAnalyst.state || 'CLEAN';
    const sbScore = sbAnalyst.risk !== undefined ? (sbAnalyst.risk * 100).toFixed(1) + '%' : '--%';
    setTextContent('#council-sb-tier', sbTier);
    setTextContent('#council-sb-score', `Pₜ: ${sbScore}`);
    const sbTierEl = document.querySelector('#council-sb-tier');
    if (sbTierEl) {
      if (sbTier === 'CRITICAL' || sbTier === 'HIGH') sbTierEl.className = 'font-mono-sm text-xs font-bold text-tertiary-fixed';
      else if (sbTier === 'ELEVATED' || sbTier === 'MEDIUM') sbTierEl.className = 'font-mono-sm text-xs font-bold text-tertiary';
      else sbTierEl.className = 'font-mono-sm text-xs font-bold text-secondary';
    }

    // 3. Council Status Pill & Banner
    const isDisagreement = council.status === 'DISAGREEMENT' || (tTier !== sbTier && (tTier === 'LOW' || sbTier === 'CRITICAL'));
    const statusPill = document.querySelector('#council-status-pill');
    const bannerEl = document.querySelector('#council-banner');
    const bannerIcon = document.querySelector('#council-banner-icon');
    const bannerText = document.querySelector('#council-banner-text');

    if (isDisagreement) {
      if (statusPill) {
        statusPill.textContent = 'DISAGREEMENT';
        statusPill.className = 'px-2 py-0.5 rounded font-mono-sm text-[11px] font-bold tracking-wider uppercase bg-amber-500/20 text-amber-300 border border-amber-500/40';
      }
      if (bannerEl) {
        bannerEl.className = 'mt-space-sm p-space-xs px-space-sm rounded bg-amber-500/10 border border-amber-500/30 flex items-center justify-center gap-1.5 font-label-sm text-xs text-amber-200 text-center';
      }
      if (bannerIcon) {
        bannerIcon.textContent = 'warning';
        bannerIcon.className = 'material-symbols-outlined text-[15px] text-amber-400';
      }
      if (bannerText) {
        bannerText.textContent = '⚠️ ANALYSTS DISAGREE (DISAGREEMENT)';
      }
    } else {
      if (statusPill) {
        statusPill.textContent = 'AGREEMENT';
        statusPill.className = 'px-2 py-0.5 rounded font-mono-sm text-[11px] font-bold tracking-wider uppercase bg-secondary/20 text-secondary border border-secondary/30';
      }
      if (bannerEl) {
        bannerEl.className = 'mt-space-sm p-space-xs px-space-sm rounded bg-secondary/10 border border-secondary/30 flex items-center justify-center gap-1.5 font-label-sm text-xs text-secondary text-center';
      }
      if (bannerIcon) {
        bannerIcon.textContent = 'check_circle';
        bannerIcon.className = 'material-symbols-outlined text-[15px] text-secondary';
      }
      if (bannerText) {
        bannerText.textContent = '✓ ANALYSTS CONCUR (AGREEMENT)';
      }
    }

    // 4. Temporal History Counts
    const historyStatEl = document.querySelector('#council-history-stat');
    if (historyStatEl && graphCtx) {
      const pTxns = graphCtx.prior_entity_txns || 0;
      const pFrauds = graphCtx.prior_entity_frauds || 0;
      historyStatEl.textContent = `${pTxns} prior txns · ${pFrauds} prior frauds`;
    }

    // 5. AI Risk Officer Synthesis
    const narrativeEl = document.querySelector('#council-officer-narrative');
    if (narrativeEl) {
      const narrative = caseFile.officer_synthesis || council.reasoning || council.summary || 'Deliberation completed.';
      narrativeEl.innerHTML = formatMarkdown(narrative);
      attachCitationListeners(narrativeEl);
    }
  }

  function formatMarkdown(rawText) {
    if (!rawText) return '';
    let parsedHtml = '';
    if (typeof marked !== 'undefined' && marked.parse) {
      parsedHtml = marked.parse(rawText);
    } else {
      // Fallback simple parser if marked is loading
      parsedHtml = rawText
        .replace(/^### (.*$)/gim, '<h3 class="font-headline-sm text-on-surface font-semibold mt-2 mb-1">$1</h3>')
        .replace(/^## (.*$)/gim, '<h2 class="font-headline-md text-on-surface font-semibold mt-3 mb-1">$1</h2>')
        .replace(/^# (.*$)/gim, '<h1 class="font-headline-lg text-on-surface font-bold mt-3 mb-2">$1</h1>')
        .replace(/\*\*(.*?)\*\*/gim, '<strong class="text-on-surface font-semibold">$1</strong>')
        .replace(/\*(.*?)\*/gim, '<em class="italic">$1</em>')
        .replace(/\n\n/gim, '</p><p class="mb-2">')
        .replace(/\n/gim, '<br/>');
      parsedHtml = `<p>${parsedHtml}</p>`;
    }

    // Convert evidence and engine citations [E1], [E2], [RISK_ENGINE], [ENGINE], [COST_MODEL] into styled interactive badges
    parsedHtml = parsedHtml.replace(/\[(E\d+|RISK_ENGINE|ENGINE|COST_MODEL|GROUNDING|GRAPH_[A-Z0-9_]+)\]/g, (match, tag) => {
      return `<button type="button" class="citation-tag inline-flex items-center px-1.5 py-0.2 mx-0.5 rounded bg-surface-container-high text-primary hover:bg-primary hover:text-on-primary font-mono-sm text-[11px] font-semibold transition-colors cursor-pointer" data-citation="${tag}">[${tag}]</button>`;
    });

    return parsedHtml;
  }

  function attachCitationListeners(containerEl) {
    if (!containerEl) return;
    const buttons = containerEl.querySelectorAll('.citation-tag');
    buttons.forEach(btn => {
      btn.onclick = (e) => {
        e.stopPropagation();
        const tag = btn.dataset.citation;
        if (!tag) return;

        // If evidence tag like E1, E2, E3...
        if (tag.startsWith('E')) {
          const idx = parseInt(tag.replace('E', ''), 10) - 1;
          const matched = evidenceCache.find(ev => ev.evidence_id === tag) || (idx >= 0 && idx < evidenceCache.length ? evidenceCache[idx] : null);
          if (matched) {
            if (graphRenderer && matched.relationship_path && matched.relationship_path.length) {
              const targetNodeId = matched.relationship_path[matched.relationship_path.length - 1];
              if (graphRenderer.selectNodeById) {
                graphRenderer.selectNodeById(targetNodeId);
              }
            }
            openNodeDrawerForEvidence(matched);
            return;
          }
        }

        // If Risk Engine or Cost Model tag
        if (tag === 'RISK_ENGINE' || tag === 'ENGINE' || tag === 'COST_MODEL') {
          const gauge = document.querySelector('#gauge-progress-path');
          if (gauge) {
            gauge.classList.add('animate-pulse');
            setTimeout(() => gauge.classList.remove('animate-pulse'), 1500);
          }
        }
      };
    });
  }

  function renderInvestigationReport(report) {
    if (!report) return;

    if (report.action) {
      const badgeEl = document.querySelector('#header-decision-badge');
      const scoreDecisionEl = document.querySelector('#score-decision');
      if (badgeEl) {
        badgeEl.textContent = report.action;
        let badgeStyle = 'bg-secondary/20 text-secondary border-secondary/30';
        if (report.action === 'BLOCK') badgeStyle = 'bg-tertiary-container/20 text-tertiary-fixed border-tertiary/30';
        else if (report.action === 'THROTTLE') badgeStyle = 'bg-tertiary/20 text-tertiary border-tertiary/30';
        else if (report.action === 'VERIFY') badgeStyle = 'bg-primary-container/20 text-primary border-primary/30';
        badgeEl.className = `px-2 py-0.5 rounded-sm font-label-sm text-label-sm uppercase tracking-wider font-semibold border ${badgeStyle}`;
      }
      if (scoreDecisionEl) {
        scoreDecisionEl.textContent = report.action;
        let actColor = 'text-secondary';
        if (report.action === 'BLOCK') actColor = 'text-tertiary-fixed';
        else if (report.action === 'THROTTLE') actColor = 'text-tertiary';
        else if (report.action === 'VERIFY') actColor = 'text-primary';
        scoreDecisionEl.className = `font-headline-md text-headline-md font-bold uppercase mt-1 ${actColor}`;
      }
    }

    const summaryBox = document.querySelector('#ai-executive-summary');
    const summaryText = report.narrative_summary || report.executive_summary || 'Investigation completed with verified graph context.';
    if (summaryBox) {
      summaryBox.innerHTML = formatMarkdown(summaryText);
      attachCitationListeners(summaryBox);
    }

    const groundBadge = document.getElementById('ai-grounding-badge');
    if (groundBadge) {
      if (!report.is_fallback && report.confidence > 0.6) {
        groundBadge.textContent = 'GROUNDED KNOWLEDGE GRAPH';
        groundBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono-sm font-semibold bg-secondary/20 text-secondary border border-secondary/30';
      } else {
        groundBadge.textContent = 'EVIDENCE-BACKED SYNTHESIS';
        groundBadge.className = 'px-2 py-0.5 rounded text-[10px] font-mono-sm font-semibold bg-primary-container/20 text-primary border border-primary/30';
      }
    }

    const findingsList = document.getElementById('ai-key-findings');
    if (findingsList) {
      findingsList.innerHTML = '';
      const reasons = report.reasons || report.key_findings || [];
      reasons.forEach(item => {
        const statement = typeof item === 'string' ? item : item.statement;
        const evidenceIds = typeof item === 'object' && item.evidence_ids ? item.evidence_ids : [];

        const li = document.createElement('li');
        li.className = 'flex items-start gap-space-xs text-body-sm text-on-surface-variant';
        
        let badgesHtml = '';
        if (evidenceIds.length) {
          badgesHtml = evidenceIds.map(eid => `<button type="button" class="citation-tag px-1.5 py-0.2 rounded bg-surface-container-high text-primary hover:bg-primary hover:text-on-primary font-mono-sm text-[11px] font-semibold transition-colors cursor-pointer" data-citation="${eid}">[${eid}]</button>`).join(' ');
        }

        li.innerHTML = `
          <span class="material-symbols-outlined text-[16px] text-primary mt-0.5 shrink-0">verified</span>
          <span class="leading-relaxed text-on-surface">${statement} ${badgesHtml}</span>
        `;
        findingsList.appendChild(li);
      });
      attachCitationListeners(findingsList);
    }
  }

  function handleNodeSelected(node) {
    selectedNodeData = node;
    const drawer = document.getElementById('node-drawer');
    if (!drawer) return;

    drawer.classList.remove('translate-x-full');

    setTextContent('#inspector-node-title', node.label || node.id);
    setTextContent('#inspector-node-type', (node.node_type || 'Unknown Node').toUpperCase());
    setTextContent('#inspector-node-risk', `Risk Weight: ${Math.round((node.risk_score || 0) * 100)}%`);
    setTextContent('#inspector-fraud-status', node.is_fraud ? 'Confirmed Historical Fraud' : 'Active Graph Entity');

    const propsContainer = document.getElementById('inspector-properties-list');
    if (propsContainer) {
      propsContainer.innerHTML = '';
      const props = node.properties || {};
      const entries = Object.entries(props);
      if (entries.length === 0) {
        propsContainer.innerHTML = '<span class="text-outline text-mono-sm py-2">No additional attributes registered.</span>';
      } else {
        entries.forEach(([k, v]) => {
          const div = document.createElement('div');
          div.className = 'flex items-center justify-between font-mono-sm text-mono-sm py-1 border-b border-outline-variant/10';
          div.innerHTML = `
            <span class="text-outline">${k}</span>
            <span class="text-on-surface truncate max-w-[140px]">${v !== undefined && v !== null ? v : 'Not available'}</span>
          `;
          propsContainer.appendChild(div);
        });
      }
    }
  }

  function setupNodeDrawer() {
    const closeBtn = document.getElementById('close-drawer');
    const drawer = document.getElementById('node-drawer');
    const copyTokenBtn = document.getElementById('btn-copy-token');

    if (closeBtn && drawer) {
      closeBtn.onclick = () => drawer.classList.add('translate-x-full');
    }

    if (copyTokenBtn) {
      copyTokenBtn.onclick = () => {
        const textToCopy = selectedNodeData ? JSON.stringify(selectedNodeData, null, 2) : `Transaction #${currentTxnId}`;
        navigator.clipboard.writeText(textToCopy).then(() => {
          const orig = copyTokenBtn.innerHTML;
          copyTokenBtn.innerHTML = '<span class="material-symbols-outlined text-[16px]">check</span><span>Copied to Clipboard!</span>';
          setTimeout(() => copyTokenBtn.innerHTML = orig, 1800);
        }).catch(() => {
          alert('Copied to clipboard.');
        });
      };
    }
  }

  function setupGraphControls() {
    const hop1Btn = document.getElementById('btn-hop-1');
    const hop2Btn = document.getElementById('btn-hop-2');
    
    if (hop1Btn && hop2Btn) {
      hop1Btn.onclick = async () => {
        currentMaxHops = 1;
        hop1Btn.className = 'px-space-xs py-1 rounded bg-surface-container-high text-primary font-mono-sm text-mono-sm font-semibold shadow-xs';
        hop2Btn.className = 'px-space-xs py-1 rounded font-mono-sm text-mono-sm text-outline hover:text-on-surface transition-colors';
        const g = await window.SentinelAPI.getGraph(currentTxnId, 1);
        if (graphRenderer && g) graphRenderer.render(g);
      };

      hop2Btn.onclick = async () => {
        currentMaxHops = 2;
        hop2Btn.className = 'px-space-xs py-1 rounded bg-surface-container-high text-primary font-mono-sm text-mono-sm font-semibold shadow-xs';
        hop1Btn.className = 'px-space-xs py-1 rounded font-mono-sm text-mono-sm text-outline hover:text-on-surface transition-colors';
        const g = await window.SentinelAPI.getGraph(currentTxnId, 2);
        if (graphRenderer && g) graphRenderer.render(g);
      };
    }

    const zoomInBtn = document.getElementById('btn-zoom-in');
    const zoomOutBtn = document.getElementById('btn-zoom-out');
    const fitBtn = document.getElementById('btn-fit');

    if (zoomInBtn) zoomInBtn.onclick = () => graphRenderer && graphRenderer.zoomIn();
    if (zoomOutBtn) zoomOutBtn.onclick = () => graphRenderer && graphRenderer.zoomOut();
    if (fitBtn) fitBtn.onclick = () => graphRenderer && graphRenderer.fit();

    const toggleFraud = document.getElementById('toggle-fraud-path');
    if (toggleFraud) {
      toggleFraud.onchange = (e) => {
        if (graphRenderer) graphRenderer.toggleFraudHighlight(e.target.checked);
      };
    }
  }

  function setupInvestigatorActions() {
    const rerunBtn = document.getElementById('btn-rerun');
    if (rerunBtn) {
      rerunBtn.onclick = async () => {
        if (!currentTxnId) return;
        const origHtml = rerunBtn.innerHTML;
        rerunBtn.disabled = true;
        rerunBtn.innerHTML = '<span class="material-symbols-outlined text-[18px] animate-spin">refresh</span><span>Re-running...</span>';
        try {
          await loadTransactionData(currentTxnId);
        } finally {
          rerunBtn.disabled = false;
          rerunBtn.innerHTML = origHtml;
        }
      };
    }
  }

  function setupQuestionChips() {
    const chips = document.querySelectorAll('.question-chip');
    const input = document.getElementById('investigator-input');

    chips.forEach(chip => {
      chip.onclick = () => {
        const questionText = chip.textContent.trim();
        if (input) {
          input.value = questionText;
        }
        submitAIQuestion(questionText);
      };
    });
  }

  function setupCopilotQA() {
    const form = document.getElementById('investigator-form');
    const input = document.getElementById('investigator-input');

    if (form && input) {
      form.onsubmit = (e) => {
        e.preventDefault();
        const q = input.value.trim();
        if (!q) return;
        submitAIQuestion(q);
      };
    }
  }

  async function submitAIQuestion(question) {
    const input = document.getElementById('investigator-input');
    if (input) input.value = '';

    appendQAMessage('user', question);
    const loadingMsg = appendQAMessage('system', 'Consulting Knowledge Graph & Verified Evidence...');

    try {
      const response = await window.SentinelAPI.askInvestigator(currentTxnId, question);
      if (loadingMsg) loadingMsg.remove();
      appendQAMessage('copilot', response.answer, response.evidence_citations);
    } catch (err) {
      if (loadingMsg) loadingMsg.remove();
      appendQAMessage('system', `Unable to complete query: ${err.message}`);
    }
  }

  function appendQAMessage(sender, text, citations = []) {
    const container = document.getElementById('copilot-history-container');
    if (!container) return null;

    const div = document.createElement('div');
    if (sender === 'user') {
      div.className = 'p-space-sm rounded-lg bg-surface-container-high text-on-surface self-end max-w-[90%] text-body-sm border border-outline-variant/20';
      div.textContent = text;
    } else if (sender === 'system') {
      div.className = 'p-space-xs text-outline text-mono-sm italic';
      div.textContent = text;
    } else {
      div.className = 'p-space-md rounded-lg bg-surface-container border border-primary/30 text-on-surface text-body-sm leading-relaxed flex flex-col gap-1';
      
      let citationHtml = '';
      if (citations && citations.length) {
        citationHtml = `
          <div class="mt-2 pt-2 border-t border-outline-variant/20 flex flex-wrap items-center gap-1">
            <span class="text-outline text-[11px] font-mono-sm">Evidence Citations:</span>
            ${citations.map(c => `<button type="button" class="citation-tag px-1.5 py-0.5 rounded bg-surface-container-high text-primary hover:bg-primary hover:text-on-primary font-mono-sm text-[10px] font-semibold transition-colors cursor-pointer" data-citation="${c}">[${c}]</button>`).join(' ')}
          </div>
        `;
      }

      div.innerHTML = `
        <div class="flex items-center gap-1 text-primary font-label-sm uppercase font-semibold">
          <span class="material-symbols-outlined text-[16px]">smart_toy</span>
          <span>Grounded Copilot</span>
        </div>
        <div class="markdown-content mt-1">${formatMarkdown(text)}</div>
        ${citationHtml}
      `;
      attachCitationListeners(div);
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }

  function setupExportEvidence() {
    const exportBtn = document.getElementById('btn-export');
    if (!exportBtn) return;

    exportBtn.onclick = () => {
      const payload = {
        transaction_id: currentTxnId,
        exported_at: new Date().toISOString(),
        risk_evaluation: currentRiskRecord,
        evidence: evidenceCache,
      };

      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `sentinel_evidence_${currentTxnId}_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };
  }

  function setupTechSpecsToggle() {
    const toggleBtn = document.getElementById('toggle-tech-specs');
    const drawer = document.getElementById('tech-specs-drawer');
    const arrow = document.getElementById('tech-arrow');

    if (toggleBtn && drawer) {
      toggleBtn.onclick = () => {
        const isHidden = drawer.classList.contains('hidden');
        if (isHidden) {
          drawer.classList.remove('hidden');
          drawer.classList.add('grid');
          if (arrow) arrow.classList.add('rotate-180');
        } else {
          drawer.classList.add('hidden');
          drawer.classList.remove('grid');
          if (arrow) arrow.classList.remove('rotate-180');
        }
      };
    }
  }

  function setTextContent(selector, text) {
    const el = document.querySelector(selector);
    if (el) el.textContent = text;
  }

  function showLoadingState(isLoading) {
    // Optional loading indicator hook
  }

  // Reactive listener for transaction selection changes
  window.addEventListener('sentinel:transactionChange', (e) => {
    if (e.detail && e.detail.transactionId) {
      currentTxnId = e.detail.transactionId;
      loadTransactionData(currentTxnId);
    }
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initInvestigation);
  } else {
    initInvestigation();
  }
})();
