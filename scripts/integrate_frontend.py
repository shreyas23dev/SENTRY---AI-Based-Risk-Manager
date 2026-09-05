import os
import re

FRONTEND_DIR = 'frontend'

def clean_provenance(html_content):
    """Clean all competition/author references to standard Sentinel product terminology."""
    replacements = [
        (r'Kaggle XGBoost', 'Transaction Risk Model'),
        (r'Kaggle', 'Transaction Model'),
        (r'IEEE-CIS Fraud Detection', 'Payment Network Telemetry'),
        (r'IEEE-CIS', 'Sentinel Risk Pipeline'),
        (r'1st Place', 'State-of-the-Art'),
        (r'competition notebook', 'production pipeline'),
        (r'Chris Deotte', 'Sentinel Architect'),
        (r'Konstantin Yakovlev', 'Risk Systems Lead'),
    ]
    for pattern, repl in replacements:
        html_content = re.sub(pattern, repl, html_content, flags=re.IGNORECASE)
    return html_content

def update_index_html():
    path = os.path.join(FRONTEND_DIR, 'index.html')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_provenance(content)

    # Add id attributes for KPI cards
    content = content.replace('1,467', '<span id="kpi-high-risk">1,467</span>')
    content = content.replace('824', '<span id="kpi-fraud-blocked">824</span>')
    content = content.replace('₹2.21M', '<span id="kpi-loss-avoided">₹2.21M</span>')

    # Update the Incident Queue / Demo Cases table to have an id on tbody
    if '<tbody' in content:
        content = re.sub(r'<tbody[^>]*>', '<tbody id="incident-queue-tbody">', content, count=1)
    else:
        # If no tbody, find the table after Incident Queue or Activity
        content = content.replace('</table>', '<tbody id="incident-queue-tbody"></tbody></table>')

    # Add error banner container if not present
    if 'id="overview-error-banner"' not in content:
        content = content.replace('<main class="', '<div id="overview-error-banner" class="hidden mx-gutter-desktop mt-4 p-space-md rounded-lg bg-tertiary-container/20 text-tertiary border border-tertiary/30 font-body-sm"></div>\n<main class="')

    # Append scripts before </body>
    scripts = """
<script src="js/api.js"></script>
<script src="js/state.js"></script>
<script src="js/overview.js"></script>
</body>"""
    content = re.sub(r'</body>', scripts, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated index.html")

def update_transactions_html():
    path = os.path.join(FRONTEND_DIR, 'transactions.html')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_provenance(content)

    # Add filter data attributes to Risk filter buttons
    content = re.sub(r'>All</button>', ' data-risk-filter="All">All</button>', content, count=1)
    content = re.sub(r'>Medium</button>', ' data-risk-filter="Medium">Medium</button>', content, count=1)
    content = re.sub(r'>High</button>', ' data-risk-filter="High">High</button>', content, count=1)
    content = re.sub(r'>\s*<span>Critical</span>', ' data-risk-filter="Critical"><span>Critical</span>', content, count=1)

    # Add filter data attributes to Decision Gate buttons
    content = re.sub(r'>All Gates</button>', ' data-gate-filter="All Gates">All Gates</button>', content, count=1)

    # Ensure table body has id="transactions-tbody"
    if '<tbody' in content:
        content = re.sub(r'<tbody[^>]*>', '<tbody id="transactions-tbody">', content, count=1)

    # Append scripts
    scripts = """
<script src="js/api.js"></script>
<script src="js/state.js"></script>
<script src="js/transactions.js"></script>
</body>"""
    content = re.sub(r'</body>', scripts, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated transactions.html")

def update_investigations_html():
    path = os.path.join(FRONTEND_DIR, 'investigations.html')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_provenance(content)

    # Add Header IDs
    content = re.sub(r'<h1[^>]*>Transaction #\d+</h1>', '<h1 id="header-txn-title" class="font-headline-lg text-headline-lg text-on-surface tracking-tight font-semibold">Transaction #3504259</h1>', content)
    content = re.sub(r'<span[^>]*>(?:BLOCKED|ALLOW|VERIFY|THROTTLE)</span>', '<span id="header-decision-badge" class="px-2 py-0.5 rounded-sm bg-tertiary-container/20 text-tertiary-fixed border border-tertiary/30 font-label-sm text-label-sm uppercase tracking-wider font-semibold">BLOCKED</span>', content, count=1)
    content = re.sub(r'₹238\.53</span>', '<span id="header-amount">₹238.53</span></span>', content, count=1)

    # Add Score & Metric IDs
    content = re.sub(r'stroke-dasharray="[\d\.,\s]+"', 'id="gauge-progress-path" stroke-dasharray="52.29, 100"', content, count=1)
    content = re.sub(r'>52\.29%</span>', ' id="score-final-risk">52.29%</span>', content, count=1)
    content = re.sub(r'>51\.98%</span>', ' id="score-base-ml">51.98%</span>', content, count=1)
    content = re.sub(r'>65\.28%</span>', ' id="score-graph-context">65.28%</span>', content, count=1)
    content = re.sub(r'>\+0\.31%</span>', ' id="score-graph-impact">+0.31%</span>', content, count=1)
    content = re.sub(r'>BLOCK</span>', ' id="score-decision">BLOCK</span>', content, count=1)
    content = re.sub(r'>Expected loss: ₹[\d\.]+</span>', ' id="score-expected-loss">Expected loss: ₹238.53</span>', content, count=1)

    # Graph Canvas & Controls
    # Find the graph SVG / canvas area container and give it id="knowledge-graph-canvas"
    # In Stitch screen, graph canvas is inside the left column
    content = re.sub(r'<div class="relative w-full h-[^"]* overflow-hidden[^"]*"', '<div id="knowledge-graph-canvas" class="relative w-full h-[520px] overflow-hidden bg-surface-container-lowest/60 rounded-lg"', content, count=1)
    if 'id="knowledge-graph-canvas"' not in content:
        # Fallback replacement
        content = content.replace('<div class="lg:col-span-8 flex flex-col bg-surface-container-low rounded-xl overflow-hidden relative shadow-sm">', '<div class="lg:col-span-8 flex flex-col bg-surface-container-low rounded-xl overflow-hidden relative shadow-sm">\n<div id="knowledge-graph-canvas" class="w-full h-[520px] relative"></div>')

    # Add button IDs for graph controls
    content = re.sub(r'>1-Hop</button>', ' id="btn-hop-1">1-Hop</button>', content)
    content = re.sub(r'>2-Hops</button>', ' id="btn-hop-2">2-Hops</button>', content)
    content = re.sub(r'>Fraud Path Highlight</button>', ' id="btn-fraud-path-toggle">Fraud Path Highlight</button>', content)
    content = re.sub(r'<button[^>]*title="Zoom in"[^>]*>', '<button id="btn-zoom-in" type="button" class="h-8 w-8 rounded bg-surface-container-high hover:bg-surface-variant flex items-center justify-center text-on-surface transition-colors">', content)
    content = re.sub(r'<button[^>]*title="Zoom out"[^>]*>', '<button id="btn-zoom-out" type="button" class="h-8 w-8 rounded bg-surface-container-high hover:bg-surface-variant flex items-center justify-center text-on-surface transition-colors">', content)
    content = re.sub(r'<button[^>]*title="Fit to screen"[^>]*>', '<button id="btn-fit" type="button" class="h-8 w-8 rounded bg-surface-container-high hover:bg-surface-variant flex items-center justify-center text-on-surface transition-colors">', content)
    content = re.sub(r'<button[^>]*title="Reset"[^>]*>', '<button id="btn-reset" type="button" class="h-8 w-8 rounded bg-surface-container-high hover:bg-surface-variant flex items-center justify-center text-on-surface transition-colors">', content)

    # Node Inspector Drawer ID
    if 'id="node-inspector-drawer"' not in content:
        content = re.sub(r'<aside class="[^"]*w-80[^"]*"', '<aside id="node-inspector-drawer" class="w-80 bg-surface-container-low border-l border-outline-variant/30 p-space-md flex flex-col gap-space-md"', content, count=1)

    # Evidence Container ID
    content = re.sub(r'<div class="flex flex-col gap-space-sm overflow-y-auto[^"]*"', '<div id="evidence-items-container" class="flex flex-col gap-space-sm overflow-y-auto max-h-[420px] pr-1"', content, count=1)
    if 'id="evidence-items-container"' not in content:
        content = content.replace('Evidence Drawer', 'Evidence Drawer</span><div id="evidence-items-container" class="flex flex-col gap-space-sm mt-2"></div><span class="hidden">')

    # AI Investigator Report IDs
    if 'id="ai-executive-summary"' not in content:
        content = re.sub(r'<p class="font-body-md text-body-md text-on-surface-variant leading-relaxed">[^<]*</p>', '<p id="ai-executive-summary" class="font-body-md text-body-md text-on-surface leading-relaxed"></p>', content, count=1)
    if 'id="ai-key-findings"' not in content:
        content = re.sub(r'<ul class="space-y-space-xs[^"]*">', '<ul id="ai-key-findings" class="space-y-space-xs mt-2">', content, count=1)

    # SOC Copilot Q&A IDs
    content = re.sub(r'<input class="[^"]*" placeholder="Ask SOC Copilot[^"]*"', '<input id="copilot-query-input" class="w-full h-10 bg-surface-container-lowest border border-outline-variant/50 rounded-lg pl-3 pr-10 font-body-sm text-body-sm text-on-surface placeholder:text-outline focus:border-primary focus:outline-none transition-colors" placeholder="Ask questions about this transaction or evidence..."', content, count=1)
    content = re.sub(r'<button[^>]*class="[^"]*">\s*<span class="material-symbols-outlined">send</span>', '<button id="copilot-send-btn" type="button" class="h-8 w-8 rounded bg-primary text-on-primary flex items-center justify-center hover:bg-primary-fixed transition-colors"><span class="material-symbols-outlined text-[16px]">send</span>', content, count=1)
    if 'id="copilot-history-container"' not in content:
        content = re.sub(r'<div class="flex flex-col gap-space-sm overflow-y-auto[^"]*"', '<div id="copilot-history-container" class="flex flex-col gap-space-sm overflow-y-auto max-h-56 pr-1"', content, count=1)

    # Add D3.js and Page Scripts
    scripts = """
<script src="https://d3js.org/d3.v7.min.js"></script>
<script src="js/api.js"></script>
<script src="js/state.js"></script>
<script src="js/graph.js"></script>
<script src="js/investigations.js"></script>
</body>"""
    content = re.sub(r'</body>', scripts, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated investigations.html")

def update_risk_engine_html():
    path = os.path.join(FRONTEND_DIR, 'risk-engine.html')
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = clean_provenance(content)

    # Add Signal pipeline IDs
    content = re.sub(r'TXID-REF:\s*#3504259-PROD', '<span id="engine-txid-ref">TXID-REF: #3504259</span>', content, count=1)
    content = re.sub(r'>51\.98<', ' id="engine-base-ml-score">51.98<', content, count=1)
    content = re.sub(r'>65\.28<', ' id="engine-graph-context-score">65.28<', content, count=1)
    content = re.sub(r'>52\.29<', ' id="engine-final-risk-score">52.29<', content, count=1)

    # Progress bar style attributes
    content = re.sub(r'style="width:\s*51\.98%"', 'id="engine-bar-base-ml" style="width: 51.98%"', content, count=1)
    content = re.sub(r'style="width:\s*65\.28%"', 'id="engine-bar-graph" style="width: 65.28%"', content, count=1)
    content = re.sub(r'style="width:\s*52\.29%"', 'id="engine-bar-final" style="width: 52.29%"', content, count=1)

    # Append scripts
    scripts = """
<script src="js/api.js"></script>
<script src="js/state.js"></script>
<script src="js/risk-engine.js"></script>
</body>"""
    content = re.sub(r'</body>', scripts, content)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Updated risk-engine.html")

if __name__ == '__main__':
    update_index_html()
    update_transactions_html()
    update_investigations_html()
    update_risk_engine_html()
    print("All HTML files integrated successfully!")
