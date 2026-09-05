import os
import shutil
import re

os.makedirs('frontend/public', exist_ok=True)
os.makedirs('frontend/src/api', exist_ok=True)
os.makedirs('frontend/src/components', exist_ok=True)

# Copy SVG logo
shutil.copy('frontend/raw_stitch/sentinel_shield_logo.svg', 'frontend/public/sentinel-shield.svg')

screens = {
    'index.html': 'frontend/raw_stitch/risk_overview.html',
    'overview.html': 'frontend/raw_stitch/risk_overview.html',
    'transactions.html': 'frontend/raw_stitch/transaction_monitor.html',
    'investigations.html': 'frontend/raw_stitch/ai_risk_investigation.html',
    'risk-engine.html': 'frontend/raw_stitch/risk_engine_console.html'
}

for target_name, source_path in screens.items():
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Fix navigation links in nav rail
    content = re.sub(r'data-path="overview" href="[^"]*"', 'data-path="overview" href="index.html"', content)
    content = re.sub(r'data-path="transactions" href="[^"]*"', 'data-path="transactions" href="transactions.html"', content)
    content = re.sub(r'data-path="investigations" href="[^"]*"', 'data-path="investigations" href="investigations.html"', content)
    content = re.sub(r'data-path="risk-engine" href="[^"]*"', 'data-path="risk-engine" href="risk-engine.html"', content)
    content = re.sub(r'data-path="graph" href="[^"]*"', 'data-path="graph" href="investigations.html"', content)
    content = re.sub(r'data-path="analytics" href="[^"]*"', 'data-path="analytics" href="risk-engine.html"', content)

    # Replace remote image url with local svg
    content = re.sub(r'src="https://lh3\.googleusercontent\.com/aida/[^"]*"', 'src="public/sentinel-shield.svg"', content)

    target_path = os.path.join('frontend', target_name)
    with open(target_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote {target_name} ({len(content)} bytes)")
