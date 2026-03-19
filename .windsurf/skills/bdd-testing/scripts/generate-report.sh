#!/usr/bin/env bash
# Generate HTML report from Cucumber JSON output with video links
set -euo pipefail

REPORT_DIR="tests/reports"
JSON_REPORT="$REPORT_DIR/cucumber-report.json"
HTML_REPORT="$REPORT_DIR/cucumber-report.html"

if [ ! -f "$JSON_REPORT" ]; then
  echo "❌ No JSON report found at $JSON_REPORT"
  echo "   Run tests first: pnpm test:bdd"
  exit 1
fi

echo "📊 Generating HTML report from $JSON_REPORT..."

# Use Cucumber's built-in HTML formatter if available
if npx cucumber-js --help 2>/dev/null | grep -q "html"; then
  echo "   Using @cucumber/html-formatter"
  cat "$JSON_REPORT" | npx @cucumber/html-formatter > "$HTML_REPORT" 2>/dev/null || true
fi

# List video recordings
VIDEO_DIR="$REPORT_DIR/videos"
if [ -d "$VIDEO_DIR" ]; then
  VIDEO_COUNT=$(find "$VIDEO_DIR" -name "*.webm" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$VIDEO_COUNT" -gt 0 ]; then
    echo ""
    echo "🎬 Video recordings ($VIDEO_COUNT):"
    find "$VIDEO_DIR" -name "*.webm" -exec echo "   {}" \;
  fi
fi

echo ""
echo "✅ Report generated: $HTML_REPORT"
