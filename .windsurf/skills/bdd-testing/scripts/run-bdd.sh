#!/usr/bin/env bash
# Run BDD tests with optional video recording
# Usage: ./run-bdd.sh [profile] [extra-args...]
#   Profiles: default, api, ui, agents, video
#   Environment:
#     VIDEO=true    — Enable video recording for UI tests
#     HEADLESS=false — Show browser window during tests
#     BASE_URL=...  — Override test target URL

set -euo pipefail

PROFILE="${1:-default}"
shift 2>/dev/null || true

# Ensure reports directory exists
mkdir -p tests/reports/videos tests/reports/traces

# Set video recording if requested
if [ "${VIDEO:-false}" = "true" ] && [ "$PROFILE" != "video" ]; then
  echo "ℹ️  Video recording enabled via VIDEO=true"
  export RECORD_VIDEO=true
fi

echo "🥒 Running BDD tests (profile: $PROFILE)"
echo "   Base URL: ${BASE_URL:-http://localhost:3000}"
echo "   Headless: ${HEADLESS:-true}"
echo ""

npx cucumber-js --profile "$PROFILE" "$@"

# Report summary
echo ""
echo "📊 Reports:"
[ -f tests/reports/cucumber-report.html ] && echo "   HTML: tests/reports/cucumber-report.html"
[ -f tests/reports/cucumber-report.json ] && echo "   JSON: tests/reports/cucumber-report.json"

VIDEO_COUNT=$(find tests/reports/videos -name "*.webm" 2>/dev/null | wc -l | tr -d ' ')
if [ "$VIDEO_COUNT" -gt 0 ]; then
  echo "   🎬 Videos: $VIDEO_COUNT recordings in tests/reports/videos/"
fi
