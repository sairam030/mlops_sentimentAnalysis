#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# smoke_test.sh — Verify deployment health
# ═══════════════════════════════════════════════════════════════
# This script runs basic smoke tests to verify the deployed
# application is working correctly. Used in CI/CD pipeline
# after deployment to test/prod environments.
#
# Usage:
#   ./scripts/smoke_test.sh http://18.61.98.223
#   ./scripts/smoke_test.sh http://localhost
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
BASE_URL=${1:-"http://localhost"}
TIMEOUT=10

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  🧪 SMOKE TESTS"
echo "═══════════════════════════════════════════════════════════"
echo "  Target: $BASE_URL"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# ───────────────────────────────────────────────────────────────
# Helper: Check HTTP response
# ───────────────────────────────────────────────────────────────
check_http() {
    local url=$1
    local expected_code=$2
    local description=$3
    
    echo -n "  ⏳ $description..."
    
    response=$(curl -s -o /dev/null -w "%{http_code}" --max-time $TIMEOUT "$url" || echo "000")
    
    if [ "$response" -eq "$expected_code" ]; then
        echo -e " ${GREEN}✅ PASS${NC} ($response)"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e " ${RED}❌ FAIL${NC} (expected $expected_code, got $response)"
        ((TESTS_FAILED++))
        return 1
    fi
}

# ───────────────────────────────────────────────────────────────
# Helper: Check JSON response
# ───────────────────────────────────────────────────────────────
check_json() {
    local url=$1
    local expected_field=$2
    local description=$3
    
    echo -n "  ⏳ $description..."
    
    response=$(curl -s --max-time $TIMEOUT "$url" || echo "{}")
    
    if echo "$response" | grep -q "\"$expected_field\""; then
        echo -e " ${GREEN}✅ PASS${NC}"
        echo "     Response contains: $expected_field"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e " ${RED}❌ FAIL${NC}"
        echo "     Expected field not found: $expected_field"
        echo "     Response: $response"
        ((TESTS_FAILED++))
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════

echo "[1] Basic Health Checks"
echo "───────────────────────────────────────────────────────────"
check_http "$BASE_URL/api/health" 200 "Health endpoint"
check_http "$BASE_URL/" 200 "Frontend root"
check_http "$BASE_URL/dashboard" 200 "Dashboard page"
echo ""

echo "[2] API Endpoints"
echo "───────────────────────────────────────────────────────────"

# Test prediction endpoint
echo -n "  ⏳ Prediction endpoint..."
prediction_response=$(curl -s --max-time $TIMEOUT \
    -X POST "$BASE_URL/api/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a great product, I love it!"}' \
    || echo "{}")

if echo "$prediction_response" | grep -q "\"prediction\""; then
    echo -e " ${GREEN}✅ PASS${NC}"
    
    # Extract prediction details
    prediction=$(echo "$prediction_response" | grep -o '"prediction":"[^"]*"' | cut -d'"' -f4)
    confidence=$(echo "$prediction_response" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    
    echo "     Prediction: $prediction (confidence: $confidence)"
    ((TESTS_PASSED++))
else
    echo -e " ${RED}❌ FAIL${NC}"
    echo "     Response: $prediction_response"
    ((TESTS_FAILED++))
fi

# Test dashboard stats
check_json "$BASE_URL/api/dashboard/stats" "total_predictions" "Dashboard stats"
echo ""

echo "[3] Dashboard Visualizations"
echo "───────────────────────────────────────────────────────────"
check_json "$BASE_URL/api/dashboard/label_distribution" "data" "Label distribution chart"
check_json "$BASE_URL/api/dashboard/confidence_histogram" "data" "Confidence histogram"
check_json "$BASE_URL/api/dashboard/confidence_ranges" "data" "Confidence ranges chart"
check_json "$BASE_URL/api/dashboard/predictions_over_time" "data" "Timeline chart"
echo ""

echo "[4] Download Endpoints"
echo "───────────────────────────────────────────────────────────"
check_http "$BASE_URL/api/download/csv" 200 "CSV download"
echo ""

echo "[5] Error Handling"
echo "───────────────────────────────────────────────────────────"

# Test invalid prediction (missing text field)
echo -n "  ⏳ Invalid request handling..."
error_response=$(curl -s --max-time $TIMEOUT \
    -X POST "$BASE_URL/api/predict" \
    -H "Content-Type: application/json" \
    -d '{"invalid": "field"}' \
    || echo "{}")

if echo "$error_response" | grep -q "error"; then
    echo -e " ${GREEN}✅ PASS${NC}"
    echo "     Error handled correctly"
    ((TESTS_PASSED++))
else
    echo -e " ${RED}❌ FAIL${NC}"
    echo "     Expected error response, got: $error_response"
    ((TESTS_FAILED++))
fi

echo ""

# ═══════════════════════════════════════════════════════════════
# RESULTS SUMMARY
# ═══════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  📊 TEST RESULTS"
echo "═══════════════════════════════════════════════════════════"
echo -e "  ${GREEN}✅ Passed:${NC} $TESTS_PASSED"
echo -e "  ${RED}❌ Failed:${NC} $TESTS_FAILED"
echo "  ───────────────────────────────────────────────────────"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "  ${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo "═══════════════════════════════════════════════════════════"
    exit 0
else
    echo -e "  ${RED}⚠️  SOME TESTS FAILED${NC}"
    echo "═══════════════════════════════════════════════════════════"
    exit 1
fi
