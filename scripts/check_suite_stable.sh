#!/usr/bin/env bash
# Is the suite stable? Ten consecutive runs at FAULT_RATE=0.4 and
# TEMPERATURE=0.7, then the same suite under `make test-graded` (fault headers
# refused). Passing once proves little against injected chaos; passing ten
# times in a row is the claim worth making.
#
# Usage: bash scripts/check_suite_stable.sh [runs]
set -uo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-10}"
PYTEST=".venv/bin/pytest"
export FAULT_RATE=0.4 TEMPERATURE=0.7

pass=0; fail=0
echo "=== $RUNS consecutive runs at FAULT_RATE=$FAULT_RATE TEMPERATURE=$TEMPERATURE ==="
for i in $(seq 1 "$RUNS"); do
  start=$(date +%s)
  out=$($PYTEST 2>&1); rc=$?
  elapsed=$(( $(date +%s) - start ))
  summary=$(echo "$out" | tail -1 | tr -d '\r')
  if [ $rc -eq 0 ]; then
    pass=$((pass+1)); printf "run %2d/%s  GREEN  %3ds  %s\n" "$i" "$RUNS" "$elapsed" "$summary"
  else
    fail=$((fail+1)); printf "run %2d/%s  RED    %3ds\n" "$i" "$RUNS" "$elapsed"
    echo "$out" | tail -25
  fi
done

echo
echo "=== graded run (MOCK_ALLOW_HEADER_OVERRIDE=false) ==="
graded_out=$(MOCK_ALLOW_HEADER_OVERRIDE=false $PYTEST 2>&1); graded_rc=$?
echo "$graded_out" | tail -3
[ $graded_rc -eq 0 ] && graded="GREEN" || graded="RED"

echo
echo "=== RESULT: $pass green, $fail red out of $RUNS; graded=$graded ==="
[ $fail -eq 0 ] && [ $graded_rc -eq 0 ] || exit 1
