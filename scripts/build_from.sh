#!/usr/bin/env bash
# Run the data pipeline from a given stage onwards: scripts/build_from.sh taxonomy [extra csre args]
set -u
STAGES=(acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard)
start=${1:-acquire}; shift || true
run=0
for s in "${STAGES[@]}"; do
  [[ $s == "$start" ]] && run=1
  [[ $run == 1 ]] || continue
  echo "=== $s $(date +%T)"
  csre "$s" "$@" || { echo "FAILED $s"; exit 1; }
done
echo ALLDONE
