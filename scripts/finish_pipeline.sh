#!/usr/bin/env bash
set -u
step() { echo "=== $1 $(date +%T)"; shift; "$@" || { echo "FAILED"; exit 1; }; }
step bench-depth csre bench --corpus full --parts rerank_depth
step bench-scale csre bench --corpus full --parts scale
step demo-index csre index --corpus demo
step snapshot csre snapshot --corpus demo
echo ALLDONE
