#!/usr/bin/env bash
set -u
step() { echo "=== $1 $(date +%T)"; shift; "$@" || { echo "FAILED"; exit 1; }; }
step train-ltr csre train-ltr
step eval csre eval --corpus full --split test
step simulate-ab csre simulate-ab --corpus full
step bench csre bench --corpus full
step demo-index csre index --corpus demo
step snapshot csre snapshot --corpus demo
echo ALLDONE
