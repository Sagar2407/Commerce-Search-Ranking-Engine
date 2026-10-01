#!/usr/bin/env bash
# Phase-2 pipeline after the encoder: run stages one at a time (memory), stop on the first failure.
set -u
for st in "$@"; do
  echo "=== $st $(date +%T)"
  case $st in
    eval) csre eval --corpus full --split test || { echo "FAILED $st"; exit 1; } ;;
    demo-index) csre index --corpus demo || { echo "FAILED $st"; exit 1; } ;;
    snapshot) csre snapshot --corpus demo || { echo "FAILED $st"; exit 1; } ;;
    *) csre "$st" || { echo "FAILED $st"; exit 1; } ;;
  esac
done
echo ALLDONE
