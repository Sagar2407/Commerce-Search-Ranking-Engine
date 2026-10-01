#!/usr/bin/env bash
# Spelling-correction ablation: off / expand / replace, same queries and methods.
set -u
COMMON="--corpus full --split test --methods bm25,hybrid,ltr --parts rerank,retrieval,robustness --set eval.retrieval_queries_per_locale=1000 --set eval.robustness_requests=3000"
csre eval $COMMON --set eval.tag=spell_off --set search.spell.enabled=false || exit 1
csre eval $COMMON --set eval.tag=spell_expand --set search.spell.mode=expand || exit 1
csre eval $COMMON --set eval.tag=spell_replace --set search.spell.mode=replace || exit 1
echo ALLDONE
