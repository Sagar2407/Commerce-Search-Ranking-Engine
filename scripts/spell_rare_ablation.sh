#!/usr/bin/env bash
set -u
COMMON="--corpus full --split test --methods bm25,hybrid,ltr --parts rerank,retrieval,robustness --set eval.retrieval_queries_per_locale=1000 --set eval.robustness_requests=3000"
csre eval $COMMON --set eval.tag=spell_rare0 --set search.spell.rare_df=0 || exit 1
csre eval $COMMON --set eval.tag=spell_rare50 --set search.spell.rare_df=50 || exit 1
echo ALLDONE
