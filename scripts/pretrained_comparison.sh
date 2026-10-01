#!/usr/bin/env bash
# Compare a pretrained multilingual encoder with the in-domain encoder under identical conditions.
# Needs Hugging Face access (huggingface.co, cdn-lfs.huggingface.co). Production models are not touched:
# the pretrained model is registered under the `candidate` alias and indexed into data/indexes/full_pretrained.
#   scripts/pretrained_comparison.sh [model]    (default: intfloat/multilingual-e5-small)
set -euo pipefail
MODEL=${1:-intfloat/multilingual-e5-small}
CAND="--set search.dense.encoder_alias=candidate --set search.index_tag=pretrained"
pip install -q "sentence-transformers>=3.0"
csre register-pretrained --target "$MODEL"
csre index --corpus full $CAND                         # encodes 1.8M products: hours on CPU, minutes on a GPU
csre eval --corpus full --split test --methods bm25,dense,hybrid_rrf,hybrid \
  --parts rerank,retrieval,latency,robustness --set eval.tag=pretrained $CAND
echo "Compare data/reports/eval_full_test_pretrained.md with data/reports/eval_full_test.md"
echo "Promote it if it wins: csre promote --target dense_encoder=<version>  (then re-run make search)"
