.PHONY: install data acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard \
        search train-dense index train-qcat tune-hybrid train-ltr eval bench demo-index serve snapshot test clean

install:
	pip install -e ".[dev]"

# ---------------------------------------------------------------- phase 1: data layer (~1 h on 4 vCPU / 16 GB)
data: acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard

acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard:
	csre $@ $(ARGS)

# Example: bigger simulation, more shards (constant memory per shard)
#   make traffic ARGS="--set simulation.traffic.n_sessions=100000000 --set simulation.traffic.shards=400"

# ---------------------------------------------------------------- phase 2: retrieval, ranking, evaluation
# Train on ESCI train queries, tune on dev, evaluate on the untouched test split (full catalog).
search: train-dense index train-qcat tune-hybrid train-ltr eval

train-dense index train-qcat tune-hybrid train-ltr:
	csre $@ $(ARGS)

eval:
	csre eval --corpus full --split test $(ARGS)

bench:
	csre bench --corpus full $(ARGS)

# ---------------------------------------------------------------- storefront demo
demo-index:          ## indexes for the 151K-product demo corpus, built with the production models
	csre index --corpus demo

serve: demo-index    ## http://localhost:8000
	csre serve --corpus demo

snapshot:            ## self-contained storefront page with precomputed results -> data/reports/storefront_snapshot.html
	csre snapshot --corpus demo

test:
	pytest -q

clean:
	rm -rf data/processed data/synthetic data/scale data/models data/indexes data/runs data/demo data/demo_portable data/_tmp
