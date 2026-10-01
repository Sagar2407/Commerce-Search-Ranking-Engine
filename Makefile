.PHONY: install data acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard test clean

install:
	pip install -e ".[dev]"

# Full data layer, in dependency order (~40 min on 2 vCPU / 8 GB; each stage is re-runnable).
data: acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard

acquire taxonomy catalog judgments graph commerce traffic replay scale demo validate datacard:
	csre $@ $(ARGS)

# Example: bigger simulation, more shards (constant memory per shard)
#   make traffic ARGS="--set simulation.traffic.n_sessions=100000000 --set simulation.traffic.shards=400"

test:
	pytest -q

clean:
	rm -rf data/processed data/synthetic data/scale data/models data/reports data/demo data/_tmp
