# Thin wrappers; every target also works without make via `python -m ...` (see README).
PY ?= python
TRACE ?= traces/mixed-small.yaml
REPS ?= 3
MODE ?= sim

.PHONY: help install lint fmt test test-gpu test-integration cov check-tools cluster-up cluster-down images \
        images-gpu images-workloads deploy status demo demo-offline experiment experiment-kind traces dashboards clean

help:
	@echo "install | lint | test | cov | check-tools | cluster-up | images | deploy | demo | demo-offline"
	@echo "experiment TRACE=... [MODE=sim|kind] [REPS=3] | cluster-down | traces | dashboards"

install:
	$(PY) -m pip install -e ".[dev,gpu]"

lint:
	$(PY) -m ruff check src tests scripts workloads

fmt:
	$(PY) -m ruff check --fix src tests scripts workloads

test:
	$(PY) -m pytest -q

cov:
	$(PY) -m pytest -q --cov=gpuorch --cov-report=term-missing

test-gpu:            ## run on a GPU node (WSL2) - TODO(verify-on-gpu)
	$(PY) -m pytest -q -m gpu

test-integration:    ## needs `make cluster-up images deploy`
	$(PY) -m pytest -q -m integration

check-tools:
	$(PY) -m gpuorch.tools.cluster check-tools

# --- M0: local cluster ------------------------------------------------------------------------
cluster-up:
	$(PY) -m gpuorch.tools.cluster up

cluster-down:
	$(PY) -m gpuorch.tools.cluster down

status:
	$(PY) -m gpuorch.tools.cluster status

# --- M1/M2: images + Helm ----------------------------------------------------------------------
images:
	$(PY) -m gpuorch.tools.cluster images

images-gpu:          ## CUDA images for the WSL2 GPU nodes (push to a registry or `k3s ctr images import`)
	docker build --build-arg EXTRAS=gpu -t gpuorch/gpuorch:dev-gpu .
	docker build -t gpuorch/inference:dev workloads/inference
	docker build -t gpuorch/training:dev workloads/training

images-workloads:    ## CPU variants of the real workloads (+ load generator)
	docker build --build-arg BASE_IMAGE=python:3.11-slim --build-arg TORCH_CPU=1 -t gpuorch/inference:cpu workloads/inference
	docker build --build-arg BASE_IMAGE=python:3.11-slim --build-arg TORCH_CPU=1 -t gpuorch/training:cpu workloads/training
	docker build -t gpuorch/load-generator:dev workloads/load-generator

deploy:
	$(PY) -m gpuorch.tools.cluster deploy

# --- M3: demo ----------------------------------------------------------------------------------
demo:                ## cluster up -> deploy -> critical job preempts batch -> summary
	$(PY) -m gpuorch.tools.demo

demo-offline:        ## same scenario in the simulator (no Docker needed)
	$(PY) -m gpuorch.tools.demo --offline

# --- M4: experiments ---------------------------------------------------------------------------
experiment:          ## make experiment TRACE=traces/mixed-small.yaml [MODE=kind]
	$(PY) -m gpuorch.experiments.runner --trace $(TRACE) --mode $(MODE) --reps $(REPS)

experiment-kind:
	$(PY) -m gpuorch.experiments.runner --trace $(TRACE) --mode kind --reps 1

traces:
	$(PY) -m gpuorch.experiments.tracegen --name mixed-small --jobs 40 --rate 0.1 --seed 42 -o traces/mixed-small.yaml
	$(PY) -m gpuorch.experiments.tracegen --name mixed-large --jobs 120 --rate 0.15 --seed 7 -o traces/mixed-large.yaml
	$(PY) -m gpuorch.experiments.tracegen --name smoke --jobs 12 --rate 0.2 --seed 1 -o traces/smoke.csv

dashboards:
	$(PY) scripts/gen_dashboards.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
