.PHONY: install test repro gaia clean

install:  ## install package + dev deps
	pip install -e ".[dev]"

test:  ## run the full test suite
	pytest -q --no-header

repro:  ## one-command demo: install + run a single GAIA L1 question
	pip install -e ".[dev]"
	python -m banna_agent.benchmarks.gaia.runner \
	    --policy verifier_retry --provider openai --model gpt-5-nano \
	    --level 1 --n 1

gaia:  ## full GAIA validation (165 Q) with verifier_retry
	python experiments/02_gaia_full/run.py \
	    --policy verifier_retry --provider openai --model gpt-5-nano \
	    --all-levels

clean:  ## strip pycache + caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
