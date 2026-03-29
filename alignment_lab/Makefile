PYTHON ?= python

.PHONY: install editable import-check test-unit test-integration test-smoke lint

install:
	$(PYTHON) -m pip install -r requirements.txt

editable:
	$(PYTHON) -m pip install -e .

import-check:
	$(PYTHON) scripts/check_imports.py

test-unit:
	$(PYTHON) -m pytest tests/unit -q

test-integration:
	$(PYTHON) -m pytest tests/integration -q

test-smoke:
	$(PYTHON) -m pytest tests/smoke -q

lint:
	$(PYTHON) -m compileall src tests
