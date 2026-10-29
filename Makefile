VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
PYTEST ?= $(VENV)/bin/pytest
RUFF ?= $(VENV)/bin/ruff

.PHONY: setup test lint verify-frozen build-data ci serve-api serve-ui \
	gate-0 gate-1 gate-2 gate-3 gate-4 gate-5 gate-5a gate-6 gate-7 gate-8 gate-9 gate-10

setup:
	$(PYTHON) -m pip install -e .

serve-api:
	$(PYTHON) -m uvicorn api.app:app --host 0.0.0.0 --port 8000

serve-ui:
	$(PYTHON) -m streamlit run ui/app.py --server.port 8501

test:
	$(PYTEST)

lint:
	$(RUFF) check .

verify-frozen:
	$(PYTHON) scripts/verify_frozen.py

build-data:
	PYTHONHASHSEED=0 $(PYTHON) scripts/build_data.py

ci: lint test verify-frozen

gate-0:
	@if [ ! -f gates/gate_00.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_00

gate-1:
	@if [ ! -f gates/gate_01.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_01

gate-2:
	@if [ ! -f gates/gate_02.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_02

gate-3:
	@if [ ! -f gates/gate_03.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_03

gate-4:
	@if [ ! -f gates/gate_04.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_04

gate-5:
	@if [ ! -f gates/gate_05.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_05

gate-5a:
	@if [ -f gates/gate_05a.py ]; then PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_05a; elif [ -f gates/gate_5a.py ]; then PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_5a; else echo "gate not implemented" >&2; exit 2; fi

gate-6:
	@if [ ! -f gates/gate_06.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_06

gate-7:
	@if [ ! -f gates/gate_07.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_07

gate-8:
	@if [ ! -f gates/gate_08.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_08

gate-9:
	@if [ ! -f gates/gate_09.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_09

gate-10:
	@if [ ! -f gates/gate_10.py ]; then echo "gate not implemented" >&2; exit 2; fi; PYTHONHASHSEED=0 $(PYTHON) -m gates.gate_10
