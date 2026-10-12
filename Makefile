PY ?= python

.PHONY: test paper pilot-analysis pilot-run reproduce lock

test:
	$(PY) -m pytest -q

paper:
	$(PY) paper/reproduce.py

pilot-analysis:
	@if ls experiments/length_pilot/results/*.jsonl >/dev/null 2>&1; then \
		$(PY) experiments/length_pilot/analyse.py; \
	else echo "No pilot logs yet in experiments/length_pilot/results/"; fi

pilot-run:
	$(PY) experiments/length_pilot/run.py --config experiments/length_pilot/pilot.json

reproduce: test paper pilot-analysis

lock:
	$(PY) -m pip freeze --exclude-editable > requirements-lock.txt
