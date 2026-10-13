PY ?= python

.PHONY: test paper pilot-analysis pilot-run pilot-run-local reproduce overleaf lock

test:
	$(PY) -m pytest -q

paper:
	$(PY) paper/reproduce.py

pilot-analysis:
	@if ls experiments/length_pilot/results/*.jsonl >/dev/null 2>&1; then \
		$(PY) experiments/length_pilot/analyse.py; \
	else echo "No pilot logs yet in experiments/length_pilot/results/"; fi
	@if ls experiments/length_pilot/results/*__tools.jsonl >/dev/null 2>&1; then \
		$(PY) experiments/length_pilot/analyse.py --task tools; \
	else echo "No tools-task logs yet"; fi

pilot-run:
	$(PY) experiments/length_pilot/run.py --config experiments/length_pilot/pilot.json

pilot-run-local:
	$(PY) experiments/length_pilot/run.py --config experiments/length_pilot/pilot_local.json
	$(PY) experiments/length_pilot/run.py --config experiments/length_pilot/pilot_mixed.json
	$(PY) experiments/length_pilot/run.py --config experiments/length_pilot/pilot_tools.json

reproduce: test paper pilot-analysis

# Copy the generated tables and figures into the paper's folder, e.g.
#   make overleaf OVERLEAF="/path/to/overleaf/project"
# Run `make reproduce` first. Only the files below are written; nothing else
# in the folder is touched.
PAPER_FILES = paper/output/table_frontier_by_length.tex paper/output/fig_parallel_lines.pdf \
	experiments/length_pilot/output/table_pilot.tex \
	experiments/length_pilot/output/table_pilot_crossovers.tex \
	experiments/length_pilot/output/table_pilot_crossovers_all.tex \
	experiments/length_pilot/output/fig_pilot_energy.pdf \
	experiments/length_pilot/output/fig_pilot_models.pdf

overleaf:
	@test -n "$(OVERLEAF)" || { echo 'Set OVERLEAF="/path/to/paper/folder"'; exit 1; }
	@test -d "$(OVERLEAF)" || { echo "No such folder: $(OVERLEAF)"; exit 1; }
	cp $(PAPER_FILES) "$(OVERLEAF)/"

lock:
	$(PY) -m pip freeze --exclude-editable > requirements-lock.txt
