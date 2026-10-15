# Reproducing the paper

Siraj and Barbati, *When Is Smaller Greener? Right-Sizing Language Models for Agentic AI* ([arXiv:2610.19345](https://arxiv.org/abs/2610.19345)).

Every computed number, table and figure in the paper is produced by code in this repository. Nothing in the paper is typed in by hand from another source.

## Environment

Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e . --no-deps
```

`requirements-lock.txt` pins the exact package versions used for the paper. `pip install -e ".[dev,experiments,paper]"` installs the same packages without pins.

## One command

```bash
make reproduce
```

This runs the tests, regenerates the Section 3 and 4 numbers, table and figure, and recomputes the pilot results from the committed logs. It needs no API keys and no network access.

The paper reads its generated tables with `\input` and its figures with `\includegraphics`, from the folder that holds `main.tex`. To update them after a new run:

```bash
make reproduce
make overleaf OVERLEAF="/path/to/paper/folder"
```

`make overleaf` copies only the twelve generated files (seven tables, five figures) and touches nothing else in that folder.

## Where each item comes from

| Paper item | Produced by | Output |
|---|---|---|
| Section 3, 20-step example (0.36, 0.82, 2.8 and 1.2 attempts, energy advantage) | `paper/reproduce.py` | `paper/output/numbers.json` (`motivation`) |
| Section 4, crossover lengths 22, 39, 56 | `paper/reproduce.py` | `numbers.json` (`crossovers`) |
| Section 4, chance-constraint examples (0.54, 0.78, 2 and 10 steps) | `paper/reproduce.py` | `numbers.json` (`chance_constraint`) |
| Illustrative example: models D and E pruned | `paper/reproduce.py` | `numbers.json` (`stage1`) |
| Table: efficient models by task length | `paper/reproduce.py` | `paper/output/table_frontier_by_length.tex` |
| Illustrative example: three task profiles | `paper/reproduce.py` | `numbers.json` (`profiles`) |
| Figure: parallel lines and trade-off band (models C and B, crossovers 16.5, 63.7, 76.9) | `paper/reproduce.py` | `paper/output/fig_parallel_lines.pdf`, `numbers.json` (`figure_crossovers_C_vs_B`) |
| Section 6, pilot numbers in the text | `experiments/length_pilot/analyse.py` | `experiments/length_pilot/output/summary.json` |
| Section 6, table of per-model results (order task, panel a) | `experiments/length_pilot/analyse.py` | `experiments/length_pilot/output/table_pilot.tex` |
| Section 6, table of energy crossovers (models shown in the figure) | `experiments/length_pilot/analyse.py` | `experiments/length_pilot/output/table_pilot_crossovers.tex` |
| Appendix, energy crossovers for every model that survives the per-step screen | `experiments/length_pilot/analyse.py` | `experiments/length_pilot/output/table_pilot_crossovers_all.tex` |
| Section 6, figure of energy per successful task | `experiments/length_pilot/analyse.py` | `experiments/length_pilot/output/fig_pilot_energy.pdf` |
| Section 6, figure of all models on energy per step and steps before an error | `experiments/length_pilot/analyse.py` | `experiments/length_pilot/output/fig_pilot_models.pdf` |
| Section 6, the same tables and figures for the tools task (panels b) | `experiments/length_pilot/analyse.py --task tools` | `experiments/length_pilot/output/*_tools.*`, `summary_tools.json` |

`tests/test_horizon.py` asserts the Section 3 and 4 numbers, so the tests fail if the code and the paper disagree.

## Data

The raw logs of every model run used in the paper are in `experiments/length_pilot/results/`. Each line records the model, provider, served model version, date, git commit of the harness, prompts and every step. The analysis is fully reproducible from these logs. Re-running the models calls live services, which can change without notice, so new runs are a replication, not a reproduction.

## Random seeds

Task generation uses fixed seeds (`seed0` in `pilot.json`). Bootstrap intervals in `analyse.py` use `--seed 7`. Model calls use temperature 0.
