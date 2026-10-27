# Shortcuts for the experiments of "Scaling Limits of the Dropout Inversion Attack".
#
#   make install                                  # .venv with requirements.txt
#   make smoke                                    # 150 iterations per attack, written to outputs/debug
#   make grid SHARDS=3                            # full grid, resumable
#   make grid MODELS=ResNet50 DROPOUTS="0.25 0.5" # subset of the grid
#   make status
#
# DATASETS, MODELS, DROPOUTS and RUNS are passed to run_grid.py when set; ARGS takes any other option.

# Interpreter used to create the virtualenv (Python 3.12, see requirements.txt). The pyenv install is
# preferred because its python3.12 shim only works where 3.12 is the selected version.
BASE_PYTHON ?= $(firstword $(wildcard $(HOME)/.pyenv/versions/3.12*/bin/python3.12) python3.12)
VENV ?= .venv
PYTHON := $(VENV)/bin/python
INSTALLED := $(VENV)/.installed

SHARDS ?= 1
DATASETS ?=
MODELS ?=
DROPOUTS ?=
RUNS ?=
ARGS ?=
GRID_FILTERS = $(if $(DATASETS),--datasets $(DATASETS)) $(if $(MODELS),--models $(MODELS)) \
	$(if $(DROPOUTS),--dropouts $(DROPOUTS)) $(ARGS)
GRID_ARGS = $(GRID_FILTERS) $(if $(RUNS),--runs $(RUNS))

# Debug runs reuse the run directory names of the full grid, so they must not share its output directory:
# a finished debug run would otherwise be skipped by `make grid` as if it were a real one.
SMOKE_DIR ?= outputs/debug

# Upscaled victim sets generated from data/CIFAR10_128.tdump
SIZES ?= 64 128 224
SCALED_DATA := $(foreach size,$(SIZES),data/CIFAR10_128_$(size)x$(size).tdump)

.DEFAULT_GOAL := help
.PHONY: help install data smoke grid status clean clean-data clean-smoke

help:
	@echo 'Targets:'
	@echo '  install      create $(VENV) and install requirements.txt'
	@echo '  data         generate the upscaled victim sets ($(SIZES))'
	@echo '  smoke        validate the pipeline: 1 run, 150 iterations per attack, in $(SMOKE_DIR)'
	@echo '  grid         run or resume the experiment grid (SHARDS=$(SHARDS))'
	@echo '  status       completed runs per combination'
	@echo '  clean        remove __pycache__ and shard logs'
	@echo '  clean-data   remove the generated upscaled victim sets'
	@echo '  clean-smoke  remove $(SMOKE_DIR)'
	@echo
	@echo 'Variables: DATASETS, MODELS, DROPOUTS, RUNS, SHARDS, ARGS, SIZES, BASE_PYTHON'
	@echo 'Example:   make grid SHARDS=3 DATASETS=CIFAR10_128_224x224 MODELS="ResNet18 ResNet50"'

install: $(INSTALLED)

$(INSTALLED): requirements.txt
	test -x $(PYTHON) || $(BASE_PYTHON) -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch $@

data: $(SCALED_DATA)

# The source is order-only: the script never overwrites an existing file, so a newer source would not
# refresh the outputs anyway. Use `make clean-data data` to regenerate them.
data/CIFAR10_128_%.tdump: | $(INSTALLED) data/CIFAR10_128.tdump
	$(PYTHON) tools/generate_scaled_tdumps.py --sizes $(firstword $(subst x, ,$*))

smoke: data
	$(PYTHON) run_grid.py --debug --runs 1 --output-dir $(SMOKE_DIR) $(GRID_FILTERS)

grid: data
	$(PYTHON) run_grid.py --shards $(SHARDS) $(GRID_ARGS)

status: $(INSTALLED)
	$(PYTHON) run_grid.py --status $(GRID_ARGS)

clean:
	find . -name __pycache__ -type d -not -path './$(VENV)/*' -prune -exec rm -rf {} +
	rm -rf logs

clean-data:
	rm -f $(SCALED_DATA)

clean-smoke:
	rm -rf $(SMOKE_DIR)
