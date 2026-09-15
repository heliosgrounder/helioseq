# Development shortcuts. Nothing here is required to install the package.

PYTHON  ?= python3
CC      ?= cc
CFLAGS  ?= -std=c11 -Wall -Wextra -Wpedantic -O2
CORE     = src/libushuffle/ushuffle.c
CORE_INC = -Isrc/libushuffle
BUILD    = build/ctest

.PHONY: help build test test-purepy test-all ctest ctest-asan golden bench \
        lint typecheck doctest structure dist clean

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build: ## compile the extension in place
	$(PYTHON) setup.py build_ext --inplace

test: ## run the Python suite against whichever backend is installed
	$(PYTHON) -m pytest tests -q

test-purepy: ## run the same suite against the pure-Python fallback
	HELIOSEQ_BACKEND=python $(PYTHON) -m pytest tests -q

test-all: test test-purepy ## both backends

$(BUILD):
	@mkdir -p build

ctest: $(BUILD) ## C-level tests for the shuffling core
	$(CC) $(CFLAGS) -DUSH_TEST_THREADS $(CORE_INC) \
	    tests/c/test_libushuffle.c $(CORE) -o $(BUILD) -lm -lpthread
	./$(BUILD)

ctest-asan: $(BUILD) ## the same, under AddressSanitizer and UBSan
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
	    -DUSH_TEST_THREADS $(CORE_INC) \
	    tests/c/test_libushuffle.c $(CORE) -o $(BUILD)-asan -lm -lpthread
	./$(BUILD)-asan

golden: $(BUILD) ## regenerate tests/data/golden.json from the C core
	$(CC) $(CFLAGS) $(CORE_INC) tests/c/test_libushuffle.c $(CORE) \
	    -o $(BUILD)-golden -lm
	./$(BUILD)-golden --golden > tests/data/golden.json
	@echo "regenerated tests/data/golden.json -- run 'make test-all' to confirm"
	@echo "both backends still agree before committing it"

bench: ## run the benchmarks
	$(PYTHON) benchmarks/bench.py

lint: ## ruff
	$(PYTHON) -m ruff check src tests benchmarks
	$(PYTHON) -m ruff format --check src tests benchmarks

typecheck: ## mypy
	$(PYTHON) -m mypy

doctest: ## run every example in every docstring
	$(PYTHON) -m pytest tests/test_docs.py -q

structure: ## check the layering rules in docs/architecture.md
	$(PYTHON) -m pytest tests/test_package.py -q

dist: clean ## build the sdist and a local wheel
	HELIOSEQ_REQUIRE_EXTENSION=1 $(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -name '__pycache__' -prune -exec rm -rf {} +
	find src -name '*.so' -o -name '*.pyd' | xargs -r rm -f
