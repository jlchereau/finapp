install:
	.venv/bin/pip install -U pip &&\
	.venv/bin/pip install -r requirements-dev.txt &&\
	.venv/bin/pip install -U twsapi/IBJts/source/pythonclient

build:
	.venv/bin/reflex compile

reset:
	rm -rf .web

# Note: add paths you want to exclude from hot reloading in .env
# REFLEX_HOT_RELOAD_EXCLUDE_PATHS=data:docs:notebooks:temp:tests
run:
	@echo "Killing any processes on ports 3000 and 8000..."
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	REFLEX_ENV_FILE=.env .venv/bin/reflex run

test:
	.venv/bin/python -m pytest -vv tests/

format:
	.venv/bin/black app/ tests/ *.py

pylint:
	.venv/bin/pylint --disable=R,C app/ tests/ *.py

# flake8 will auto-read .flake8
flake8:
	.venv/bin/flake8 app/ tests/ *.py

# lint with flake8 and pylint
lint: flake8 pylint

all: install format lint test
