# update code agents
agents:
	npm install -g @anthropic-ai/claude-code @google/gemini-cli @openai/codex

build:
	.venv/bin/reflex compile

# export frontend and backend (beware API_URL)
# https://reflex.dev/docs/hosting/self-hosting/
export:
	.venv/bin/reflex export

# flake8 will auto-read .flake8
flake8:
	.venv/bin/flake8 app/ tests/ *.py

# format using black
format:
	.venv/bin/black app/ tests/ *.py

# install dev environment
install:
	.venv/bin/pip install -U pip &&\
	.venv/bin/pip install -r requirements-dev.txt &&\
	.venv/bin/pip install -U twsapi/IBJts/source/pythonclient

# lint with pylint
pylint:
	.venv/bin/pylint --disable=R,C app/ tests/ *.py

# lint with pyrefly (see pyrefly.toml)
# Note: consider https://docs.astral.sh/ty/ in the future
pyrefly:
	.venv/bin/pyrefly check

# reset frontend (nextJS application)
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

# run unit tests
test:
	.venv/bin/python -m pytest -vv tests/

# lint with flake8, pylint and pyrefly
lint: flake8 pylint pyrefly

# Run all
all: install format lint test
