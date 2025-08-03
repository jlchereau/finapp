install:
	.venv/bin/pip install -U pip &&\
		.venv/bin/pip install -r requirements-dev.txt

build:
	.venv/bin/reflex compile

reset:
	rm -rf .web

run:
	@echo "Killing any processes on ports 3000 and 8000..."
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	.venv/bin/reflex run

test:
	.venv/bin/python -m pytest -vv tests/

format:
	.venv/bin/black app/ tests/ *.py

lint:
	.venv/bin/pylint --disable=R,C app/ tests/ *.py

all:
	install lint test format