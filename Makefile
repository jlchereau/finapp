install:
	.venv/bin/pip install -U pip &&\
		.venv/bin/pip install -r requirements-dev.txt

build:
	.venv/bin/reflex compile

reset:
	rm -rf .web

run:
	.venv/bin/reflex run

test:
	.venv/bin/python -m pytest -vv tests/

format:
	.venv/bin/black app/ tests/ *.py

lint:
	.venv/bin/pylint --disable=R,C app/ tests/ *.py

all:
	install lint test format