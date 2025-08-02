install:
	.venv/bin/pip install -U pip &&\
		.venv/bin/pip install -r requirements-dev.txt

run:
	.venv/bin/reflex run

test:
	.venv/bin/python -m pytest -vv test_hello.py

format:
	.venv/bin/black app/ tests/ *.py

lint:
	.venv/bin/pylint --disable=R,C app/ tests/ *.py

all:
	install lint test format