PYTHON ?= python

.PHONY: train test lint dashboard

train:
	$(PYTHON) -m fraudlab --source auto

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

dashboard:
	$(PYTHON) -m streamlit run app/streamlit_app.py
