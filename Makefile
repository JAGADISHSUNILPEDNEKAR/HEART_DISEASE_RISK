.PHONY: install lint test train run all

install:
	pip install -r requirements.txt

lint:
	ruff check .
	black --check .

format:
	black .

test:
	pytest tests/

train:
	python train.py

run:
	streamlit run app.py

all: install lint test
