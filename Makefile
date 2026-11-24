export PYTHONPATH := src

# "data" is also a folder name, so every target must be phony or make skips it.
.PHONY: setup data index eval test serve

setup:
	pip install -r requirements-dev.txt

data:
	python -m finrag.cli download
	python -m finrag.cli extract

index:
	python -m finrag.cli build

eval:
	python -m finrag.cli eval-retrieval

test:
	pytest -q

serve:
	uvicorn finrag.api:app --reload --port 8000
