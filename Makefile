export PYTHONPATH := src

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
