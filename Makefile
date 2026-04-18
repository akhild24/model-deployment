install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt -r requirements-dev.txt

test:
	python -m pytest

docker-build:
	docker build -t sentence-embeddings-api .
	docker image ls sentence-embeddings-api

docker-run:
	docker run --rm -p 8000:8000 sentence-embeddings-api
