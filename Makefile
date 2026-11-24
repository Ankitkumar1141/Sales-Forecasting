# Makefile — common commands for the Store Sales Forecasting ML pipeline

.PHONY: help install train predict serve test lint clean

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies
	pip install -r requirements.txt

train:  ## Run the full training pipeline
	python src/pipelines/train_pipeline.py

serve:  ## Start the FastAPI server locally
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

test:  ## Run all unit and integration tests
	pytest tests/ -v

lint:  ## Lint the source code with flake8
	flake8 src/ app.py --max-line-length=120

docker-build:  ## Build the Docker image
	docker build -t store-sales-api:latest .

docker-run:  ## Run the Docker container locally
	docker run -p 8000:8000 store-sales-api:latest

clean:  ## Remove generated artifacts and logs
	rm -rf artifacts/*.csv artifacts/*.joblib artifacts/*.json logs/*.log
