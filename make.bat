@echo off
REM ============================================================
REM Windows Batch helper for Makefile targets
REM Usage: make <command> (e.g., make train, make serve)
REM ============================================================

if "%1"=="" goto help
if "%1"=="train" goto train
if "%1"=="serve" goto serve
if "%1"=="frontend" goto frontend
if "%1"=="test" goto test
if "%1"=="install" goto install
if "%1"=="docker-build" goto docker_build
if "%1"=="docker-run" goto docker_run
goto help

:train
echo Running training pipeline...
python src/pipelines/train_pipeline.py
goto end

:serve
echo Starting FastAPI server...
uvicorn app:app --reload --host 0.0.0.0 --port 8000
goto end

:frontend
echo Starting Streamlit frontend...
streamlit run frontend/streamlit_app.py
goto end

:test
echo Running tests...
pytest tests/ -v
goto end

:install
echo Installing requirements...
pip install -r requirements.txt
goto end

:docker_build
echo Building Docker image...
docker build -t store-sales-api:latest .
goto end

:docker_run
echo Running Docker container...
docker run -p 8000:8000 store-sales-api:latest
goto end

:help
echo.
echo Sales Forecasting - Commands:
echo   make train         - Run training pipeline
echo   make serve         - Start FastAPI server locally
echo   make frontend      - Start Streamlit frontend (port 8501)
echo   make test          - Run tests
echo   make install       - Install dependencies
echo   make docker-build  - Build Docker image
echo   make docker-run    - Run Docker container
echo.
goto end

:end
