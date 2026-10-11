@echo off
title PneumoPred Backend (FastAPI)
echo ===================================================
echo   Starting PneumoPred Backend on http://127.0.0.1:8000
echo   Swagger Docs available at http://127.0.0.1:8000/docs
echo ===================================================
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo Error: Python virtual environment not found in .\venv.
    echo Please create it with: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
