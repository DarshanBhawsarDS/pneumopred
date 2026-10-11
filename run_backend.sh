#!/bin/bash
echo "==================================================="
echo "  Starting PneumoPred Backend on http://127.0.0.1:8000"
echo "  Swagger Docs: http://127.0.0.1:8000/docs"
echo "==================================================="

cd "$(dirname "$0")"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --reload
