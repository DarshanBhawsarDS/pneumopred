#!/bin/bash
echo "==================================================="
echo "  Starting PneumoPred UI via Docker (Vite)"
echo "  Accessible at http://localhost:5173"
echo "==================================================="

cd "$(dirname "$0")"

docker compose up --build frontend
