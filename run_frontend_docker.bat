@echo off
title PneumoPred Frontend (Docker)
echo ===================================================
echo   Starting PneumoPred UI via Docker (Vite)
echo   Accessible at http://localhost:5173
echo ===================================================
cd /d "%~dp0"

docker compose up --build frontend
