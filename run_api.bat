@echo off
cd /d "%~dp0"
venv\Scripts\python.exe -m uvicorn v2.api:app --host 0.0.0.0 --port 8000
pause
