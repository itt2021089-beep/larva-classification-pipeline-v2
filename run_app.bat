@echo off
cd /d "%~dp0"
if not exist venv (
  echo Creating a virtual environment ^(one time^)...
  python -m venv venv
  venv\Scripts\python.exe -m pip install --upgrade pip
  venv\Scripts\python.exe -m pip install -r requirements-app.txt
)
echo.
echo Starting Safe Zone AI - a browser tab will open shortly.
venv\Scripts\python.exe -m streamlit run v2\app.py
pause
