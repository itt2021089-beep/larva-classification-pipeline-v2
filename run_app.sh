#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d venv ]; then
  echo "Creating a virtual environment (one time)..."
  python3 -m venv venv
  venv/bin/python -m pip install --upgrade pip
  venv/bin/python -m pip install -r requirements-app.txt
fi
echo
echo "Starting Safe Zone AI - a browser tab will open shortly."
venv/bin/python -m streamlit run v2/app.py
