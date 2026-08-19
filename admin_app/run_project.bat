@echo off
cd /d "%~dp0\.."
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist admin_app\admin_auth.json (
  echo Please run setup_admin.bat from the project root first.
  pause
  exit /b 1
)
python start_app.py
