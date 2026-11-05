@echo off
cd /d "%~dp0"
if not defined ROBOT_PYTHON set "ROBOT_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%ROBOT_PYTHON%" set "ROBOT_PYTHON=python"
"%ROBOT_PYTHON%" run_keyboard.py --step-demo
if errorlevel 1 pause
