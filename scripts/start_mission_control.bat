@echo off
REM Launch the Harvy Mission Control bridge server on Windows.

setlocal
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

call .venv\Scripts\activate.bat
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

python -m mission_control.server
