@echo off
title ROAD-WATCH SETUP
echo =========================================
echo        ROAD-WATCH SETUP SCRIPT
echo =========================================

set BACKEND_DIR=backend
set VENV_DIR=%BACKEND_DIR%\.venv

REM --- CHECK PYTHON ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH.
    pause
    exit /b
)

echo.
echo === Checking backend folder ===
if not exist "%BACKEND_DIR%" (
    echo [ERROR] Backend folder "%BACKEND_DIR%" does not exist!
    pause
    exit /b
)

echo.
echo === Checking for virtual environment ===
if not exist "%VENV_DIR%" (
    echo No venv found. Creating one...
    python -m venv "%VENV_DIR%"
) else (
    echo venv already exists.
)

echo.
echo === Activating venv ===
call "%VENV_DIR%\Scripts\activate"

echo.
echo === Installing requirements ===
if exist "%BACKEND_DIR%\requirements.txt" (
    pip install -r "%BACKEND_DIR%\requirements.txt"
) else (
    echo [WARNING] requirements.txt not found. Skipping installation.
)

echo.
echo Setup completed successfully!
pause
