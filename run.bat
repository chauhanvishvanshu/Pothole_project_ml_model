@echo off
title ROAD-WATCH RUN
echo =========================================
echo        ROAD-WATCH RUN SCRIPT
echo =========================================

set BACKEND_DIR=backend
set FRONTEND_DIR=frontend
set VENV_DIR=%BACKEND_DIR%\.venv

REM --- CHECK PYTHON ---
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    pause
    exit /b
)

echo.
echo === Checking venv ===
if not exist "%VENV_DIR%" (
    echo [ERROR] venv does not exist. Run setup.bat first.
    pause
    exit /b
)

echo.
echo === Activating venv ===
call "%VENV_DIR%\Scripts\activate"

echo.
echo === Starting backend ===
if not exist "%BACKEND_DIR%\app.py" (
    echo [ERROR] app.py not found in backend!
) else (
    pushd "%BACKEND_DIR%"
    start "" cmd /k "python app.py"
    popd
)

echo.
echo === Starting frontend ===
if not exist "%FRONTEND_DIR%\index.html" (
    echo [WARNING] index.html missing in frontend folder!
)

pushd "%FRONTEND_DIR%"
start "" cmd /k "python -m http.server 5500 --directory ."
popd

echo.
echo Opening browser: http://localhost:5500/index.html
start "" http://localhost:5500/index.html

echo.
echo ROAD-WATCH is now running!
pause
