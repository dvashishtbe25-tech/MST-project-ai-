@echo off
rem ------------------------------------------------------------------
rem  Double-click launcher for the churn model training.
rem  First run: creates a private Python environment (.venv) and installs
rem  the packages from requirements.txt. Later runs reuse it.
rem  Options are passed on to train.py, e.g.  train.bat --quick
rem ------------------------------------------------------------------
setlocal
cd /d "%~dp0"

rem 1. Is Python installed?
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found on this computer.
    echo Install Python 3.11 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during installation, then run this file again.
    goto :end
)

rem 2. Create the virtual environment and install the packages (first run only).
rem    The marker file .venv\installed.txt is written only after a successful install,
rem    so an interrupted install is retried next time.
if not exist ".venv\installed.txt" (
    if not exist ".venv\Scripts\python.exe" (
        echo Creating the virtual environment in .venv ...
        python -m venv .venv
        if errorlevel 1 (
            echo Could not create the virtual environment.
            goto :end
        )
    )
    echo Installing the required packages - this can take a few minutes the first time ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Installing the packages failed. Check your internet connection and try again.
        goto :end
    )
    echo installed > ".venv\installed.txt"
)

rem 3. Train the model (any options, such as --quick, are passed on).
".venv\Scripts\python.exe" train.py %*

:end
echo.
pause
