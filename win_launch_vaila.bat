@echo off
REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda is not available. Please ensure Conda is installed and available in your PATH.
    pause
    exit /b
)

REM Activate the Conda environment named "vaila"
call conda activate vaila

REM Run the vailá Python script
python "C:\vaila_programs\vaila\vaila.py"

REM Pause to keep the terminal open
pause
