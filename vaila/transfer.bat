@echo off
REM === Interactive RSYNC Folder Transfer Script (NO SENSITIVE DATA) ===
REM === This script is used to transfer a folder to a remote server using SSH ===
REM === It prompts the user for the remote username, host, port, local directory, and remote directory ===
REM === It then uses rsync command to transfer the folder to the remote server ===
REM === It then prints a message to the console indicating that the transfer is complete ===
REM === It then pauses the script so the user can see the message ===
REM === It then exits the script ===
REM === Author: Paulo Santiago
REM === Date: 2025-06-24
REM === Updated: 2025-01-11
REM === Contact: paulosantiago@usp.br
REM === Version: 0.1.0
REM === Description: This script is used to transfer a folder to a remote server using SSH
REM === It prompts the user for the remote username, host, port, local directory, and remote directory
REM === It then uses rsync command to transfer the folder to the remote server
REM === It then prints a message to the console indicating that the transfer is complete
SETLOCAL ENABLEDELAYEDEXPANSION

REM Change to user's Downloads directory for safety
cd /d "%USERPROFILE%\Downloads"

echo ============================================
echo RSYNC Folder Transfer Tool
echo ============================================
echo Current directory: %CD%
echo.

REM Check if rsync is available
where rsync >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: rsync is not found in PATH.
    echo.
    echo rsync is required for this script to work.
    echo Please install rsync for Windows:
    echo   - Option 1: Install via WSL (Windows Subsystem for Linux)
    echo   - Option 2: Install via Cygwin
    echo   - Option 3: Install via Git for Windows (includes rsync)
    echo   - Option 4: Download from https://www.itefix.net/cwrsync
    echo.
    pause
    exit /b 1
)

REM Ask user if they want to use Downloads directory or choose another
set /p USE_DOWNLOADS=Do you want to use Downloads directory? (Y/N) [Y]: 
if /i "!USE_DOWNLOADS!"=="" set USE_DOWNLOADS=Y
if /i "!USE_DOWNLOADS!"=="Y" (
    set "DEF_LOCAL_DIR=%CD%"
    echo Using Downloads directory: %DEF_LOCAL_DIR%
) else (
    set "DEF_LOCAL_DIR=."
    echo You can specify a different directory below.
)

echo.

REM No defaults for sensitive information
set "DEF_REMOTE_USER="
set "DEF_REMOTE_HOST="
set "DEF_REMOTE_PORT=22"
set "DEF_REMOTE_DIR="

REM Prompt user for parameters (no defaults except for port and local dir)
set /p REMOTE_USER=Enter remote username: 
set /p REMOTE_HOST=Enter remote host (IP or hostname): 
set /p REMOTE_PORT=Enter SSH port [22]: 
if "!REMOTE_PORT!"=="" set REMOTE_PORT=%DEF_REMOTE_PORT%
set /p LOCAL_DIR=Enter FULL path to local folder [!DEF_LOCAL_DIR%]: 
if "!LOCAL_DIR!"=="" set LOCAL_DIR=%DEF_LOCAL_DIR%
set /p REMOTE_DIR=Enter FULL path to destination on server: 

echo.
echo ============================================
echo Transferring: %LOCAL_DIR%
echo To: %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%
echo Using SSH port: %REMOTE_PORT%
echo Method: RSYNC
echo ============================================
echo.

REM Execute rsync command with -avhP flags
REM -a: archive mode (preserves permissions, timestamps, etc.)
REM -v: verbose
REM -h: human-readable output
REM -P: progress bar and partial transfer support
rsync -avhP -e "ssh -p %REMOTE_PORT%" "%LOCAL_DIR%/" "%REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/"

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo Transfer completed successfully!
) ELSE (
    echo.
    echo Transfer failed!
)

pause
ENDLOCAL
