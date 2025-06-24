@echo off
REM === Interactive SCP Folder Transfer Script (NO SENSITIVE DATA) ===
REM === This script is used to transfer a folder to a remote server using SSH ===
REM === It prompts the user for the remote username, host, port, local directory, and remote directory ===
REM === It then uses the scp command to transfer the folder to the remote server ===
REM === It then prints a message to the console indicating that the transfer is complete ===
REM === It then pauses the script so the user can see the message ===
REM === It then exits the script ===
REM === This script is used to transfer a folder to a remote server using SSH ===
REM === Author: Paulo Santiago
REM === Date: 2025-06-24
REM === Updated: 2025-06-24
REM === Contact: paulosantiago@usp.br
REM === Version: 0.0.1
REM === Description: This script is used to transfer a folder to a remote server using SSH
REM === It prompts the user for the remote username, host, port, local directory, and remote directory
REM === It then uses the scp command to transfer the folder to the remote server
REM === It then prints a message to the console indicating that the transfer is complete
SETLOCAL ENABLEDELAYEDEXPANSION

REM No defaults for sensitive information
set "DEF_REMOTE_USER="
set "DEF_REMOTE_HOST="
set "DEF_REMOTE_PORT=22"
set "DEF_LOCAL_DIR=."
set "DEF_REMOTE_DIR="

REM Prompt user for parameters (no defaults except for port and local dir)
set /p REMOTE_USER=Enter remote username: 
set /p REMOTE_HOST=Enter remote host (IP or hostname): 
set /p REMOTE_PORT=Enter SSH port [22]: 
if "!REMOTE_PORT!"=="" set REMOTE_PORT=%DEF_REMOTE_PORT%
set /p LOCAL_DIR=Enter FULL path to local folder [.]: 
if "!LOCAL_DIR!"=="" set LOCAL_DIR=%DEF_LOCAL_DIR%
set /p REMOTE_DIR=Enter FULL path to destination on server: 

echo.
echo ============================================
echo Transferring: %LOCAL_DIR%
echo To: %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%
echo Using SSH port: %REMOTE_PORT%
echo ============================================
echo.

REM Execute the recursive SCP command (password will be asked)
scp -P %REMOTE_PORT% -r "%LOCAL_DIR%" %REMOTE_USER%@%REMOTE_HOST%:"%REMOTE_DIR%"

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo Transfer completed successfully!
) ELSE (
    echo.
    echo Transfer failed!
)

pause
ENDLOCAL
