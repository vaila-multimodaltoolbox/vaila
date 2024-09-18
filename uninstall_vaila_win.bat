@echo off
REM Script: uninstall_vaila_win.bat
REM Description: Uninstalls the vaila - Multimodal Toolbox from Windows,
REM              including removing the Conda environment and program files.
REM              Cannot remove the Windows Terminal profile automatically.

echo Starting vaila uninstallation on Windows...

REM Ensure the script is running as administrator
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo This script requires administrative privileges. Please run as administrator.
    pause
    exit /b
)

REM Check if conda is available
where conda >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Conda is not installed or not in PATH. Cannot proceed with uninstallation.
    pause
    exit /b
)

REM Activate Conda base environment
CALL conda activate base

REM Remove the 'vaila' Conda environment if it exists
echo Checking for 'vaila' Conda environment...
conda env list | findstr /i "^vaila" >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo Removing 'vaila' Conda environment...
    conda remove --name vaila --all -y
    IF %ERRORLEVEL% EQU 0 (
        echo 'vaila' environment removed successfully.
    ) ELSE (
        echo Failed to remove 'vaila' environment.
    )
) ELSE (
    echo 'vaila' environment does not exist. Skipping environment removal.
)

REM Define user paths
SET USER_HOME=%USERPROFILE%
SET VAILA_HOME=%USER_HOME%\vaila

REM Remove the vaila directory from the user's home directory
IF EXIST "%VAILA_HOME%" (
    echo Removing vaila directory from user's home directory...
    rmdir /S /Q "%VAILA_HOME%"
    IF %ERRORLEVEL% EQU 0 (
        echo vaila directory removed successfully.
    ) ELSE (
        echo Failed to remove vaila directory.
    )
) ELSE (
    echo vaila directory not found in user's home directory. Skipping removal.
)

REM Remove desktop shortcut if it exists
SET SHORTCUT_PATH=%USER_HOME%\Desktop\vaila.lnk
IF EXIST "%SHORTCUT_PATH%" (
    echo Removing desktop shortcut...
    del "%SHORTCUT_PATH%" /Q
    IF %ERRORLEVEL% EQU 0 (
        echo Desktop shortcut removed successfully.
    ) ELSE (
        echo Failed to remove desktop shortcut.
    )
) ELSE (
    echo Desktop shortcut not found. Skipping removal.
)

echo Uninstallation completed. vaila has been removed from your system.
echo Please remove the vaila profile from Windows Terminal settings manually if it was added.
pause

