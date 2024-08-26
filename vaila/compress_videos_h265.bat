@echo off
if "%~2"=="" (
    echo Usage: %0 input_file output_file
    exit /b 1
)

set INPUT_FILE=%1
set OUTPUT_FILE=%2

ffmpeg -y -i "%INPUT_FILE%" -c:v libx265 -preset medium -crf 23 "%OUTPUT_FILE%"

if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to compress video
    exit /b 1
)

echo Compression completed successfully: %OUTPUT_FILE%
