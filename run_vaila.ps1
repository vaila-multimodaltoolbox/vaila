# Run vaila using uv
Set-Location "C:\Users\paulo\Preto\vaila"
& uv run --no-sync "C:\Users\paulo\Preto\vaila\vaila.py"
# Keep terminal open after execution
Write-Host ""
Write-Host "Program finished. Press Enter to close this window..." -ForegroundColor Yellow
Read-Host
