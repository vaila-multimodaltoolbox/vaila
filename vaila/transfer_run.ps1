# Transfer script - allows interactive password input
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "File Transfer (RSYNC)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Transferring: C:/Users/paulo/Preto/vaila/tests/markerless_2d_analysis" -ForegroundColor Yellow
Write-Host "To: preto@143.107.157.169:/home/preto/Downloads" -ForegroundColor Yellow
Write-Host "SSH Port: 22" -ForegroundColor Yellow
Write-Host ""
Write-Host "You will be prompted for your SSH password." -ForegroundColor Green
Write-Host "Enter your password when prompted (it will not be visible)." -ForegroundColor Green
Write-Host ""
Write-Host "Starting transfer..." -ForegroundColor Cyan
Write-Host ""

# Execute the command with proper argument handling
$cmdArgs = @(
    "-avhP",
    "-e",
    "ssh -p 22",
    "C:/Users/paulo/Preto/vaila/tests/markerless_2d_analysis/",
    "preto@143.107.157.169:/home/preto/Downloads/"
)
& "C:\\ProgramData\\chocolatey\\bin\\rsync.EXE" $cmdArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Transfer completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[FAIL] Transfer failed! (Exit code: $LASTEXITCODE)" -ForegroundColor Red
    Write-Host "Please check your connection and credentials." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
