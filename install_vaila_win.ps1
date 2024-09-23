# Define installation path
$vailaProgramPath = "C:\ProgramData\vaila"

# Ensure the directory exists
If (-Not (Test-Path $vailaProgramPath)) {
    Write-Output "Creating directory $vailaProgramPath..."
    New-Item -ItemType Directory -Force -Path $vailaProgramPath
}

# Check if the "vaila" environment already exists
Write-Output "Checking if the 'vaila' Conda environment exists..."
$envExists = conda env list | Select-String -Pattern "^vaila"
If ($envExists) {
    Write-Output "Conda environment 'vaila' already exists. Updating..."
    Try {
        conda env update -n vaila -f "yaml_for_conda_env\vaila_win.yaml" --prune
        Write-Output "'vaila' environment updated successfully."
    } Catch {
        Write-Error "Failed to update 'vaila' environment. Error: $_"
        Exit
    }
} Else {
    Write-Output "Creating Conda environment from vaila_win.yaml..."
    Try {
        conda env create -f "yaml_for_conda_env\vaila_win.yaml"
        Write-Output "'vaila' environment created successfully."
    } Catch {
        Write-Error "Failed to create 'vaila' environment. Error: $_"
        Exit
    }
}

# Activate the 'vaila' environment
& "$condaPath\Scripts\activate.bat" vaila

# Install moviepy using pip
Write-Output "Installing moviepy..."
Try {
    pip install moviepy
    Write-Output "moviepy installed successfully."
} Catch {
    Write-Error "Failed to install moviepy. Error: $_"
}

# Remove ffmpeg installed via Conda, if any
Write-Output "Removing ffmpeg installed via Conda, if any..."
conda remove -n vaila ffmpeg -y

# Check if FFmpeg is installed
If (-Not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Output "Installing FFmpeg via winget..."
    Try {
        winget install --id Gyan.FFmpeg -e --silent
        Write-Output "FFmpeg installed successfully."
    } Catch {
        Write-Warning "Failed to install FFmpeg via winget. Attempting installation via Chocolatey..."
        If (-Not (Get-Command choco -ErrorAction SilentlyContinue)) {
            Write-Warning "Chocolatey is not installed. Installing Chocolatey..."
            Set-ExecutionPolicy Bypass -Scope Process -Force
            [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
            Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
        }
        choco install ffmpeg -y
    }
} Else {
    Write-Output "FFmpeg is already installed."
}

# Copy the vaila program to C:\ProgramData\vaila
Write-Output "Copying vaila program to $vailaProgramPath..."
Copy-Item -Path (Get-Location) -Destination "$vailaProgramPath" -Recurse -Force

# Configure the vaila profile in Windows Terminal
If ($wtInstalled) {
    Write-Output "Configuring the vaila profile in Windows Terminal..."
    $settingsPath = "$wtPath\LocalState\settings.json"
    $settingsBackupPath = "$wtPath\LocalState\settings_backup.json"

    # Backup the current settings.json
    Copy-Item -Path $settingsPath -Destination $settingsBackupPath -Force

    # Load settings.json
    $settingsJson = Get-Content -Path $settingsPath -Raw | ConvertFrom-Json

    # Remove existing vaila profile if it exists
    $existingProfileIndex = $settingsJson.profiles.list.FindIndex({ $_.name -eq "vailá" -or $_.name -eq "vaila" })
    If ($existingProfileIndex -ge 0) {
        Write-Output "Removing existing vaila profile..."
        $settingsJson.profiles.list.RemoveAt($existingProfileIndex)
    }

    # Define the new profile
    $vailaProfile = @{
        name = "vailá"
        commandline = "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; cd '$vailaProgramPath' ; python 'vaila.py'`""
        startingDirectory = "$vailaProgramPath"
        icon = "$vailaProgramPath\docs\images\vaila_ico.png"
        colorScheme = "Vintage"
        guid = "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}"
        hidden = $false
    }

    # Add the profile to settings.json
    If (-Not $settingsJson.profiles.list) {
        $settingsJson.profiles.list = @()
    }
    $settingsJson.profiles.list += $vailaProfile

    # Save the updated settings.json
    $settingsJson | ConvertTo-Json -Depth 100 | Set-Content -Path $settingsPath -Encoding UTF8

    Write-Output "vailá profile added to Windows Terminal successfully."

    # Open settings.json in Notepad for verification
    Write-Output "Opening settings.json in Notepad for verification..."
    notepad "$settingsPath"
} Else {
    # Create a desktop shortcut
    Write-Output "Creating a desktop shortcut for vailá..."
    $shortcutPath = "$env:USERPROFILE\Desktop\vailá.lnk"
    $wshell = New-Object -ComObject WScript.Shell
    $shortcut = $wshell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "pwsh.exe"
    $shortcut.Arguments = "-ExecutionPolicy Bypass -NoExit -Command `"& '$condaPath\shell\condabin\conda-hook.ps1' ; conda activate 'vaila' ; cd '$vailaProgramPath' ; python 'vaila.py'`""
    $shortcut.IconLocation = "$vailaProgramPath\docs\images\vaila_ico.ico"
    $shortcut.WorkingDirectory = "$vailaProgramPath"
    $shortcut.Save()

    Write-Output "Desktop shortcut created at $shortcutPath"
}

Write-Output "Installation and configuration completed successfully!"
Pause

