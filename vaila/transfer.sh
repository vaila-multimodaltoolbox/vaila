#!/bin/bash
# === Interactive RSYNC Folder Transfer Script (NO SENSITIVE DATA) ===
# === This script is used to transfer a folder to a remote server using SSH ===
# === It prompts the user for the remote username, host, port, local directory, and remote directory ===
# === It then uses rsync command to transfer the folder to the remote server ===
# === It then prints a message to the console indicating that the transfer is complete ===
# === It then pauses the script so the user can see the message ===
# === It then exits the script ===
# === Author: Paulo Santiago
# === Date: 2025-06-24
# === Updated: 2026-07-01
# === Contact: paulosantiago@usp.br
# === Version: 0.3.67
# === Description: This script is used to transfer a folder to a remote server using SSH
# === It prompts the user for the remote username, host, port, local directory, and remote directory
# === It then uses rsync command to transfer the folder to the remote server
# === It then prints a message to the console indicating that the transfer is complete

# Change to user's Downloads directory for safety
cd "$HOME/Downloads"

echo "============================================"
echo "RSYNC Folder Transfer Tool"
echo "============================================"
echo "Current directory: $(pwd)"
echo

# Ask user if they want to use Downloads directory or choose another
read -p "Do you want to use Downloads directory? (Y/N) [Y]: " USE_DOWNLOADS
USE_DOWNLOADS=${USE_DOWNLOADS:-Y}
USE_DOWNLOADS=$(echo "$USE_DOWNLOADS" | xargs)

if [[ "$USE_DOWNLOADS" =~ ^[Yy]$ ]]; then
    DEF_LOCAL_DIR="$(pwd)"
    echo "Using Downloads directory: $DEF_LOCAL_DIR"
else
    DEF_LOCAL_DIR="."
    echo "You can specify a different directory below."
fi

echo

# No defaults for sensitive information
DEF_REMOTE_USER=""
DEF_REMOTE_HOST=""
DEF_REMOTE_PORT="22"
DEF_REMOTE_DIR=""

# Prompt user for parameters (no defaults except for port and local dir)
read -p "Enter remote username: " REMOTE_USER
read -p "Enter remote host (IP or hostname): " REMOTE_HOST
read -p "Enter SSH port [22]: " REMOTE_PORT
read -p "Enter FULL path to local folder [$DEF_LOCAL_DIR]: " LOCAL_DIR
read -p "Enter FULL path to destination on server: " REMOTE_DIR

# Trim leading/trailing whitespace using xargs
REMOTE_USER=$(echo "$REMOTE_USER" | xargs)
REMOTE_HOST=$(echo "$REMOTE_HOST" | xargs)
REMOTE_PORT=$(echo "$REMOTE_PORT" | xargs)
LOCAL_DIR=$(echo "$LOCAL_DIR" | xargs)
REMOTE_DIR=$(echo "$REMOTE_DIR" | xargs)

# Set defaults if empty
if [ -z "$REMOTE_PORT" ]; then
    REMOTE_PORT="$DEF_REMOTE_PORT"
fi

if [ -z "$LOCAL_DIR" ]; then
    LOCAL_DIR="$DEF_LOCAL_DIR"
fi

# Trim again in case default paths had spaces
LOCAL_DIR=$(echo "$LOCAL_DIR" | xargs)
REMOTE_PORT=$(echo "$REMOTE_PORT" | xargs)

echo
echo "============================================"
echo "Transferring: $LOCAL_DIR"
echo "To: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo "Using SSH port: $REMOTE_PORT"
echo "Method: RSYNC"
echo "============================================"
echo

# Use rsync with -avhP flags
# -a: archive mode (preserves permissions, timestamps, etc.)
# -v: verbose
# -h: human-readable output
# -P: progress bar and partial transfer support
rsync -avhP -e "ssh -p $REMOTE_PORT" "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

if [ $? -eq 0 ]; then
    echo
    echo "Transfer completed successfully!"
else
    echo
    echo "Transfer failed!"
fi

# Pause to show the message (equivalent to Windows pause)
read -p "Press Enter to continue..."
