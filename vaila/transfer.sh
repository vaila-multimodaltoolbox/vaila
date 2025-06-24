#!/bin/bash
# === Interactive SCP/RSYNC Folder Transfer Script (NO SENSITIVE DATA) ===
# === This script is used to transfer a folder to a remote server using SSH ===
# === It prompts the user for the remote username, host, port, local directory, and remote directory ===
# === It then uses either scp or rsync command to transfer the folder to the remote server ===
# === It then prints a message to the console indicating that the transfer is complete ===
# === It then pauses the script so the user can see the message ===
# === It then exits the script ===
# === This script is used to transfer a folder to a remote server using SSH ===
# === Author: Paulo Santiago
# === Date: 2025-06-24
# === Updated: 2025-06-24
# === Contact: paulosantiago@usp.br
# === Version: 0.0.2
# === Description: This script is used to transfer a folder to a remote server using SSH
# === It prompts the user for the remote username, host, port, local directory, and remote directory
# === It then uses either scp or rsync command to transfer the folder to the remote server
# === It then prints a message to the console indicating that the transfer is complete

# No defaults for sensitive information
DEF_REMOTE_USER=""
DEF_REMOTE_HOST=""
DEF_REMOTE_PORT="22"
DEF_LOCAL_DIR="."
DEF_REMOTE_DIR=""

# Prompt user for parameters (no defaults except for port and local dir)
read -p "Enter remote username: " REMOTE_USER
read -p "Enter remote host (IP or hostname): " REMOTE_HOST
read -p "Enter SSH port [22]: " REMOTE_PORT
read -p "Enter FULL path to local folder [.]: " LOCAL_DIR
read -p "Enter FULL path to destination on server: " REMOTE_DIR

# Set defaults if empty
if [ -z "$REMOTE_PORT" ]; then
    REMOTE_PORT="$DEF_REMOTE_PORT"
fi

if [ -z "$LOCAL_DIR" ]; then
    LOCAL_DIR="$DEF_LOCAL_DIR"
fi

# Prompt user to choose transfer method
echo
echo "Choose transfer method:"
echo "1) SCP (Simple Copy Protocol)"
echo "2) RSYNC (Remote Synchronization - recommended for large files)"
read -p "Enter choice [1/2]: " TRANSFER_METHOD

# Set default to SCP if no choice made
if [ -z "$TRANSFER_METHOD" ]; then
    TRANSFER_METHOD="1"
fi

echo
echo "============================================"
echo "Transferring: $LOCAL_DIR"
echo "To: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
echo "Using SSH port: $REMOTE_PORT"
echo "Method: $([ "$TRANSFER_METHOD" = "2" ] && echo "RSYNC" || echo "SCP")"
echo "============================================"
echo

# Execute the transfer command based on user choice
if [ "$TRANSFER_METHOD" = "2" ]; then
    # Use rsync with -avhP flags
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose
    # -h: human-readable output
    # -P: progress bar and partial transfer support
    rsync -avhP -e "ssh -p $REMOTE_PORT" "$LOCAL_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"
else
    # Use scp (default)
    scp -P "$REMOTE_PORT" -r "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
fi

if [ $? -eq 0 ]; then
    echo
    echo "Transfer completed successfully!"
else
    echo
    echo "Transfer failed!"
fi

# Pause to show the message (equivalent to Windows pause)
read -p "Press Enter to continue..."
