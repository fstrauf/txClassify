#!/bin/bash
# Description: Views the last N lines of logs for a specified Docker Compose service on the remote server.
# Usage: ./scripts/view_logs.sh [service_name] [tail_count]
#   Example: ./scripts/view_logs.sh backend 100
#   Example: ./scripts/view_logs.sh caddy
#   Example: ./scripts/view_logs.sh (defaults to backend, 50 lines)

# --- Configuration (Update if your server details change) ---
DROPLET_IP="146.190.163.238"
DROPLET_USER="root"
# IMPORTANT: Assumes the private key is always accessible at this path relative to the user running the script
SSH_KEY_PATH="~/.ssh/id_ed25519_new"
APP_DIR="/opt/txClassify"
# --- End Configuration ---

# Get arguments or use defaults
SERVICE_NAME=${1:-backend}
TAIL_COUNT=${2:-50}

# Input validation (optional but good)
if ! [[ "$TAIL_COUNT" =~ ^[0-9]+$ ]]; then
    echo "Error: Tail count '$TAIL_COUNT' is not a valid number. Using default 50."
    TAIL_COUNT=50
fi

echo "Fetching last $TAIL_COUNT lines for '$SERVICE_NAME' service from $DROPLET_USER@$DROPLET_IP..."

# Construct the remote command
REMOTE_COMMAND="cd $APP_DIR && docker compose logs --tail=$TAIL_COUNT $SERVICE_NAME"

# Execute via SSH
# Note: This will still prompt for the SSH key passphrase if the key is not in the ssh-agent
ssh -i "$SSH_KEY_PATH" "$DROPLET_USER@$DROPLET_IP" "$REMOTE_COMMAND"

# Check SSH exit status (optional)
SSH_EXIT_CODE=$?
if [ $SSH_EXIT_CODE -ne 0 ]; then
    echo "Error: SSH command failed with exit code $SSH_EXIT_CODE."
    exit $SSH_EXIT_CODE
fi

echo "--- Log fetch complete ---"
exit 0 