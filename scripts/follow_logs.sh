#!/bin/bash
# Description: Follows logs in real-time for a specified Docker Compose service on the remote server.
# Usage: ./scripts/follow_logs.sh [service_name]
#   Example: ./scripts/follow_logs.sh backend
#   Example: ./scripts/follow_logs.sh caddy
#   Example: ./scripts/follow_logs.sh (defaults to backend)

# --- Configuration (Update if your server details change) ---
DROPLET_IP="146.190.163.238"
DROPLET_USER="root"
# IMPORTANT: Assumes the private key is always accessible at this path relative to the user running the script
SSH_KEY_PATH="~/.ssh/id_ed25519_new"
APP_DIR="/opt/txClassify"
# --- End Configuration ---

# Get argument or use default
SERVICE_NAME=${1:-backend}

echo "Following logs for '$SERVICE_NAME' service from $DROPLET_USER@$DROPLET_IP... (Press Ctrl+C to stop)"

# Construct the remote command
REMOTE_COMMAND="cd $APP_DIR && docker compose logs -f $SERVICE_NAME"

# Execute via SSH, use -t to allocate a pseudo-terminal which is often needed for `docker compose logs -f`
# Note: This will still prompt for the SSH key passphrase if the key is not in the ssh-agent
ssh -i "$SSH_KEY_PATH" -t "$DROPLET_USER@$DROPLET_IP" "$REMOTE_COMMAND"

# Note: The script might not reach here easily if Ctrl+C is handled remotely.
echo "--- Log following stopped ---"
exit 0 