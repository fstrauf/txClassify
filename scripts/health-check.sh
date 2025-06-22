#!/bin/bash

# Health check script for txClassify deployment
# Usage: ./scripts/health-check.sh

set -e

echo "=== txClassify Health Check ==="
echo "Timestamp: $(date)"
echo

# Check if we're on the droplet or local
if [[ "$HOSTNAME" == *"droplet"* ]] || [[ -d "/opt/txClassify" ]]; then
    # Running on droplet
    PROJECT_DIR="/opt/txClassify"
    IS_LOCAL=false
else
    # Running locally, use SSH
    PROJECT_DIR="/opt/txClassify"
    IS_LOCAL=true
    SSH_CMD="ssh -i ~/.ssh/id_ed25519_droplet root@146.190.163.238"
fi

run_command() {
    if [ "$IS_LOCAL" = true ]; then
        $SSH_CMD "$1"
    else
        eval "$1"
    fi
}

echo "=== Container Status ==="
run_command "cd $PROJECT_DIR && docker compose ps"
echo

echo "=== Health Endpoint Test ==="
if curl -f -s --max-time 10 https://api.expensesorted.com/health > /dev/null; then
    echo "✅ Health endpoint: PASS"
else
    echo "❌ Health endpoint: FAIL"
    echo "=== Recent Backend Logs ==="
    run_command "cd $PROJECT_DIR && docker compose logs --tail=10 backend"
    exit 1
fi

echo "=== Disk Space ==="
run_command "df -h | head -3"
echo

echo "=== Memory Usage ==="
run_command "free -h"
echo

echo "=== Recent Service Logs ==="
echo "Backend (last 5 lines):"
run_command "cd $PROJECT_DIR && docker compose logs --tail=5 backend"
echo

echo "Caddy (last 5 lines):"
run_command "cd $PROJECT_DIR && docker compose logs --tail=5 caddy"
echo

echo "=== Summary ==="
echo "✅ All systems operational"
echo "🌐 API: https://api.expensesorted.com"
echo "�� Status: HEALTHY" 