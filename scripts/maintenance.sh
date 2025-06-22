#!/bin/bash

# Daily maintenance script for txClassify droplet
# Add to crontab: 0 2 * * * /opt/txClassify/scripts/maintenance.sh >> /var/log/txclassify-maintenance.log 2>&1

set -e

echo "=== txClassify Daily Maintenance - $(date) ==="

cd /opt/txClassify

echo "=== Before Cleanup ==="
df -h | head -3

# Get current disk usage
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
echo "Current disk usage: ${DISK_USAGE}%"

# Clean up if usage > 60%
if [ "$DISK_USAGE" -gt 60 ]; then
    echo "=== Cleaning Docker System ==="
    docker system prune -af --volumes
    
    echo "=== Cleaning System Logs ==="
    journalctl --vacuum-time=7d
    
    echo "=== Cleaning APT Cache ==="
    apt-get clean
    apt-get autoclean
    apt-get autoremove -y
    
    echo "=== Cleaning Temp Files ==="
    find /tmp -type f -atime +7 -delete 2>/dev/null || true
    find /var/tmp -type f -atime +7 -delete 2>/dev/null || true
    
    echo "=== After Cleanup ==="
    df -h | head -3
    
    NEW_DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    FREED=$((DISK_USAGE - NEW_DISK_USAGE))
    echo "Freed ${FREED}% disk space"
else
    echo "Disk usage is acceptable (${DISK_USAGE}%), skipping cleanup"
fi

echo "=== Container Health Check ==="
docker compose ps

echo "=== Memory Usage ==="
free -h

echo "=== Maintenance Complete - $(date) ==="
echo "==========================================" 