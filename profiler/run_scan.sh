#!/bin/bash

CONFIG_FILE="/var/lib/kubelet/config.yaml"
BACKUP_FILE="/var/lib/kubelet/config.yaml.bak"

if [ ! -f "$BACKUP_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo "Backup created: $BACKUP_FILE"
else
    echo "Backup already exists: $BACKUP_FILE"
fi

if grep -q "containerLogMaxSize" "$CONFIG_FILE"; then
    echo "containerLogMaxSize is already present. No changes made."
else
    echo "Adding containerLogMaxSize configuration."
    sed -i '/^kind: KubeletConfiguration/a\containerLogMaxSize: "100Mi"\ncontainerLogMaxFiles: 5' "$CONFIG_FILE"
    echo "Added containerLogMaxSize: \"100Mi\" and containerLogMaxFiles: 5 to the configuration."
fi

echo "Restarting kubelet"
if systemctl restart kubelet; then
    echo "kubelet restarted."
else
    echo "Failed to restart kubelet."
fi

# locust --headless -f locustfile-bi.py -u 10 -r 10 -H http://localhost:8080 -t 5
# locust --headless -f locustfile-hr.py -u 50 -r 100 -H http://localhost:8080 -t 10
