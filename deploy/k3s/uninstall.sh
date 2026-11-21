#!/usr/bin/env bash
# Cleanly remove k3s from a laptop afterwards (agent or server).
set -euo pipefail
if [ -x /usr/local/bin/k3s-agent-uninstall.sh ]; then sudo /usr/local/bin/k3s-agent-uninstall.sh; fi
if [ -x /usr/local/bin/k3s-uninstall.sh ]; then sudo /usr/local/bin/k3s-uninstall.sh; fi
echo "k3s removed. Optional: sudo apt remove --purge nvidia-container-toolkit; restore %UserProfile%\\.wslconfig; in PowerShell (admin): deploy\\k3s\\firewall.ps1 -Remove"
