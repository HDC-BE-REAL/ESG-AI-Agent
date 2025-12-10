#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y curl wget

DEB="scripts/google-chrome-stable_current_amd64.deb"
if [ ! -f "$DEB" ]; then
  wget -O "$DEB" https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
fi

sudo apt install -y ./"$DEB"
google-chrome --version
