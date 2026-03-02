#!/bin/bash
# HealthMax — EC2 Deployment Script
# Deploys the FastAPI backend to an AWS EC2 Ubuntu 22.04 instance.
#
# Prerequisites (run once before this script):
#   1. Launch EC2 t3.small (Ubuntu 22.04), allocate Elastic IP, open ports 80/443/8000
#   2. Place your private key at ~/.ssh/healthmax-ec2.pem
#   3. Set EC2_HOST below to your Elastic IP
#   4. Run: chmod +x infra/deploy_ec2.sh && ./infra/deploy_ec2.sh
#
# What this script does:
#   - Syncs the codebase to the EC2 instance via rsync
#   - Installs Python dependencies into a venv
#   - Installs and configures nginx as reverse proxy
#   - Starts the FastAPI app via systemd service (restarts on crash)
#
# TODO (collaborator): Update EC2_HOST before running.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — UPDATE THESE
# ---------------------------------------------------------------------------
EC2_HOST="YOUR_ELASTIC_IP_HERE"
EC2_USER="ubuntu"
EC2_KEY="~/.ssh/healthmax-ec2.pem"
REMOTE_APP_DIR="/home/ubuntu/healthmax"
PYTHON_VERSION="python3.11"

# ---------------------------------------------------------------------------
# Step 1: Sync code to EC2
# ---------------------------------------------------------------------------
echo ">>> Syncing codebase to EC2..."
rsync -avz \
  --exclude='.git' \
  --exclude='data/raw/' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  -e "ssh -i ${EC2_KEY}" \
  ./ ${EC2_USER}@${EC2_HOST}:${REMOTE_APP_DIR}/

# ---------------------------------------------------------------------------
# Step 2: Remote setup
# ---------------------------------------------------------------------------
echo ">>> Running remote setup..."
ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST} << 'REMOTE_SCRIPT'
set -euo pipefail

cd /home/ubuntu/healthmax

# Install system packages (first deploy only)
sudo apt-get update -qq
sudo apt-get install -y python3.11 python3.11-venv python3-pip nginx ffmpeg

# Create/update virtualenv
if [ ! -d "venv" ]; then
  python3.11 -m venv venv
fi
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ">>> Dependencies installed."
REMOTE_SCRIPT

# ---------------------------------------------------------------------------
# Step 3: Create systemd service
# ---------------------------------------------------------------------------
echo ">>> Configuring systemd service..."
ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST} bash -s << 'SERVICE_SCRIPT'
sudo tee /etc/systemd/system/healthmax.service > /dev/null << 'EOF'
[Unit]
Description=HealthMax FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/healthmax/backend
Environment="PATH=/home/ubuntu/healthmax/venv/bin"
EnvironmentFile=/home/ubuntu/healthmax/.env
ExecStart=/home/ubuntu/healthmax/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable healthmax
sudo systemctl restart healthmax
echo ">>> HealthMax service started."
SERVICE_SCRIPT

# ---------------------------------------------------------------------------
# Step 4: Configure nginx
# ---------------------------------------------------------------------------
echo ">>> Configuring nginx..."
scp -i ${EC2_KEY} infra/nginx.conf ${EC2_USER}@${EC2_HOST}:/tmp/healthmax_nginx.conf
ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST} bash -s << 'NGINX_SCRIPT'
sudo cp /tmp/healthmax_nginx.conf /etc/nginx/sites-available/healthmax
sudo ln -sf /etc/nginx/sites-available/healthmax /etc/nginx/sites-enabled/healthmax
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
echo ">>> nginx configured."
NGINX_SCRIPT

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
echo ">>> Waiting for service to start..."
sleep 5
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://${EC2_HOST}/health || echo "000")
if [ "$HTTP_STATUS" = "200" ]; then
  echo "✅ HealthMax is live at http://${EC2_HOST}/"
else
  echo "⚠️ Health check returned HTTP ${HTTP_STATUS} — check systemd logs:"
  echo "   ssh -i ${EC2_KEY} ${EC2_USER}@${EC2_HOST} 'sudo journalctl -u healthmax -n 50'"
fi
