#!/bin/bash
# HealthMax EC2 Deployment Script
# Run once on a fresh Ubuntu 22.04 EC2 t3.small instance
# Usage: bash infra/deploy_ec2.sh

set -e

echo "======================================"
echo " HealthMax EC2 Deployment Script"
echo "======================================"

# 1. System update
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y python3 python3-pip python3-venv nginx git ffmpeg libsndfile1

# 2. Clone repo
cd /home/ubuntu
if [ ! -d "HealthMax" ]; then
  git clone https://github.com/Shafin2954/HealthMax.git
fi
cd HealthMax

# 3. Python venv + dependencies
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Create .env from template (fill manually after deployment)
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "[!] .env created. Edit it: nano .env"
fi

# 5. Build FAISS index and train classifier
python data/process_datasets.py

# 6. Configure nginx
sudo cp infra/nginx.conf /etc/nginx/sites-available/healthmax
sudo ln -sf /etc/nginx/sites-available/healthmax /etc/nginx/sites-enabled/healthmax
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx

# 7. Install systemd service
sudo tee /etc/systemd/system/healthmax.service > /dev/null <<EOF
[Unit]
Description=HealthMax FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/HealthMax
ExecStart=/home/ubuntu/HealthMax/venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 2
Restart=always
RestartSec=3
EnvironmentFile=/home/ubuntu/HealthMax/.env

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable healthmax
sudo systemctl start healthmax

echo ""
echo "======================================"
echo " ✅ HealthMax deployed successfully!"
echo " Check status: sudo systemctl status healthmax"
echo " View logs:    sudo journalctl -u healthmax -f"
echo "======================================"
