# Deployment

## First-time Pi setup
1. Clone repo
2. Run `bash setup.sh`
3. Install systemd services (see below)
...

## Systemd services
- `train-detector.service` — runs the app
- `train-deploy.timer` — polls GitHub and redeploys every 5 min

## Useful commands
- `journalctl -u train-detector -f` — live logs
- `sudo systemctl status train-detector` — check app
- `sudo systemctl status train-deploy.timer` — check deploy timer

## deploy.sh
## cat $(pwd)/deploys/train_detector_deploy.sh
#!/bin/bash
set -e

cd /home/USERNAME/train_detector
git stash
git pull origin main

# install any new deps
source /home/USERNAME/train_detector/venv/bin/activate
pip install -r requirements.txt

# restart the service
sudo systemctl restart train-detector

## /etc/systemd/system/train-deploy.timer
[Unit]
Description=Deploy train detector updates

[Timer]
OnBootSec=1m
OnUnitActiveSec=5m   # polls GitHub every 5 minutes

[Install]
WantedBy=timers.target

## /etc/systemd/system/train-deploy.service
[Unit]
Description=Deploy train detector

[Service]
Type=oneshot
User=USERNAME
ExecStart=<path to above deploy.sh>
StandardOutput=journal

## /etc/systemd/system/train-detector.service
[Unit]
Description=Train Detector
After=network.target

[Service]
Type=simple
User=USERNAME
WorkingDirectory=/home/USERNAME/train_detector
ExecStart=/home/USERNAME/train_detector/venv/bin/python train_detector.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target