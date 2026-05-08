#!/bin/bash
# setup.sh — run once on the Pi to install dependencies
# Usage: bash setup.sh

set -e
echo "=== Train detector setup ==="

# System deps for sounddevice and audio
sudo apt-get update -q
sudo apt-get install -y libportaudio2 portaudio19-dev libsndfile1 ffmpeg

# Python deps
pip install --upgrade pip
pip install sounddevice numpy scipy

# PANNs inference — the PyPI package has ARM issues, install direct from source
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install librosa

# Clone and install panns_inference manually (avoids broken PyPI ARM wheel)
if [ ! -d "panns_inference" ]; then
    git clone https://github.com/qiuqiangkong/panns_inference.git
fi
pip install -e panns_inference/

echo ""
echo "=== Downloading PANNs model weights (CNN14, ~200MB) ==="
python - <<'EOF'
from panns_inference import AudioTagging
# This triggers the weight download on first run
at = AudioTagging(checkpoint_path=None, device='cpu')
print("Model downloaded and ready.")
EOF

echo ""
echo "=== Setup complete ==="
