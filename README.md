# Train Detector

Train Detector is a Python-based audio monitoring tool that listens for train-related sounds from a microphone, classifies them using an audio model, and saves detections when a train horn or stopping pattern is detected.

It is designed for use on a Raspberry Pi or similar low-power device, but it can also run on a desktop machine with a compatible microphone.

## Features

- Live microphone monitoring
- Train horn and train-stopping detection
- Rolling audio buffer for clip capture around events
- Local file storage and optional Supabase storage
- Optional ntfy push notifications for train approaching alerts
- Config reloads automatically when settings change

## Project Overview

The detector pipeline is built around the following pieces:

- [train_detector.py](train_detector.py) - Main entry point and runtime setup
- [detector.py](detector.py) - Audio processing and detection logic
- [classifier.py](classifier.py) - Audio classifier wrapper
- [detector.conf](detector.conf) - Runtime configuration
- [storage/](storage) - Storage backends for local files and Supabase

## Requirements

- Python 3.10+
- A working microphone input device
- System audio libraries such as PortAudio and libsndfile

The repository includes a setup script that installs the required dependencies and downloads the model weights.

## Installation

Run the setup script once:

```bash
bash setup.sh
```

This will:

- create a Python virtual environment
- install system audio dependencies
- install Python packages from [requirements.txt](requirements.txt)
- download the classifier model weights

## Configuration

Edit [detector.conf](detector.conf) to tune the detector behavior.

Common settings include:

- `DEVICE` - microphone device name
- `SAMPLE_RATE` - audio sample rate
- `CLIP_SECONDS` - amount of audio analyzed per classification window
- `CLASSIFIER_THRESHOLD` - score needed to trigger a detection
- `MIN_RMS` - quiet-signal cutoff
- `USE_FILE_STORAGE` - enable local clip saving

You can also configure optional environment variables:

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_KEY="your-anon-or-service-role-key"
export NTFY_TOPIC="your-topic-name"
```

## Running the Detector

Start the detector in the background:

```bash
bash starter.sh
```

Stop it later with:

```bash
bash stopper.sh
```

For debugging in the foreground, you can run:

```bash
python train_detector.py
```

## Storage Options

The project supports:

- Local file storage for audio clips and detection logs
- Supabase storage for cloud-backed uploads

If `USE_FILE_STORAGE` is enabled, detections will be written to local files. Supabase uploads are used when the required environment variables are present.

## Notes

- The detector reloads configuration changes automatically while running.
- Logs are written to `train_log.txt` when started with the background script.
- The detector uses a rolling audio buffer so it can save a short clip around each detection.
