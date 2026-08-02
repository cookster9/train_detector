# Train Detector

Train Detector is a Python-based audio monitoring tool that listens for train-related sounds from a microphone, classifies them using an audio model, and saves detections when a train horn or stopping pattern is detected.

It is designed for use on a Raspberry Pi or similar low-power device, but it can also run on a desktop machine with a compatible microphone.

## Implementation for <a href="https://github.com/cookster9/train-detection-dashboard>Train Detector Dashboard</a>
I was tired of leaving the house, or coming back from the house, and a train was crossing the road near my house. I wanted to see if I could give myself a notification for whether the train is there. So I set this up to listen for train noises and write to a database if it's detected.

I set up this project on a Raspberry Pi 5. I attached a cheap usb lavalier mic to the Pi and threaded it out the window:
<img width="4284" height="5712" alt="IMG_3388" src="https://github.com/user-attachments/assets/e354814f-51a3-4c77-9bbf-e495ed207dc4" />

It is a quiet corner of the house so the train horn noise comes through clear. I'm only a few blocks away from the train crossing that I want to track.

The audio classifier is here: https://github.com/qiuqiangkong/panns_inference
It can take an audio clip and list the categories of sounds that it can hear with a confidence score. "Train horn" happens to be a sound category that it supports, so we can simply look for whether the score for "Train horn" is over a certain threshold to decide if a train is passing. Because it pulls from a finite set, and only outputs scores for each category, it is very performant and can run on something like a Raspberry Pi. This is in contrast to something like generative AI which is much more performance intensive.

The trian near my house stops and blocks the crossing relatively often, which is extra frustrating. So I have also tried to implement a "train stopping" event looking for another category for "train wheels screeching". If that happens around the time the train horn is detected, you could infer that the train is stopping or at least passing slowly.

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
