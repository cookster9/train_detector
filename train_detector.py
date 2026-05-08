import sys
import os
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector.conf")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

def load_config():
    cfg = {}
    try:
        with open(CONFIG_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.split("#")[0].strip()
                try:
                    cfg[key] = eval(val)
                except Exception:
                    pass
    except FileNotFoundError:
        print(f"[config] {CONFIG_FILE} not found", flush=True)
    return cfg

_cfg            = load_config()
_last_cfg_mtime = 0.0

def get_cfg():
    global _cfg, _last_cfg_mtime
    try:
        mtime = os.path.getmtime(CONFIG_FILE)
    except OSError:
        return _cfg
    if mtime != _last_cfg_mtime:
        _cfg = load_config()
        _last_cfg_mtime = mtime
        print(f"[config] reloaded at {time.strftime('%H:%M:%S')}", flush=True)
    return _cfg


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'panns_inference'))

print("Loading PANNs classifier...", flush=True)
try:
    from panns_inference.inference import AudioTagging
    import librosa as _librosa
    _classifier = AudioTagging(checkpoint_path=None, device='cpu')
    print("Classifier ready.", flush=True)
except Exception as e:
    print(f"[classifier] Failed to load: {e}", flush=True)
    _classifier = None

TRAIN_HORN_CLASS = 331
VEHICLE_CLASS = 300
AIRCRAFT_CLASS = 335
AIR_CONDITIONER_CLASS = 413


def classify(audio_float32, sr):
    """
    Run PANNs on a float32 [-1,1] audio array.
    Returns (all_scores, labels) tuple or (None, None, None) on failure.
    """
    if _classifier is None:
        return None, None
    try:
        # Normalize audio: amplify to make quiet sounds more detectable
        # Try 1.0 (no amplification) first to establish baseline
        audio_float32 = np.clip(audio_float32 * 1.0, -1.0, 1.0)
        
        print(f"[classify] peak={np.max(np.abs(audio_float32)):.4f} "
            f"rms={np.sqrt(np.mean(audio_float32**2)):.4f}", flush=True)

        # Mono check
        if audio_float32.ndim > 1:
            audio_float32 = audio_float32[:, 0]

        if sr != 32000:
            audio_32k = _librosa.resample(audio_float32, orig_sr=sr, target_sr=32000)
        else:
            audio_32k = audio_float32

        out, _ = _classifier.inference(audio_32k[np.newaxis, :])        
        
        # Get all scores and labels for debugging
        all_scores = out[0]
        labels = _classifier.labels
        
        return all_scores, labels

    except Exception as e:
        print(f"[classifier] error: {e}", flush=True)
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# STATE
# ──────────────────────────────────────────────────────────────────────────────

_cfg         = get_cfg()
_buf_max     = int(_cfg.get("BUFFER_SECONDS", 60) / _cfg.get("BLOCK_DURATION", 1.0))
audio_buffer = deque(maxlen=_buf_max)

last_classify_time = 0.0
last_save_time     = 0.0
classifier_busy    = False

os.makedirs("clips", exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────────

def save_clip(audio, scores, sr, timestamp, label):
    filename = f"clips/{label}_{timestamp}.wav"
    write(filename, sr, audio)
    duration  = len(audio) / sr
    score_str = "  ".join(f"{s:.3f}" for s in scores)
    with open("detections.txt", "a") as f:
        f.write(f"{timestamp}  hits={len(scores)}  scores=[{score_str}]  "
                f"duration={duration:.1f}s  label={label}\n")
    print(f"\n🚂 SAVED: {filename}  "
          f"({len(scores)} hits  scores=[{score_str}]  {duration:.1f}s)",
          flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLASSIFY WORKER
# ──────────────────────────────────────────────────────────────────────────────

def classify_worker(audio_snap, cfg_snap, snap_time):
    global classifier_busy, last_save_time, audio_buffer

    try:
        sr        = cfg_snap["SAMPLE_RATE"]
        threshold = cfg_snap["CLASSIFIER_THRESHOLD"]
        threshold_faraway = cfg_snap["CLASSIFIER_THRESHOLD_FARAWAY"]
        threshold_interference = cfg_snap["CLASSIFIER_THRESHOLD_INTERFERENCE"]
        cooldown  = cfg_snap["COOLDOWN"]
        save_secs = cfg_snap["SAVE_SECONDS"]

        all_scores, labels = classify(audio_snap, sr)

        score = all_scores[TRAIN_HORN_CLASS] if all_scores is not None else None

        if score is None:
            return

        now = time.time()

        if cfg_snap.get("DEBUG"):
            print(f"[classifier] score={score:.4f}  threshold={threshold}  ", flush=True)
            
            # Print top 10 classifications
            if all_scores is not None and labels is not None:
                top_indices = np.argsort(all_scores)[::-1][:10]
                print("[classifier] ------ Top classifications ------", flush=True)
                for idx in top_indices:
                    if idx < len(labels):
                        class_score = float(all_scores[idx])
                        if class_score > 0.001:  # Only print if score > 0.001
                            print(f"  {labels[idx]}: {class_score:.3f}", flush=True)

        # my air conditioner runs constantly and can mask train horns, so if we detect it running along with vehicles and aircraft, we treat it as an interference case even if the train horn score is low
        # classifier thinks it is a vehicle sound, but it is really the AC, so we check for both and if they are present with a low train horn score, we label it as interference instead of faraway or miss
        ac_is_running = (all_scores[VEHICLE_CLASS] >= 0.3)

        # ── Record hit if above threshold ─────────────────────────────────────
        if score >= threshold:
            print(f"  ✓ hit  score={score:.4f}  ", flush=True)
            label = "train_close"
        elif score >= threshold_faraway:
            print(f"  ⚠️  faraway  score={score:.4f}  ", flush=True) 
            label = "train_faraway"
        elif ac_is_running and score >= threshold_interference:
            print(f"  ⚠️  interference  score={score:.4f}  ", flush=True) 
            label = "train_interference"
        else:
            return
        

        if (now - last_save_time) < cooldown:
            if cfg_snap.get("DEBUG"):
                print(f"[classifier] cooldown active — skipping save", flush=True)
            return

        # ── Save clip centered on the hit window ──────────────────────────────
        last_save_time = now

        # Grab SAVE_SECONDS of audio from the rolling buffer
        buf_snap   = list(audio_buffer)
        save_blocks = int(save_secs / cfg_snap["BLOCK_DURATION"])
        save_blocks = min(save_blocks, len(buf_snap))
        if save_blocks > 0:
            # audio_save  = np.concatenate([a for _, a in buf_snap[-save_blocks:]] if isinstance(buf_snap[0], tuple) else buf_snap[-save_blocks:]).astype(np.float32)
            audio_save  = np.concatenate(buf_snap[-save_blocks:]).astype(np.float32)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            scores = [score]
            threading.Thread(
                target=save_clip,
                args=(audio_save, scores, sr, timestamp, label),
                daemon=True
            ).start()

    finally:
        classifier_busy = False


# ──────────────────────────────────────────────────────────────────────────────
# CALLBACK
# ──────────────────────────────────────────────────────────────────────────────

def callback(indata, frames, time_info, status):
    global last_classify_time, classifier_busy, audio_buffer

    if status:
        print(f"[stream status] {status}", flush=True)

    cfg   = get_cfg()
    audio = indata[:, 0].copy()

    # Resize buffer if config changed
    new_max = int(cfg["BUFFER_SECONDS"] / cfg["BLOCK_DURATION"])
    if audio_buffer.maxlen != new_max:
        audio_buffer = deque(list(audio_buffer)[-new_max:], maxlen=new_max)

    audio_buffer.append(audio)

    # ── Classify every CLASSIFY_EVERY seconds ─────────────────────────────────
    now = time.time()
    if (now - last_classify_time) < cfg["CLASSIFY_EVERY"]:
        return

    if classifier_busy:
        if cfg.get("DEBUG"):
            print("[classifier] still busy — skipping", flush=True)
        return

    # Build snapshot
    clip_blocks = int(cfg["CLIP_SECONDS"] / cfg["BLOCK_DURATION"])
    buf_list    = list(audio_buffer)
    clip_blocks = min(clip_blocks, len(buf_list))
    audio_snap  = np.concatenate(buf_list[-clip_blocks:]).astype(np.float32)

    # ── Pre-filter: skip if window is too quiet ───────────────────────────────
    block_size = int(cfg["SAMPLE_RATE"] * cfg["BLOCK_DURATION"])
    block_rms  = [np.sqrt(np.mean(audio_snap[i:i + block_size] ** 2))
                  for i in range(0, len(audio_snap), block_size)]
    window_rms = max(block_rms)

    last_classify_time = now   # always update so we wait full interval on quiet

    if window_rms < cfg["MIN_RMS"]:
        if cfg.get("DEBUG"):
            print(f"window_rms={window_rms:.4f}  [quiet]", flush=True)
        return

    if cfg.get("DEBUG"):
        print(f"window_rms={window_rms:.4f}  classifying...", flush=True)

    classifier_busy = True
    snap_time       = now

    threading.Thread(
        target=classify_worker,
        args=(audio_snap, dict(cfg), snap_time),
        daemon=True
    ).start()


# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────

cfg = get_cfg()
print(f"\nDevice info: {sd.query_devices(cfg['DEVICE'])['name']}")
print(f"Config: {CONFIG_FILE}  (edit and save to update live)")
print(f"clip={cfg['CLIP_SECONDS']}s  "
      f"threshold={cfg['CLASSIFIER_THRESHOLD']}")
print(f"cooldown={cfg['COOLDOWN']}s")
print(f"  rms floor={cfg['MIN_RMS']}  save={cfg['SAVE_SECONDS']}s\n")

try:
    with sd.InputStream(
        device=cfg["DEVICE"],
        channels=1,
        samplerate=cfg["SAMPLE_RATE"],
        blocksize=int(cfg["SAMPLE_RATE"] * cfg["BLOCK_DURATION"]),
        callback=callback,
    ):
        while True:
            time.sleep(1)

except KeyboardInterrupt:
    print("\nShutting down.")