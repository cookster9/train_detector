"""Train horn detector - main entry point."""
import os
import time
import sounddevice as sd
from dotenv import load_dotenv

from config import ConfigManager
from classifier import AudioClassifier
from detector import TrainDetector
from storage.file_storage import FileStorage
from storage.supabase_storage import SupabaseStorage
from storage.composite_storage import CompositeStorage

# Load environment variables from .env file
load_dotenv()


def main():
    """Main entry point for the train detector."""
    # Initialize components
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "detector.conf"
    )
    
    config = ConfigManager(config_file)
    cfg = config.get()
    
    classifier = AudioClassifier()
    if not classifier.is_ready():
        print("[main] Classifier failed to load. Exiting.", flush=True)
        return
    
    # Initialize storage backends
    backends = []
    
    # Add file storage if enabled
    if cfg.get("USE_FILE_STORAGE"):
        file_storage = FileStorage(clips_dir="clips", log_file="detections.txt")
        backends.append(file_storage)
    
    # Add Supabase storage if credentials available
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if supabase_url and supabase_key:
        supabase_storage = SupabaseStorage(supabase_url, supabase_key)
        backends.append(supabase_storage)
    else:
        print(
            "[main] SUPABASE_URL and/or SUPABASE_KEY not set. "
            "Skipping Supabase storage.",
            flush=True
        )
    
    # Create composite storage (writes to all backends)
    storage = CompositeStorage(backends)
    if not storage.initialize():
        print("[main] Storage initialization failed. Exiting.", flush=True)
        return
    
    detector = TrainDetector(config, classifier, storage)
    
    # Print startup info
    print(f"\nDevice info: {sd.query_devices(cfg['DEVICE'])['name']}")
    print(f"Config: {config_file}  (edit and save to update live)")
    print(f"clip={cfg['CLIP_SECONDS']}s  "
          f"threshold={cfg['CLASSIFIER_THRESHOLD']}")
    print(f"cooldown={cfg['COOLDOWN']}s")
    print(f"  rms floor={cfg['MIN_RMS']}  save={cfg['SAVE_SECONDS']}s\n")
    
    # Audio callback
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[stream status] {status}", flush=True)
        
        audio = indata[:, 0].copy()
        detector.process_audio_block(audio)
    
    # Run audio stream
    try:
        with sd.InputStream(
            device=cfg["DEVICE"],
            channels=1,
            samplerate=cfg["SAMPLE_RATE"],
            blocksize=int(cfg["SAMPLE_RATE"] * cfg["BLOCK_DURATION"]),
            callback=audio_callback,
        ):
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)


if __name__ == "__main__":
    main()