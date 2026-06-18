"""Core train horn detection logic."""
import threading
import time
import numpy as np
from config import ConfigManager
from classifier import AudioClassifier
from audio_buffer import AudioBuffer
from storage.base import StorageBackend


class TrainDetector:
    """Main detector that runs audio stream classification and saves detections."""
    
    def __init__(
        self,
        config: ConfigManager,
        classifier: AudioClassifier,
        storage: StorageBackend,
    ):
        """
        Initialize the detector.
        
        Args:
            config: Configuration manager
            classifier: Audio classifier instance
            storage: Storage backend for saving detections
        """
        self.config = config
        self.classifier = classifier
        self.storage = storage
        
        cfg = self.config.get()
        self.audio_buffer = AudioBuffer(
            cfg.get("BUFFER_SECONDS", 60),
            cfg.get("BLOCK_DURATION", 1.0),
        )
        
        self.last_classify_time = 0.0
        self.last_save_time = 0.0
        self.classifier_busy = False
        
        # Train stopping detection state
        self.last_train_close_time = 0.0
        self.last_train_stopping_time = 0.0
        self.train_activity_window = cfg.get("TRAIN_ACTIVITY_WINDOW", 30.0)  # seconds
        self.last_train_stopping_save = 0.0
        self.baseline_train_score = None  # Track average train sound during activity
    
    def _classify_window(self, audio_snap: np.ndarray) -> tuple:
        """
        Classify a window of audio and determine detection status.
        
        Returns:
            Tuple of (score, label) or (None, None) if no detection
        """
        cfg = self.config.get()
        sr = cfg["SAMPLE_RATE"]
        threshold = cfg["CLASSIFIER_THRESHOLD"]
        threshold_faraway = cfg["CLASSIFIER_THRESHOLD_FARAWAY"]
        
        all_scores, labels = self.classifier.classify(audio_snap, sr)
        
        if all_scores is None:
            return None, None
        
        score = all_scores[AudioClassifier.TRAIN_HORN_CLASS]
        
        if cfg.get("DEBUG"):
            print(f"[classifier] score={score:.4f}  threshold={threshold}  ", flush=True)
            
            # Print top 10 classifications
            if labels is not None:
                top_indices = np.argsort(all_scores)[::-1][:10]
                print("[classifier] ------ Top classifications ------", flush=True)
                for idx in top_indices:
                    if idx < len(labels):
                        class_score = float(all_scores[idx])
                        if class_score > 0.001:
                            print(f"  {labels[idx]}: {class_score:.3f}", flush=True)        
        
        if score >= threshold:
            print(f"  ✓ hit  score={score:.4f}  ", flush=True)
            return score, "train_close"
        elif score >= threshold_faraway:
            print(f"  ⚠️  faraway  score={score:.4f}  ", flush=True)
            return score, "train_faraway"
        
        return None, None
    
    def _classify_stopping_window(self, audio_snap: np.ndarray) -> tuple:
        """
        Classify a window of audio for train stopping indicators.
        
        Used after train_close detected to look for squealing wheels, air brake, etc.
        
        Returns:
            Tuple of (stopping_score, label) or (None, None) if no strong stopping signal
        """
        cfg = self.config.get()
        sr = cfg["SAMPLE_RATE"]
        
        all_scores, labels = self.classifier.classify(audio_snap, sr)
        
        if all_scores is None:
            return None, None
        
        # Extract stopping indicator scores
        wheels_squeal = all_scores[AudioClassifier.TRAIN_WHEELS_SQUEALING_CLASS]
        air_brake = all_scores[AudioClassifier.AIR_BRAKE_CLASS]
        train_sound = all_scores[AudioClassifier.TRAIN_CLASS]
        
        # Composite stopping confidence: high weight on squealing, moderate on brake
        stopping_score = (wheels_squeal * 0.6) + (air_brake * 0.3) + (train_sound * 0.1)
        
        stopping_threshold = cfg.get("STOPPING_THRESHOLD", 0.15)
        
        if cfg.get("DEBUG"):
            print(f"[stopping] wheels_squeal={wheels_squeal:.4f}  "
                  f"air_brake={air_brake:.4f}  train_sound={train_sound:.4f}  "
                  f"composite={stopping_score:.4f}  threshold={stopping_threshold}", flush=True)
        
        if stopping_score >= stopping_threshold:
            print(f"  🛑 stopping detected  score={stopping_score:.4f}", flush=True)
            return stopping_score, "train_stopping"
        
        return None, None
    
    def _save_detection(self, score: float, label: str):
        """Save a detection to storage."""
        cfg = self.config.get()
        sr = cfg["SAMPLE_RATE"]
        save_secs = cfg["SAVE_SECONDS"]
        
        # Grab SAVE_SECONDS of audio from buffer
        save_blocks = int(save_secs / cfg["BLOCK_DURATION"])
        save_blocks = min(save_blocks, len(self.audio_buffer))
        
        if save_blocks > 0:
            audio_save = self.audio_buffer.get_latest(save_secs)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            scores = [score]
            
            # Save asynchronously
            threading.Thread(
                target=self.storage.save_detection,
                args=(audio_save, scores, sr, timestamp, label),
                daemon=True
            ).start()
    
    def _classify_worker(self, audio_snap: np.ndarray, snap_time: float):
        """Worker thread that classifies audio and saves detections."""
        try:
            cfg = self.config.get()
            cooldown = cfg["COOLDOWN"]
            cooldown_stopping = cfg.get("STOPPING_COOLDOWN", cooldown)
            now = time.time()
            
            # Check if we're in train activity window (post-horn monitoring)
            time_since_train_close = now - self.last_train_close_time
            in_activity_window = time_since_train_close < self.train_activity_window
            
            if in_activity_window and cfg.get("DEBUG"):
                print(f"[classifier] in activity window  "
                      f"t_since_close={time_since_train_close:.1f}s", flush=True)
            
            # Check for horn first
            score, label = self._classify_window(audio_snap)
            
            if label is not None:
                # Update train activity marker
                if label == "train_close":
                    self.last_train_close_time = now
                
                # Only save if cooldown has passed
                if (now - self.last_save_time) < cooldown:
                    if cfg.get("DEBUG"):
                        print(f"[classifier] cooldown active — skipping save", flush=True)
                    return
                
                self.last_save_time = now
                self._save_detection(score, label)
            
            # Check for stopping indicators
            stopping_score, stopping_label = self._classify_stopping_window(audio_snap)
            
            if stopping_label is not None:
                # Only save if cooldown has passed
                self.last_train_stopping_time = now
                if abs(self.last_train_stopping_time - self.last_train_close_time) < self.train_activity_window:
                    if (now - self.last_train_stopping_save) < cooldown_stopping:
                        if cfg.get("DEBUG"):
                            print(f"[classifier] stopping cooldown active — skipping save", flush=True)
                        return
                    self.last_train_stopping_save = now
                    self._save_detection(stopping_score, stopping_label)
            
        finally:
            self.classifier_busy = False
    
    def process_audio_block(self, audio: np.ndarray):
        """
        Process a new audio block from the stream.
        
        Args:
            audio: Audio block (typically 1 second)
        """
        cfg = self.config.get()
        
        # Add to buffer
        self.audio_buffer.append(audio)
        
        # Update buffer size if config changed
        new_buffer_secs = cfg.get("BUFFER_SECONDS", 60)
        new_block_dur = cfg.get("BLOCK_DURATION", 1.0)
        if (self.audio_buffer.buffer_seconds != new_buffer_secs or
            self.audio_buffer.block_duration != new_block_dur):
            self.audio_buffer.update_config(new_buffer_secs, new_block_dur)
        
        # Classify at configured interval
        now = time.time()
        if (now - self.last_classify_time) < cfg["CLASSIFY_EVERY"]:
            return
        
        if self.classifier_busy:
            if cfg.get("DEBUG"):
                print("[classifier] still busy — skipping", flush=True)
            return
        
        # Pre-filter: skip if window is too quiet
        audio_snap = self.audio_buffer.get_latest(cfg["CLIP_SECONDS"])
        if len(audio_snap) == 0:
            return
        
        block_size = int(cfg["SAMPLE_RATE"] * cfg["BLOCK_DURATION"])
        block_rms = [
            np.sqrt(np.mean(audio_snap[i:i + block_size] ** 2))
            for i in range(0, len(audio_snap), block_size)
        ]
        window_rms = max(block_rms) if block_rms else 0.0
        
        self.last_classify_time = now
        
        if window_rms < cfg["MIN_RMS"]:
            if cfg.get("DEBUG"):
                print(f"window_rms={window_rms:.4f}  [quiet]", flush=True)
            return
        
        if cfg.get("DEBUG"):
            print(f"window_rms={window_rms:.4f}  classifying...", flush=True)
        
        # Start classification worker
        self.classifier_busy = True
        threading.Thread(
            target=self._classify_worker,
            args=(audio_snap, now),
            daemon=True
        ).start()
