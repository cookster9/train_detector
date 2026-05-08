"""File-based storage backend for train detections."""
import os
import numpy as np
from scipy.io.wavfile import write
from storage.base import StorageBackend


class FileStorage(StorageBackend):
    """Store detections as local files and a log file."""
    
    def __init__(self, clips_dir: str = "clips", log_file: str = "detections.txt"):
        """
        Initialize file storage.
        
        Args:
            clips_dir: Directory to save audio clips
            log_file: File to log detection metadata
        """
        self.clips_dir = clips_dir
        self.log_file = log_file
    
    def initialize(self) -> bool:
        """Create clips directory if it doesn't exist."""
        try:
            os.makedirs(self.clips_dir, exist_ok=True)
            return True
        except Exception as e:
            print(f"[storage] Failed to initialize: {e}", flush=True)
            return False
    
    def save_detection(
        self,
        audio: np.ndarray,
        scores: list,
        sr: int,
        timestamp: str,
        label: str,
    ) -> bool:
        """
        Save detection to local files.
        
        Args:
            audio: Audio array (float32)
            scores: List of classification scores
            sr: Sample rate
            timestamp: Detection timestamp
            label: Detection label
            
        Returns:
            True if save was successful
        """
        try:
            filename = os.path.join(self.clips_dir, f"{label}_{timestamp}.wav")
            write(filename, sr, audio)
            
            duration = len(audio) / sr
            score_str = "  ".join(f"{s:.3f}" for s in scores)
            
            with open(self.log_file, "a") as f:
                f.write(
                    f"{timestamp}  hits={len(scores)}  scores=[{score_str}]  "
                    f"duration={duration:.1f}s  label={label}\n"
                )
            
            print(
                f"\n🚂 SAVED: {filename}  "
                f"({len(scores)} hits  scores=[{score_str}]  {duration:.1f}s)",
                flush=True
            )
            
            return True
            
        except Exception as e:
            print(f"[storage] Failed to save: {e}", flush=True)
            return False
