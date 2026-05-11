"""Composite storage backend that writes to multiple backends."""
import numpy as np
from typing import List
from storage.base import StorageBackend


class CompositeStorage(StorageBackend):
    """Save detections to multiple storage backends simultaneously."""
    
    def __init__(self, backends: List[StorageBackend]):
        """
        Initialize with multiple storage backends.
        
        Args:
            backends: List of StorageBackend instances to write to
        """
        self.backends = backends
    
    def initialize(self) -> bool:
        """Initialize all backends."""
        all_ok = True
        for i, backend in enumerate(self.backends):
            try:
                if not backend.initialize():
                    print(
                        f"[composite] Backend {i} ({backend.__class__.__name__}) "
                        f"failed to initialize",
                        flush=True
                    )
                    all_ok = False
                else:
                    print(
                        f"[composite] Backend {i} ({backend.__class__.__name__}) "
                        f"initialized",
                        flush=True
                    )
            except Exception as e:
                print(
                    f"[composite] Backend {i} ({backend.__class__.__name__}) "
                    f"error: {e}",
                    flush=True
                )
                all_ok = False
        
        return all_ok
    
    def save_detection(
        self,
        audio: np.ndarray,
        scores: list,
        sr: int,
        timestamp: str,
        label: str,
    ) -> bool:
        """
        Save detection to all backends.
        
        Args:
            audio: Audio array (float32)
            scores: List of classification scores
            sr: Sample rate
            timestamp: Detection timestamp
            label: Detection label
            
        Returns:
            True if all backends succeeded
        """
        all_ok = True
        for i, backend in enumerate(self.backends):
            try:
                success = backend.save_detection(audio, scores, sr, timestamp, label)
                if not success:
                    print(
                        f"[composite] Backend {i} ({backend.__class__.__name__}) "
                        f"returned False",
                        flush=True
                    )
                    all_ok = False
            except Exception as e:
                print(
                    f"[composite] Backend {i} ({backend.__class__.__name__}) "
                    f"error: {e}",
                    flush=True
                )
                all_ok = False
        
        return all_ok
