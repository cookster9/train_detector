"""Abstract base class for storage backends."""
from abc import ABC, abstractmethod
import numpy as np


class StorageBackend(ABC):
    """Abstract interface for storing train detection data."""
    
    @abstractmethod
    def save_detection(
        self,
        audio: np.ndarray,
        scores: list,
        sr: int,
        timestamp: str,
        label: str,
    ) -> bool:
        """
        Save a detected train horn event.
        
        Args:
            audio: Audio array (float32)
            scores: List of classification scores
            sr: Sample rate
            timestamp: Detection timestamp (YYYYMMDD_HHMMSS format)
            label: Detection label ('train_close', 'train_faraway', 'train_interference')
            
        Returns:
            True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize storage backend.
        
        Returns:
            True if initialization was successful
        """
        pass
