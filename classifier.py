"""PANNs audio classifier wrapper."""
import sys
import os
import numpy as np


class AudioClassifier:
    """Wrapper around PANNs AudioTagging for train horn detection."""
    
    # Class indices from PANNs
    TRAIN_HORN_CLASS = 331
    VEHICLE_CLASS = 300
    AIRCRAFT_CLASS = 335
    AIR_CONDITIONER_CLASS = 413
    
    def __init__(self, panns_path: str = None, device: str = 'cpu'):
        """
        Initialize the audio classifier.
        
        Args:
            panns_path: Path to panns_inference directory. If None, uses default relative path.
            device: Device to load model on ('cpu' or 'cuda')
        """
        self.device = device
        self._classifier = None
        self._librosa = None
        self._load(panns_path)
    
    def _load(self, panns_path: str = None):
        """Load PANNs model."""
        try:
            if panns_path is None:
                panns_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'panns_inference'
                )
            
            sys.path.insert(0, panns_path)
            
            print("Loading PANNs classifier...", flush=True)
            from panns_inference.inference import AudioTagging
            import librosa
            
            self._classifier = AudioTagging(checkpoint_path=None, device=self.device)
            self._librosa = librosa
            print("Classifier ready.", flush=True)
            
        except Exception as e:
            print(f"[classifier] Failed to load: {e}", flush=True)
            self._classifier = None
            self._librosa = None
    
    def is_ready(self) -> bool:
        """Check if classifier is loaded and ready."""
        return self._classifier is not None
    
    def classify(self, audio_float32: np.ndarray, sr: int) -> tuple:
        """
        Run PANNs on a float32 [-1,1] audio array.
        
        Args:
            audio_float32: Audio array in float32 format
            sr: Sample rate
            
        Returns:
            Tuple of (all_scores, labels) or (None, None) on failure
        """
        if not self.is_ready():
            return None, None
        
        try:
            # Normalize audio
            audio_float32 = np.clip(audio_float32 * 1.0, -1.0, 1.0)
            
            print(f"[classify] peak={np.max(np.abs(audio_float32)):.4f} "
                  f"rms={np.sqrt(np.mean(audio_float32**2)):.4f}", flush=True)
            
            # Mono check
            if audio_float32.ndim > 1:
                audio_float32 = audio_float32[:, 0]
            
            # Resample to 32kHz if needed
            if sr != 32000:
                audio_32k = self._librosa.resample(
                    audio_float32,
                    orig_sr=sr,
                    target_sr=32000
                )
            else:
                audio_32k = audio_float32
            
            out, _ = self._classifier.inference(audio_32k[np.newaxis, :])
            
            all_scores = out[0]
            labels = self._classifier.labels
            
            return all_scores, labels
            
        except Exception as e:
            print(f"[classifier] error: {e}", flush=True)
            return None, None
    
    def get_labels(self):
        """Get list of all class labels."""
        if self.is_ready():
            return self._classifier.labels
        return None
