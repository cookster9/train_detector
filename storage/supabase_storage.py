"""Supabase storage backend for train detections."""
import numpy as np
from storage.base import StorageBackend


class SupabaseStorage(StorageBackend):
    """Store detections in Supabase database."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Initialize Supabase storage.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (service_role or anon)
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.client = None
    
    def initialize(self) -> bool:
        """Initialize Supabase client."""
        try:
            from supabase import create_client
            self.client = create_client(self.supabase_url, self.supabase_key)
            print("[supabase] Connected to Supabase", flush=True)
            return True
        except ImportError:
            print(
                "[supabase] supabase-py not installed. "
                "Install with: pip install supabase",
                flush=True
            )
            return False
        except Exception as e:
            print(f"[supabase] Failed to initialize: {e}", flush=True)
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
        Save detection to Supabase.
        
        Args:
            audio: Audio array (float32) - unused for DB storage
            scores: List of classification scores
            sr: Sample rate
            timestamp: Detection timestamp
            label: Detection label
            
        Returns:
            True if save was successful
        """
        try:
            if not self.client:
                print("[supabase] Client not initialized", flush=True)
                return False
            
            data = {
                "timestamp": timestamp,
                "label": label,
                "score": float(scores[0]) if scores else 0.0,
                "sample_rate": sr,
            }
            
            response = self.client.table("detections").insert(data).execute()
            
            print(f"[supabase] Saved detection: {timestamp} {label}", flush=True)
            return True
            
        except Exception as e:
            print(f"[supabase] Failed to save: {e}", flush=True)
            return False