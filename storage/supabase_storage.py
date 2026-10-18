"""Supabase storage backend for train detections."""
import numpy as np
from storage.base import StorageBackend


class SupabaseStorage(StorageBackend):
    """Store detections in Supabase database."""
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        fra_crossing_id: str | None,
    ):
        """
        Initialize Supabase storage.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (service_role or anon)
            fra_crossing_id: FRA crossing identifier used to find the crossing row
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.fra_crossing_id = (
            fra_crossing_id.strip() if fra_crossing_id else None
        )
        self.crossing_id = None
        self.client = None
    
    def initialize(self) -> bool:
        """Initialize Supabase client."""
        if not self.fra_crossing_id:
            print(
                "[supabase] FRA_CROSSING_ID is not set; cannot resolve crossing",
                flush=True,
            )
            return False

        try:
            from supabase import create_client
            self.client = create_client(self.supabase_url, self.supabase_key)

            response = (
                self.client.table("crossings")
                .select("id")
                .eq("fra_crossing_id", self.fra_crossing_id)
                .execute()
            )
            crossings = response.data
            if not crossings or len(crossings) != 1 or not crossings[0].get("id"):
                print(
                    "[supabase] Expected exactly one crossing with "
                    f"fra_crossing_id={self.fra_crossing_id!r}; "
                    f"found {len(crossings) if crossings else 0}",
                    flush=True,
                )
                return False

            self.crossing_id = crossings[0]["id"]
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
                "crossing_id": self.crossing_id,
            }
            
            response = self.client.table("detections").insert(data).execute()
            
            print(f"[supabase] Saved detection: {timestamp} {label}", flush=True)
            return True
            
        except Exception as e:
            print(f"[supabase] Failed to save: {e}", flush=True)
            return False