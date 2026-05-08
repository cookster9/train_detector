"""Configuration manager for train detector."""
import os
import time


class ConfigManager:
    """Loads and hot-reloads configuration from a file."""
    
    def __init__(self, config_file: str):
        self.config_file = config_file
        self._cfg = {}
        self._last_mtime = 0.0
        self._load()
    
    def _load(self):
        """Load configuration from file."""
        cfg = {}
        try:
            with open(self.config_file) as f:
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
            print(f"[config] {self.config_file} not found", flush=True)
        self._cfg = cfg
    
    def get(self, reload_if_changed: bool = True) -> dict:
        """
        Get current configuration, optionally reloading if file changed.
        
        Args:
            reload_if_changed: If True, check for file changes and reload if needed
            
        Returns:
            Configuration dictionary
        """
        if reload_if_changed:
            try:
                mtime = os.path.getmtime(self.config_file)
                if mtime != self._last_mtime:
                    self._last_mtime = mtime
                    self._load()
                    print(f"[config] reloaded at {time.strftime('%H:%M:%S')}", flush=True)
            except OSError:
                pass
        
        return self._cfg
