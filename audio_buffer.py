"""Rolling audio buffer for maintaining recent audio history."""
from collections import deque
import numpy as np


class AudioBuffer:
    """Maintains a rolling buffer of audio blocks."""
    
    def __init__(self, buffer_seconds: float, block_duration: float):
        """
        Initialize the audio buffer.
        
        Args:
            buffer_seconds: Total buffer duration in seconds
            block_duration: Duration of each audio block in seconds
        """
        self.buffer_seconds = buffer_seconds
        self.block_duration = block_duration
        self._resize_buffer()
    
    def _resize_buffer(self):
        """Resize the internal deque based on current settings."""
        max_blocks = int(self.buffer_seconds / self.block_duration)
        self._buffer = deque(maxlen=max_blocks)
    
    def append(self, audio_block: np.ndarray):
        """
        Add an audio block to the buffer.
        
        Args:
            audio_block: Numpy array of audio samples
        """
        self._buffer.append(audio_block)
    
    def get_latest(self, clip_seconds: float) -> np.ndarray:
        """
        Get the most recent N seconds of audio as a concatenated array.
        
        Args:
            clip_seconds: Duration of audio to retrieve
            
        Returns:
            Concatenated audio array
        """
        clip_blocks = int(clip_seconds / self.block_duration)
        buf_list = list(self._buffer)
        clip_blocks = min(clip_blocks, len(buf_list))
        
        if clip_blocks == 0:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(buf_list[-clip_blocks:]).astype(np.float32)
    
    def get_all(self) -> np.ndarray:
        """
        Get all buffered audio as a concatenated array.
        
        Returns:
            Concatenated audio array
        """
        if len(self._buffer) == 0:
            return np.array([], dtype=np.float32)
        
        return np.concatenate(list(self._buffer)).astype(np.float32)
    
    def update_config(self, buffer_seconds: float, block_duration: float):
        """
        Update buffer configuration and resize if needed.
        
        Args:
            buffer_seconds: New buffer duration
            block_duration: New block duration
        """
        old_max = self._buffer.maxlen
        self.buffer_seconds = buffer_seconds
        self.block_duration = block_duration
        self._resize_buffer()
        
        # Preserve existing audio if downsizing
        if old_max != self._buffer.maxlen:
            old_data = list(self._buffer)
            self._buffer = deque(old_data[-self._buffer.maxlen:], maxlen=self._buffer.maxlen)
    
    def __len__(self) -> int:
        """Get number of blocks in buffer."""
        return len(self._buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0
