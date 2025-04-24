import torch
from transformers import pipeline
import logging
from typing import List, Tuple
import os
from config import WHISPER_MODEL

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
        self.model = None
        
    def load_model(self) -> None:
        """Loads the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
            self.model = pipeline(
                "automatic-speech-recognition", 
                model=WHISPER_MODEL, 
                device=self.device,
                torch_dtype=torch.float16
            )
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise
            
    def transcribe_segments(self, segment_files: List[Tuple[str, int]]) -> str:
        """Transcribes audio segments and combines them."""
        if not self.model:
            self.load_model()
            
        full_transcription = ""
        
        for segment_path, idx in segment_files:
            logger.info(f"Processing segment {idx+1}/{len(segment_files)}...")
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            try:
                transcription = self.model(
                    inputs=segment_path, 
                    return_timestamps=True,
                    batch_size=16,
                    chunk_length_s=30
                )["text"]
                
                full_transcription += transcription + " "
                logger.info(f"Segment {idx+1} transcription complete. Length: {len(transcription)} characters")
            except Exception as e:
                logger.error(f"Segment {idx+1} transcription error: {e}")
                continue
        
        if not full_transcription.strip():
            logger.error("Transcription empty! Audio file could not be processed or content could not be detected.")
            return "Transcription process failed. Please check the audio file."
        
        return full_transcription
    
    def cleanup(self) -> None:
        """Cleans up model memory."""
        del self.model
        self.model = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
