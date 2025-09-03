from typing import List, Optional, Union
import time
import numpy as np
from google import genai
from google.genai import types
from .config import EmbedderConfig


class Embedder:
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.cfg = EmbedderConfig.from_env()
        self.cfg.api_key = self.cfg.api_key or api_key
        if not self.cfg.api_key:
            raise ValueError("GEMINI_API_KEY is required. Set GEMINI_API_KEY env or pass GEMINI_API_KEY.")
        self.client = genai.Client(api_key = self.cfg.api_key)

    def _feature_extraction(self, text: Union[List[str], str], is_query: bool = False):
        attempt = 0
        while True:
            try:
                response = self.client.models.embed_content(
                    contents = text,
                    model=self.cfg.model_name,
                    config=types.EmbedContentConfig(
                        task_type = "RETRIEVAL_DOCUMENT" if not is_query else "RETRIEVAL_QUERY",
                        output_dimensionality=self.cfg.dimension
                    )
                )
                if isinstance(text, str):
                    return response.embeddings[0].values
                return [emb.values for emb in response.embeddings]
            except Exception:
                attempt += 1
                if attempt > self.cfg.max_retries:
                    raise
                time.sleep(self.cfg.retry_backoff ** attempt)

    @staticmethod
    def _l2_normalize(v) -> List[float]:
        x = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(x)
        if n > 0:
            x = x / n
        return x.tolist()

    def embed(self, texts: List[str], is_query: bool = False):
        if not texts:
            return []
        out = []
        for t in texts:
            vec = self._feature_extraction(t, is_query)
            if self.cfg.normalize:
                vec = self._l2_normalize(vec)
            out.append(vec)
        return out