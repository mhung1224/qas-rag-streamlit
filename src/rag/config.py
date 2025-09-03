import os
from dataclasses import dataclass
from typing import Optional
import streamlit as st
    
@dataclass
class ChunkConfig:
    chunk_size: int = 128
    chunk_overlap: int = 32

    @classmethod
    def from_env(cls):
        return cls(
            chunk_size=int(st.secrets['CHUNKING']['SIZE'] or os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(st.secrets['CHUNKING']['OVERLAP'] or os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
        )

# Pinecone vector DB
@dataclass
class PineconeConfig: 
    api_key: Optional[str] = None
    index_name: Optional[str] = "rag-docs"
    dimension: int = 768
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    namespace: Optional[str] = "default"
    batch_size: int = 256
    normalize_embeddings: bool = True

    @classmethod
    def from_env(cls):
        return cls(
            api_key=st.secrets['API_KEY']['PINECONE'] or os.getenv("PINECONE_API_KEY", cls.api_key)
        )

@dataclass
class EmbedderConfig:
    model_name: str = "gemini-embedding-001"
    dimension: int = 768
    api_key: Optional[str] = None
    timeout: int = 60
    normalize: bool = True
    max_retries: int = 3
    retry_backoff: float = 1.2

    @classmethod
    def from_env(cls):
        return cls(
            api_key=st.secrets['API_KEY']['GEMINI'] or os.getenv("GEMINI_API_KEY", cls.api_key),
            model_name=os.getenv("EMBED_MODEL", cls.model_name)
        )
    
@dataclass
class LLMConfig:
    model_name: str = "gemini-2.5-flash"
    api_key: Optional[str] = None
    temperature: float = 0.2
    timeout: int = 120

    @classmethod
    def from_env(cls):
        return cls(
            model_name=os.getenv("LLM_MODEL", cls.model_name),
            api_key=st.secrets['API_KEY']['GEMINI'] or os.getenv("GEMINI_API_KEY", cls.api_key)
        )

    @classmethod
    def change_model(cls, new_model: str):
        cls.model_name = new_model
    
@dataclass
class AppConfig:
    chunk_cfg = ChunkConfig.from_env()
    pinecone_cfg = PineconeConfig.from_env()
    embedder_cfg = EmbedderConfig.from_env()
    llm_cfg = LLMConfig.from_env()
    model_list = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2", "gemini-2.5-turbo", "gemini-2.5-flash"]