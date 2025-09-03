from google import genai
from .config import LLMConfig
from typing import Optional
# import streamlit as st
# import time


class LLM:
    def __init__(self, api_key: Optional[str] = None):
        self.cfg = LLMConfig.from_env()
        self.cfg.api_key = self.cfg.api_key or api_key
        if not self.cfg.api_key:
            raise "[LLM_API_KEY is missing]"
        self.client = genai.Client(api_key = self.cfg.api_key)

    async def generate(self, prompt: str):
        response = self.client.models.generate_content_stream(
            model = self.cfg.model_name,
            contents = prompt
        )
        if not response:
            raise RuntimeError("[LLM stream failed]")

        return response