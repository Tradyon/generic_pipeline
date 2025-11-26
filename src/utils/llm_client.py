import os
import json
import logging
from typing import Any, Dict, Optional, Union, List

import dotenv
import google.generativeai as genai
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, request_timeout: Optional[float] = 120.0):
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
        self.request_timeout = request_timeout
        model_lower = self.model_name.lower()

        # Load .env if no explicit key provided
        if not api_key:
            dotenv_path = dotenv.find_dotenv(usecwd=True)
            if dotenv_path:
                dotenv.load_dotenv(dotenv_path)
            else:
                dotenv.load_dotenv()

        # Prefer OpenRouter/OpenAI unless explicitly using Gemini without a base_url
        self.provider = "google" if (not self.base_url and model_lower.startswith("gemini")) else "openai"

        if self.provider == "google":
            self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("Google API Key not found. Set GOOGLE_API_KEY.")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
        else:
            self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OpenAI/OpenRouter API Key not found. Set OPENROUTER_API_KEY or OPENAI_API_KEY.")
            if not self.base_url:
                self.base_url = "https://openrouter.ai/api/v1"
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        logger.info(f"Initialized LLMClient with provider={self.provider}, model={self.model_name}")

    def generate(self, prompt: str, schema: Optional[Dict[str, Any]] = None, temperature: float = 0.0) -> Any:
        """
        Generate content from the LLM.
        Returns the response object (which usually has a .text property or similar, 
        but here we will return a unified object or just the text/parsed json).
        
        Actually, the existing code expects a response object with .text.
        Let's return a simple object that mimics the needed interface or just return text.
        
        Existing code:
        response = model.generate_content(...)
        result = json.loads(response.text)
        
        So I should return an object with a .text attribute.
        """
        
        class ResponseWrapper:
            def __init__(self, text, usage=None):
                self.text = text
                self.usage_metadata = usage

        if self.provider == "google":
            generation_config = {
                "temperature": temperature,
            }
            if schema:
                generation_config["response_mime_type"] = "application/json"
                generation_config["response_schema"] = schema
            
            try:
                response = self.client.generate_content(prompt, generation_config=generation_config)
                return response
            except Exception as e:
                logger.error(f"Gemini generation error: {e}")
                raise

        else:
            # OpenAI / OpenRouter
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "timeout": self.request_timeout,
            }
            
            # Prefer structured JSON schema; fall back to json_object if unsupported
            if schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "schema": schema,
                        "strict": True
                    }
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}
            
            # "thinking to be off" - we don't add reasoning parameters.
            
            try:
                response = self.client.chat.completions.create(**kwargs)
            except TypeError:
                if schema:
                    kwargs["response_format"] = {"type": "json_object"}
                    response = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            except Exception as e:
                logger.error(f"OpenAI generation error: {e}")
                raise

            content = response.choices[0].message.content
            
            # Usage stats
            usage = None
            if response.usage:
                usage = type('Usage', (), {})()
                usage.prompt_token_count = response.usage.prompt_tokens
                usage.candidates_token_count = response.usage.completion_tokens
            
            return ResponseWrapper(content, usage)

    def count_tokens(self, text: str) -> int:
        if self.provider == "google":
            return self.client.count_tokens(text).total_tokens
        else:
            # Rough estimation
            return len(text) // 4
