"""Kimi 3.1 Pro (Moonshot AI via NVIDIA NIM) Client for Road Damage Vision Analysis."""

import base64
import json
import mimetypes
import os
from typing import Generator, Optional, Union
import requests


class KimiVisionClient:
    """Client for Moonshot AI Kimi 3.1 Pro multimodal reasoning model hosted on NVIDIA NIM."""

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    DEFAULT_MODEL = "moonshotai/kimi-k3.1-pro"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or self._load_key_from_env()
        self.base_url = (base_url or os.getenv("NVIDIA_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("KIMI_MODEL") or self.DEFAULT_MODEL
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Please add it to your .env file or pass it to KimiVisionClient."
            )

    @staticmethod
    def _load_key_from_env() -> Optional[str]:
        # Check environment variable first
        key = os.getenv("NVIDIA_API_KEY")
        if key:
            return key.strip()

        # Check local .env file
        if os.path.exists(".env"):
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("NVIDIA_API_KEY="):
                        return line.split("=", 1)[1].strip()
        return None

    def _prepare_image_url(self, image_source: str) -> str:
        """Convert a local file path to a base64 data URI or return external HTTP URL."""
        if image_source.startswith("http://") or image_source.startswith("https://") or image_source.startswith("data:"):
            return image_source

        if not os.path.isfile(image_source):
            raise FileNotFoundError(f"Image file not found: {image_source}")

        mime_type, _ = mimetypes.guess_type(image_source)
        if not mime_type:
            mime_type = "image/jpeg"

        with open(image_source, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def build_payload(
        self,
        prompt: str,
        image_source: Optional[str] = None,
        max_tokens: int = 4096,
        stream: bool = True,
        reasoning_effort: str = "low",
    ) -> dict:
        """Construct the OpenAI-compatible chat completion payload."""
        content = [{"type": "text", "text": prompt}]

        if image_source:
            image_url = self._prepare_image_url(image_source)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "stream": stream,
            "seed": 0,
            "reasoning_effort": reasoning_effort,
        }

    def analyze_stream(
        self,
        prompt: str,
        image_source: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """
        Stream the reasoning & vision analysis response from Kimi 3.1 Pro chunk-by-chunk.
        """
        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload = self.build_payload(prompt, image_source=image_source, max_tokens=max_tokens, stream=True)

        try:
            response = requests.post(endpoint, headers=headers, json=payload, stream=True, timeout=self.timeout)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded.startswith("data: ") and "[DONE]" not in decoded:
                    try:
                        chunk = json.loads(decoded[6:])
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        except requests.exceptions.RequestException as e:
            yield f"\n[API Connection Error: {e}]"

    def analyze(
        self,
        prompt: str,
        image_source: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> str:
        """Non-streaming analysis helper."""
        chunks = list(self.analyze_stream(prompt, image_source, max_tokens))
        return "".join(chunks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query Moonshot Kimi 3.1 Pro via NVIDIA NIM")
    parser.add_argument("prompt", help="Instruction or query for Kimi 3.1 Pro")
    parser.add_argument("--image", "-i", default=None, help="Local image file path or remote image URL")
    parser.add_argument("--max-tokens", "-m", type=int, default=2048, help="Max tokens to generate")
    args = parser.parse_args()

    client = KimiVisionClient()
    for chunk in client.analyze_stream(args.prompt, image_source=args.image, max_tokens=args.max_tokens):
        print(chunk, end="", flush=True)
    print()

