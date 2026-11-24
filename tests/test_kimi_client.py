"""Unit tests for KimiVisionClient."""

import os
import tempfile
import pytest
from road_damage.kimi_client import KimiVisionClient


def test_kimi_client_initialization() -> None:
    client = KimiVisionClient(api_key="test_key")
    assert client.api_key == "test_key"
    assert "integrate.api.nvidia.com" in client.base_url
    assert client.model == "moonshotai/kimi-k3.1-pro"


def test_build_payload_text_only() -> None:
    client = KimiVisionClient(api_key="test_key")
    payload = client.build_payload("Analyze this road", max_tokens=1024)

    assert payload["model"] == "moonshotai/kimi-k3.1-pro"
    assert payload["max_tokens"] == 1024
    assert len(payload["messages"][0]["content"]) == 1
    assert payload["messages"][0]["content"][0]["type"] == "text"


def test_build_payload_with_image_url() -> None:
    client = KimiVisionClient(api_key="test_key")
    image_url = "https://example.com/pothole.jpg"
    payload = client.build_payload("Find cracks", image_source=image_url)

    content = payload["messages"][0]["content"]
    assert len(content) == 2
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == image_url


def test_build_payload_with_local_file() -> None:
    client = KimiVisionClient(api_key="test_key")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00")
        tmp_path = tmp.name

    try:
        payload = client.build_payload("Analyze", image_source=tmp_path)
        content = payload["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
