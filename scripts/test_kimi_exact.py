import json
import os
import requests

# Load API key from .env
api_key = None
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.startswith("NVIDIA_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env")

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = True

headers = {
    "Authorization": f"Bearer {api_key}",
    "Accept": "text/event-stream" if stream else "application/json",
}

payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image? Describe in 1 sentence."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://assets.ngc.nvidia.com/products/api-catalog/phi-3-5-vision/example1b.jpg"
                    }
                }
            ]
        }
    ],
    "model": "moonshotai/kimi-k3.1-pro",
    "max_tokens": 512,
    "seed": 0,
    "stream": stream,
    "reasoning_effort": "low"
}

print("Connecting to NVIDIA NIM Kimi 3.1 Pro...")
response = requests.post(invoke_url, headers=headers, json=payload, stream=stream, timeout=90)
print("HTTP Status Code:", response.status_code)

if stream:
    collected = []
    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and "[DONE]" not in decoded:
                try:
                    chunk = json.loads(decoded[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                        collected.append(content)
                except Exception:
                    pass
    print("\n--- Completed ---")
else:
    print(response.json())
