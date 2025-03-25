import os
import httpx
from dotenv import load_dotenv

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

async def query_mistral(prompt: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=60
            )
        response.raise_for_status()
        return response.json().get("response", "[No response from model]")
