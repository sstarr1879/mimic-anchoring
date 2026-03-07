"""
Ollama API client for running inference on LLaMA 3.1 8B.

Handles both zero-shot (base model) and fine-tuned model inference.
"""

import requests
import json
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, host="http://localhost:11434", model="llama3.1:8b"):
        self.host = host.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str,
                 temperature: float = 0.1, max_tokens: int = 256) -> dict:
        """
        Generate a response from Ollama.

        Returns dict with 'response', 'risk', 'reasoning', and raw 'full_response'.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            return {"response": "", "risk": None, "reasoning": "", "error": str(e)}

        risk, reasoning = parse_risk_response(text)
        return {
            "response": text,
            "risk": risk,
            "reasoning": reasoning,
            "full_response": data,
        }

    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list:
        """List available models on the Ollama server."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.RequestException:
            return []


def parse_risk_response(text: str) -> tuple[Optional[float], str]:
    """
    Parse the model's response to extract risk probability and reasoning.

    Expected format:
        RISK: 0.65
        REASONING: Elevated lactate and rising heart rate suggest...

    Returns (risk_float, reasoning_string). risk is None if parsing fails.
    """
    risk = None
    reasoning = ""

    # Try structured format first
    risk_match = re.search(r"RISK:\s*([\d.]+)", text, re.IGNORECASE)
    if risk_match:
        try:
            val = float(risk_match.group(1))
            if 0.0 <= val <= 1.0:
                risk = val
        except ValueError:
            pass

    reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Fallback: try to find any decimal between 0 and 1
    if risk is None:
        numbers = re.findall(r"(?<!\d)0\.\d+(?!\d)", text)
        for n in numbers:
            val = float(n)
            if 0.0 < val < 1.0:
                risk = val
                break

    if not reasoning:
        reasoning = text.strip()

    return risk, reasoning
