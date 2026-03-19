"""Export saved LoRA adapter to GGUF format for Ollama deployment."""

import logging
from pathlib import Path

from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_dir = Path("/SEAS/home/g32999482/emse-6575/mimic-anchoring/data/models")
adapter_path = model_dir / "llama31-sepsis-lora"
gguf_path = model_dir / "llama31-sepsis"

logger.info(f"Loading LoRA adapter from {adapter_path}")
model, tokenizer = FastLanguageModel.from_pretrained(str(adapter_path))

logger.info(f"Exporting GGUF to {gguf_path}.gguf")
model.save_pretrained_gguf(
    str(gguf_path),
    tokenizer,
    quantization_method="q4_k_m",
)
logger.info("GGUF export complete")
