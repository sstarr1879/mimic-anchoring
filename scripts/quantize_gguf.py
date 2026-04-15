"""Quantize F16 GGUF to Q4_K_M using llama-cpp-python."""
from llama_cpp import Llama
import subprocess
import sys
from pathlib import Path

model_dir = Path("data/models")
f16_path = model_dir / "llama31-sepsis-f16.gguf"
q4_path = model_dir / "llama31-sepsis-q4km.gguf"

if not f16_path.exists():
    print(f"ERROR: {f16_path} not found")
    sys.exit(1)

print(f"Quantizing {f16_path} -> {q4_path}")
print("This may take 10-20 minutes...")

# Use llama.cpp's quantize via the installed package
import llama_cpp
lib_dir = Path(llama_cpp.__file__).parent
quantize_bin = lib_dir / "llama-quantize"

if quantize_bin.exists():
    subprocess.run([str(quantize_bin), str(f16_path), str(q4_path), "Q4_K_M"], check=True)
else:
    # Fallback: try system path
    subprocess.run(["llama-quantize", str(f16_path), str(q4_path), "Q4_K_M"], check=True)

print(f"Done! Quantized model saved to {q4_path}")
print(f"Size: {q4_path.stat().st_size / 1e9:.1f} GB")
