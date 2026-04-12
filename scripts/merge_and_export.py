"""Merge LoRA adapter into base model, then export to GGUF via llama.cpp."""
import subprocess
import sys
from pathlib import Path

model_dir = Path("data/models")
adapter_path = model_dir / "llama31-sepsis-lora"
merged_path = model_dir / "llama31-sepsis-merged"
gguf_f16 = model_dir / "llama31-sepsis-f16.gguf"
gguf_q4 = model_dir / "llama31-sepsis.gguf"
llama_cpp = Path.home() / "llama.cpp"

# Step 1: Merge LoRA into base model
print("=== Merging LoRA adapter into base model ===")
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(str(adapter_path))
model.save_pretrained_merged(str(merged_path), tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {merged_path}")

# Step 2: Convert to GGUF f16
print("\n=== Converting to GGUF (f16) ===")
convert_script = llama_cpp / "convert_hf_to_gguf.py"
subprocess.run([
    sys.executable, str(convert_script),
    str(merged_path),
    "--outfile", str(gguf_f16),
    "--outtype", "f16",
], check=True)
print(f"F16 GGUF saved to {gguf_f16}")

# Step 3: Quantize to Q4_K_M
print("\n=== Quantizing to Q4_K_M ===")
quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
subprocess.run([
    str(quantize_bin),
    str(gguf_f16),
    str(gguf_q4),
    "Q4_K_M",
], check=True)
print(f"Quantized GGUF saved to {gguf_q4}")
print("\nDone! You can now submit step 5.")
