"""Merge LoRA and export to quantized GGUF using Unsloth directly.

Unsloth's save_pretrained_gguf handles: merge -> HF -> GGUF -> quantize
in one call, building llama.cpp internally if needed.
"""
from pathlib import Path
from unsloth import FastLanguageModel

model_dir = Path("data/models")
adapter_path = model_dir / "llama31-sepsis-lora"
gguf_out_dir = model_dir / "llama31-sepsis-gguf"

# Quantization methods to produce. Q4_K_M is the Ollama deployment target;
# add others here if you want multiple options (e.g. "q5_k_m", "q8_0", "f16").
QUANT_METHODS = ["q4_k_m"]

print(f"Loading LoRA adapter from {adapter_path}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(adapter_path),
    max_seq_length=4096,
    load_in_4bit=False,
    dtype=None,
)

for qm in QUANT_METHODS:
    print(f"\n=== Exporting GGUF ({qm}) ===")
    model.save_pretrained_gguf(
        str(gguf_out_dir),
        tokenizer,
        quantization_method=qm,
    )

print(f"\nDone. GGUF files in {gguf_out_dir}")
for f in sorted(gguf_out_dir.glob("*.gguf")):
    print(f"  {f.name}  ({f.stat().st_size / 1e9:.2f} GB)")
