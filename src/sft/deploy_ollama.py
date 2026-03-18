"""
Phase 3c: Deploy fine-tuned GGUF model to Ollama.

Creates an Ollama Modelfile and registers the fine-tuned model
so it can be used with the same inference pipeline as the base model.
"""

import subprocess
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def create_modelfile(gguf_path, output_path):
    """Create an Ollama Modelfile for the fine-tuned model."""
    content = f"""FROM {gguf_path}

PARAMETER temperature 0.1
PARAMETER num_predict 256

SYSTEM \"\"\"You are an ICU patient monitoring system. Your role is to assess sepsis risk based on sequential vital signs and laboratory values.

At each observation, you must provide:
1. A sepsis risk probability between 0.0 and 1.0
2. A brief clinical reasoning explaining your assessment

Format your response exactly as:
RISK: <probability>
REASONING: <one or two sentences explaining your assessment>\"\"\"
"""
    with open(output_path, "w") as f:
        f.write(content)
    logger.info(f"Created Modelfile at {output_path}")


def deploy(config_path="config/paths.yaml"):
    """Deploy fine-tuned model to Ollama."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_dir = Path(cfg["model_dir"])
    model_name = cfg["ollama"]["model_finetuned"]

    # Find the GGUF file
    gguf_files = list(model_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No GGUF files found in {model_dir}")
    gguf_path = gguf_files[0]
    logger.info(f"Using GGUF: {gguf_path}")

    # Create Modelfile
    modelfile_path = model_dir / "Modelfile"
    create_modelfile(str(gguf_path), str(modelfile_path))

    # Register with Ollama
    logger.info(f"Creating Ollama model '{model_name}'...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"Ollama create failed: {result.stderr}")
        raise RuntimeError(result.stderr)

    logger.info(f"Model '{model_name}' deployed successfully")
    logger.info(f"Test with: ollama run {model_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    deploy()
