# rag/generator.py
import os
from dotenv import load_dotenv
import subprocess

# Load .env variables
load_dotenv()

# Ollama model name, default to llama2 (change as needed)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")


def generate_with_llama(prompt: str) -> str:
    """
    Generate a response using Ollama CLI with the specified local model.

    Args:
        prompt (str): The text prompt to send to the model.

    Returns:
        str: The model-generated response.
    """
    try:
        # Run Ollama CLI: ensure 'ollama' is installed and the model is pulled locally
        result = subprocess.run([
            "ollama", "run", OLLAMA_MODEL,
            "--prompt", prompt
        ], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., model not found, CLI not installed)
        print(f"Error generating with Ollama CLI: {e.stderr}")
        return ""