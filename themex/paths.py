import os
from pathlib import Path

# Project root path
PROJECT_ROOT = Path(__file__).parent.parent

# ---------------- Data ----------------
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)
# RAW_DATA_DIR = DATA_DIR / "raw"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"

# # ---------------- LLaMA Models ----------------
# LLAMA_MODEL_DIR = PROJECT_ROOT / "llama_models"
# LLAMA_MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# LLAMA_MODEL_PATH = LLAMA_MODEL_DIR / LLAMA_MODEL_NAME
# LLAMA_MODEL_URL = f"https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/{LLAMA_MODEL_NAME}"

# ---------------- Prompt ----------------
PROMPT_DIR = Path(__file__).parent / 'themex' /'prompts'
PROMPT_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- Logs ----------------
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True, parents=True)