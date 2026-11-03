"""One-off helper: pre-download base model weights (resumable via HF cache)."""
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-1.5B-Instruct"
print("downloading model...", flush=True)
AutoModelForCausalLM.from_pretrained(name)
print("downloading tokenizer...", flush=True)
AutoTokenizer.from_pretrained(name)
print("DOWNLOAD_DONE", flush=True)
