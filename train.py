
#!/usr/bin/env python3
"""A minimal local LoRA fine-tuning experiment for Apple Silicon."""

import json
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "adapters"

# A small teaching dataset. Replace these examples with your own later.
EXAMPLES = [
    ("What is a chargeback?",
     "A chargeback is a payment reversal initiated by a cardholder's bank."),
    ("What is a false positive in fraud detection?",
     "A false positive is a legitimate transaction incorrectly flagged as fraud."),
    ("What is a false negative in fraud detection?",
     "A false negative is fraudulent activity incorrectly classified as legitimate."),
    ("What is two-factor authentication?",
     "Two-factor authentication verifies identity using two different types of evidence."),
    ("Why monitor transaction velocity?",
     "Transaction velocity can reveal unusually frequent payments that may indicate fraud."),
    ("What is an account takeover?",
     "An account takeover occurs when an unauthorized person gains control of another user's account."),
    ("What is a suspicious login?",
     "A suspicious login is an authentication event with unusual signals, such as a new device or location."),
    ("What is a fraud detection rule?",
     "A fraud detection rule evaluates transaction signals against conditions to identify potentially risky activity."),
    ("What does KYC mean?",
     "KYC means Know Your Customer, a process used to verify a customer's identity."),
    ("What is an anomaly in transaction data?",
     "An anomaly is a transaction or pattern that differs significantly from expected behavior."),
    ("What is precision in fraud detection?",
     "Precision is the proportion of alerts classified as fraud that are actually fraudulent."),
    ("What is recall in fraud detection?",
     "Recall is the proportion of actual fraudulent cases that the system correctly identifies."),
]

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_DIR = ROOT / "models" / MODEL_ID.split("/")[-1]
QUESTION = "What is a false positive in fraud detection?"


def ensure_model() -> Path:
    """Keep one local copy of the model so later runs never re-download it."""
    has_weights = (MODEL_DIR / "model.safetensors").exists()
    has_sharded_weights = (MODEL_DIR / "model.safetensors.index.json").exists()
    if has_weights or has_sharded_weights:
        return MODEL_DIR

    print(f"Downloading {MODEL_ID} to {MODEL_DIR} (one time only)...")
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    return MODEL_DIR


def save_jsonl(path: Path, examples):
    """Write MLX-LM's chat dataset format."""
    with path.open("w", encoding="utf-8") as f:
        for question, answer in examples:
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise assistant specializing "
                            "in fraud detection concepts."
                        ),
                    },
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(command):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def generate(model_path: Path, prompt: str, use_adapter: bool):
    command = [
        "mlx_lm.generate",
        "--model", str(model_path),
        "--prompt", prompt,
        "--max-tokens", "100",
    ]

    if use_adapter:
        command += ["--adapter-path", str(ADAPTER_DIR)]

    run(command)


def main():
    DATA_DIR.mkdir(exist_ok=True)
    ADAPTER_DIR.mkdir(exist_ok=True)

    model_path = ensure_model()

    # Hold out two examples for validation and testing.
    save_jsonl(DATA_DIR / "train.jsonl", EXAMPLES[:8])
    save_jsonl(DATA_DIR / "valid.jsonl", EXAMPLES[8:10])
    save_jsonl(DATA_DIR / "test.jsonl", EXAMPLES[10:])

    prompt = (
        "<|im_start|>system\n"
        "You are a concise assistant specializing in fraud detection concepts."
        "<|im_end|>\n"
        f"<|im_start|>user\n{QUESTION}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    print("\n=== 1. BASELINE: BEFORE TRAINING ===")
    generate(model_path, prompt, use_adapter=False)

    print("\n=== 2. TRAINING LoRA ADAPTERS ===")
    run([
        "mlx_lm.lora",
        "--model", str(model_path),
        "--train",
        "--data", str(DATA_DIR),
        "--iters", "30",
        "--batch-size", "1",
        "--adapter-path", str(ADAPTER_DIR),
    ])

    print("\n=== 3. FINE-TUNED ANSWER ===")
    generate(model_path, prompt, use_adapter=True)

    print("\nExperiment complete.")
    print(f"Training data: {DATA_DIR}")
    print(f"Model: {model_path}")
    print(f"Saved adapters: {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
