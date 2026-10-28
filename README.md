# Model fine-tuning example

A minimal, end-to-end **LoRA fine-tuning** experiment for Apple Silicon, built with
[MLX](https://github.com/ml-explore/mlx) and `mlx-lm`.

It fine-tunes **Qwen2.5-0.5B-Instruct (4-bit)** on a tiny fraud-detection Q&A set,
then compares the model's answer before and after training.

> This is a learning demo, not a production model. The dataset has 12 examples and
> training runs for 30 iterations — enough to show the full workflow, not enough to
> create a genuinely specialized model.

---

## What the script does

`train.py` runs five steps:

1. **Creates the dataset** — writes 12 question/answer pairs to MLX chat-format
   JSONL files: 8 for training, 2 for validation, 2 for testing.
2. **Prepares the model** — downloads Qwen2.5 0.5B (4-bit) once into `models/`
   (~276 MB) and reuses that local copy on later runs.
3. **Generates a baseline answer** — asks "What is a false positive in fraud
   detection?" *before* any training.
4. **Fine-tunes with LoRA** — trains small adapter weights while leaving the base
   model frozen.
5. **Generates the tuned answer** — asks the same question using the trained
   adapter. The adapter itself is saved in `adapters/` during training.

The useful comparison is the difference between steps 3 and 5.

---

## Quick start

**Requirements:** macOS on Apple Silicon (M1/M2/M3/M4), Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

The first run downloads the model to `models/Qwen2.5-0.5B-Instruct-4bit/`.
Later runs reuse it. If you delete `models/`, it is recreated from the Hugging Face
cache, or downloaded again if that cache is also empty.

---

## Model at a glance

| Property | Value |
|---|---|
| Model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Base model | Qwen2.5-0.5B-Instruct |
| Parameters | ~0.49B |
| Quantization | 4-bit, group size 64 |
| Context window | **32,768 tokens** |
| Training sequence limit | **2,048 tokens** (configured in `adapter_config.json`) |
| Output limit in this script | **100 tokens** (`--max-tokens 100`) |
| Architecture | 24 layers, hidden size 896, 14 attention heads, 2 KV heads (GQA) |
| Vocabulary | 151,936 tokens |
| Chat format | ChatML: `<|im_start|>role ... <|im_end|>` |
| On-disk size | ~276 MB |
| License | Apache-2.0 |

**Important distinction:** the model can accept up to 32K tokens of context, but this
experiment trains on at most 2,048 tokens per example and generates at most 100 new
tokens. Those are script settings, not model limits. The 0.5B variant is a 32K model;
larger Qwen2.5 variants support up to 128K.

---

## Training configuration

| Parameter | Value | Why it matters |
|---|---|---|
| Method | LoRA (SFT) | Trains a small adapter, not the full model |
| Iterations | 30 | Tiny demo run; roughly 30 optimizer steps |
| Batch size | 1 | Low memory use |
| Learning rate | `1e-5` | Conservative; typical LoRA range is `1e-5` to `2e-4` |
| LoRA rank | 8 | Controls adapter capacity |
| LoRA scale | 20 | Scaling factor for adapter updates |
| LoRA dropout | 0.0 | No dropout in this run |
| Adapted layers | 16 of 24 | Only part of the network gets adapters |
| Max sequence length | 2,048 | Training examples are truncated beyond this |
| Optimizer | Adam | Default optimizer |
| Seed | 0 | Reproducibility |

Outputs:

| Path | Contents |
|---|---|
| `data/train.jsonl` | 8 training examples |
| `data/valid.jsonl` | 2 validation examples |
| `data/test.jsonl` | 2 test examples (written but not scored by this script) |
| `models/Qwen2.5-0.5B-Instruct-4bit/` | Local model copy (gitignored) |
| `adapters/adapter_config.json` | Training configuration |
| `adapters/adapters.safetensors` | Trained LoRA weights (~11 MB) |

---

## How to read the output

```text
=== 1. BASELINE: BEFORE TRAINING ===
...
=== 2. TRAINING LoRA ADAPTERS ===
...
=== 3. FINE-TUNED ANSWER ===
...
```

- **Baseline** shows what the pretrained model already knows.
- **Fine-tuned** shows the effect of the 30-step adapter.
- With only 12 examples, expect a small style/format shift, not a dramatic knowledge
  gain. That is normal.

---

## Limitations

- 12 examples are far too few for reliable fine-tuning.
- 30 iterations is a smoke test, not a real training run.
- `test.jsonl` exists but the script does not evaluate it.
- Validation only runs every 200 steps, so this 30-step run produces no useful
  validation curve.
- The prompt is built manually in ChatML; it is not using a generic chat template.
- No safety, factuality, or regression evaluation is included.

---

## Next steps

1. **Build an evaluation set first.** Create 20–50 held-out questions and score the
   base and tuned model on the same set. Without this, you cannot tell whether
   training helped.
2. **Tune hyperparameters.** Try LoRA rank 16/32, learning rates `1e-5` to `2e-4`,
   100–500 iterations, and different adapted-layer counts. Change one variable at a
   time and record the result.
3. **Grow and clean the dataset.** Aim for hundreds to thousands of consistent,
   deduplicated examples. Check licenses and remove PII.
4. **Try bigger models.** Run the same experiment with Qwen2.5 1.5B, 3B, or 7B and
   compare quality, latency, and memory. QLoRA lets you fine-tune larger models on
   limited hardware.
5. **Learn evaluation and tracking.** Track dataset versions, adapter versions, and
   scores with tools such as MLflow or Weights & Biases.
6. **Serve and benchmark the adapter.** Fuse it with `mlx_lm.fuse` for standalone
   deployment, then measure tokens/sec, time-to-first-token, and memory with
   `mlx_lm.server`, Ollama, or llama.cpp.
7. **Learn RAG next.** Retrieval-augmented generation is often a better fit than
   fine-tuning for changing or private factual knowledge.
8. **Then explore alignment.** After SFT, learn DPO/ORPO using chosen/rejected answer
   pairs if you need preference or safety behavior.

---

## Retraining methods used in industry

Most industry "retraining" is not training a model from scratch — it is adapting an
existing foundation model with one of these methods.

| Method | What it updates | Best for | Trade-off |
|---|---|---|---|
| **Training from scratch** | All weights on a very large corpus | A new foundation model or major domain shift | Extremely expensive; rarely the first move |
| **Full fine-tuning** | All model weights | Maximum quality when you have lots of data and compute | Expensive; risk of catastrophic forgetting |
| **LoRA** | Small low-rank adapter matrices | Most task/style adaptation | Cheap and portable; slightly less capacity than full FT |
| **QLoRA** | LoRA on a 4-bit base model | Fine-tuning larger models on limited memory | Small quality/compute overhead |
| **DoRA** | Magnitude and direction of weights | Better quality than LoRA at similar cost | More complex, slightly slower |
| **Prefix / prompt tuning / IA3** | Prompts or small vectors | Many tasks with very few parameters | Less common for chat models |
| **Continued pretraining** | All weights on domain text | New domain, language, or vocabulary | Needs a large unlabeled corpus |
| **SFT (supervised fine-tuning)** | All weights or adapters | Teaching format, tone, and task behavior | Needs high-quality input/output pairs |
| **DPO / ORPO / KTO** | Model or adapters | Preference alignment without a full RL loop | Needs chosen/rejected answer pairs |
| **RLHF (PPO)** | Policy model + reward model | Nuanced safety and preference alignment | Complex and expensive to operate |
| **Distillation** | Student model | Smaller, faster production models | Requires a strong teacher and data |
| **RAG** | No weights | Changing, private, or citation-heavy knowledge | Retrieval quality becomes the bottleneck |

### Choosing an approach

| Goal | Start with |
|---|---|
| Add new or changing facts | **RAG**, not fine-tuning |
| Change tone, format, or task behavior | **LoRA / SFT** |
| Teach a new domain or language | **Continued pretraining**, then SFT |
| Align answers with human preferences | **DPO** (simpler) or RLHF (more control) |
| Reduce cost and latency | **Distillation + quantization** |
| Maximize quality with ample data/compute | **Full fine-tuning** |

---

## Files

```text
model-fine-tuning-example/
├── train.py                    # End-to-end experiment
├── data/                       # Generated train/valid/test JSONL
├── models/                     # Local model copy (gitignored, ~276 MB)
├── adapters/                   # Trained LoRA adapter (~11 MB)
├── requirements.txt            # mlx-lm[train]
└── README.md
```
