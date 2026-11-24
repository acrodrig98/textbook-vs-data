"""Day-1 smoke test for Project 1 (Textbook vs. Data).

Checks three things on Gemma-2-2B:
  1. The model and transcoders load, and a known two-hop prompt gives the expected answer.
  2. How numbers and answer letters are split into tokens, which matters for designing the task.
  3. A first look at the effect itself: an anonymized prompt vs. one that names Ohm's law.

Run it on a CUDA GPU after setup.sh:  python smoke_test.py
"""
import time

import torch
from circuit_tracer import ReplacementModel

assert torch.cuda.is_available(), "No CUDA GPU visible - run this on a GPU machine."
print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading Gemma-2-2B + transcoders. After the downloads finish, it can stay quiet for")
print("several minutes while it builds the model and moves it to the GPU. Don't stop it.", flush=True)
t0 = time.time()
# bfloat16 as in the official demos; the float32 default does not fit on a 15GB GPU
model = ReplacementModel.from_pretrained("google/gemma-2-2b", "gemma", dtype=torch.bfloat16)
tok = model.tokenizer
print(f"Model loaded in {time.time() - t0:.0f}s. "
      f"GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)


def next_token_logits(prompt: str) -> torch.Tensor:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        return model(tokens)[0, -1].float()


def top_k(prompt: str, k: int = 5):
    probs = next_token_logits(prompt).softmax(-1)
    top = probs.topk(k)
    return [(tok.decode([int(i)]), round(p.item(), 3)) for p, i in zip(top.values, top.indices)]


def logit_diff(prompt: str, a: str = " A", b: str = " B") -> float:
    """Positive means the model prefers option A."""
    ids_a, ids_b = tok.encode(a, add_special_tokens=False), tok.encode(b, add_special_tokens=False)
    assert len(ids_a) == 1 and len(ids_b) == 1, f"{a!r} or {b!r} is not a single token"
    logits = next_token_logits(prompt)
    return (logits[ids_a[0]] - logits[ids_b[0]]).item()


# 1. Load check: the two-hop example from the attribution-graphs work
print("\n[1] Two-hop check (expect ' Austin' near the top):")
print(top_k("Fact: the capital of the state containing Dallas is"))

# 2. How numbers and answer letters tokenize
print("\n[2] Tokenization:")
for s in [" 1234", " 12.5", " 2x^2", " A", " B", "voltage", " V = 2I"]:
    ids = tok.encode(s, add_special_tokens=False)
    print(f"  {s!r:12} -> {[tok.decode([i]) for i in ids]}")

# 3. Prior vs. data: the same data (y = 2x^2) with and without a textbook law named
DATA_ANON = """Here are measurements of y versus x:
x=1, y=2
x=2, y=8
x=3, y=18
x=4, y=32
Which formula fits the data? A) y = 2x  B) y = 2x^2
Answer:"""

DATA_OHM = """An Ohm's law experiment measured voltage V (volts) versus current I (amps) for a resistor:
I=1, V=2
I=2, V=8
I=3, V=18
I=4, V=32
Which formula fits the data? A) V = 2I  B) V = 2I^2
Answer:"""

print("\n[3] Logit diff A - B (negative = picks B, the answer the data supports):")
print(f"  anonymized: {logit_diff(DATA_ANON):+.2f}")
print(f"  Ohm's law : {logit_diff(DATA_OHM):+.2f}")
print("  (These are single prompts. Swap the A/B order and average over many prompts before trusting this.)")
