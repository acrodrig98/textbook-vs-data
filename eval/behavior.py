"""Score every Week 1 prompt: does the model's next token favor the data's answer or the alternative?

For each prompt, this records the next-token logits of `target_data` (the answer the data supports)
and `target_alt` (the textbook law in conflict items, a made-up formula in congruent items).
logit_diff = logit(target_data) - logit(target_alt) > 0 means the model sides with the data.

It uses the plain Hugging Face model. Transcoders aren't needed until the attribution-graph step,
and loading straight onto the GPU avoids the large RAM spike that circuit-tracer's loader causes.

Usage (from the repo root, on a GPU):
  python -m prompts.generate --out data/week1.jsonl
  python -m eval.behavior --prompts data/week1.jsonl --model google/gemma-2-2b
  python -m eval.behavior --limit 50          # quick check on the first 50 prompts
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import torch

KEEP_FIELDS = (
    "id", "base_id", "law", "prior_level", "format", "order", "congruent", "alt_kind",
    "k", "p_textbook", "p_data", "p_alt", "shift", "n_rows", "n_shots",
    "target_data", "target_alt", "formula_data", "formula_alt", "generator_version", "seed",
)


def load_items(path: Path, limit: int | None = None) -> list[dict]:
    with path.open() as f:
        items = [json.loads(line) for line in f]
    return items[:limit] if limit else items


def target_token_ids(tokenizer, items: list[dict]) -> dict[str, int]:
    """Map each target string (" A", " B", "1".."9") to its single token id, or fail loudly."""
    ids = {}
    for s in sorted({it[key] for it in items for key in ("target_data", "target_alt")}):
        toks = tokenizer.encode(s, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"Target {s!r} is {len(toks)} tokens for this tokenizer; it must be exactly 1.")
        ids[s] = toks[0]
    return ids


@torch.no_grad()
def next_token_logits(model, tokenizer, prompts: list[str], device) -> torch.Tensor:
    """Logits at the last real token of each prompt. Right padding keeps positions correct."""
    tokenizer.padding_side = "right"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    logits = model(**enc).logits
    last = enc["attention_mask"].sum(dim=1) - 1
    return logits[torch.arange(len(prompts), device=logits.device), last].float()


def score_items(model, tokenizer, items: list[dict], batch_size: int = 16, device="cuda") -> list[dict]:
    tok_ids = target_token_ids(tokenizer, items)
    results = []
    t0 = time.time()
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        logits = next_token_logits(model, tokenizer, [it["prompt"] for it in batch], device)
        log_probs = logits.log_softmax(-1)
        top1 = logits.argmax(-1)
        for row, it in enumerate(batch):
            i_d, i_a = tok_ids[it["target_data"]], tok_ids[it["target_alt"]]
            lp_d, lp_a = log_probs[row, i_d].item(), log_probs[row, i_a].item()
            results.append({
                **{k: it.get(k) for k in KEEP_FIELDS},
                "logit_data": logits[row, i_d].item(),
                "logit_alt": logits[row, i_a].item(),
                "logit_diff": logits[row, i_d].item() - logits[row, i_a].item(),
                # how much of the model's probability lands on the two answers at all
                "p_two_targets": float(torch.tensor([lp_d, lp_a]).exp().sum()),
                "top1_token": tokenizer.decode([int(top1[row])]),
            })
        done = min(start + batch_size, len(items))
        if done == len(items) or (start // batch_size) % 20 == 0:
            print(f"  {done}/{len(items)} prompts scored ({time.time() - t0:.0f}s)", flush=True)
    return results


def load_model(name: str, dtype: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = dict(dtype=getattr(torch, dtype), low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = "cuda"
    if "gemma-2" in name:
        kwargs["attn_implementation"] = "eager"  # recommended for Gemma 2 (attention soft-capping)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if device != "cuda":
        model = model.to(device)
    return model.eval(), tokenizer


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, default=Path("data/week1.jsonl"))
    ap.add_argument("--model", default="google/gemma-2-2b")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"],
                    help="float16 can overflow in Gemma 2; use bfloat16 or float32")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N prompts")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: results/<prompts stem>_<model>.jsonl")
    args = ap.parse_args(argv)

    out = args.out or Path("results") / f"{args.prompts.stem}_{args.model.split('/')[-1]}.jsonl"
    items = load_items(args.prompts, args.limit)
    print(f"Loading {args.model} ({args.dtype}) on {args.device} ...", flush=True)
    t0 = time.time()
    model, tokenizer = load_model(args.model, args.dtype, args.device)
    print(f"Loaded in {time.time() - t0:.0f}s. Scoring {len(items)} prompts.", flush=True)

    results = score_items(model, tokenizer, items, args.batch_size, args.device)
    meta = {
        "model": args.model, "dtype": args.dtype, "prompts_file": str(args.prompts),
        "n_prompts": len(results), "date": dt.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
    }
    try:
        import transformers
        meta["transformers"] = transformers.__version__
    except ImportError:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps({**r, "meta": meta}) + "\n")
    print(f"Wrote {out}. Next: python -m eval.analyze {out}")


if __name__ == "__main__":
    main()
