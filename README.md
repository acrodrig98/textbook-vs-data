# Textbook vs. Data

When an LLM sees data that contradicts a physical law it has memorized, which one wins? Where is that conflict resolved inside the model, and does the model's stated reasoning reflect it?

This is Project 1 of the interpretability research plan. It uses attribution graphs (circuit-tracer) on Gemma-2-2B and Qwen3-4B, then Gemma 3 with the Gemma Scope 2 transcoders.

## Setup (on a GPU machine)
1. Accept the Gemma license at https://huggingface.co/google/gemma-2-2b and create a read token.
2. `bash setup.sh`: creates `.venv`, installs `requirements.txt`, runs the Hugging Face login and `tests/`.
3. `python smoke_test.py`: loads Gemma-2-2B with its transcoders, runs the Dallas→Austin check, prints how numbers tokenize, and compares a named (Ohm's law) prompt against an anonymized one.

## Week 1: generate the prompts
```
python -m prompts.generate --out data/week1.jsonl --preview 3
```
This writes 4,347 prompts covering 483 data tables. The defaults are: 5 laws (Ohm, Hooke, free fall, Coulomb, Kepler III); 1–4 table rows; 3 prior-strength levels; multiple-choice in both letter orders, plus numeric; seed 0. Options: `--rows`, `--laws`, `--formats`, `--shots 2` (adds two anonymous worked examples), `--seed`. Each record lists the token for the answer the data supports (`target_data`) and the token for the alternative (`target_alt`). `data/` is gitignored because the seed makes it reproducible.

## Week 1: score the prompts and make the phase diagrams (GPU)
Easiest: open `notebooks/week1_colab.ipynb` in Colab and run the cells. Or, on any GPU machine:
```
python -m eval.behavior --model google/gemma-2-2b --prompts data/week1.jsonl   # about 4,300 prompts
python -m eval.analyze results/week1_gemma-2-2b.jsonl
```
`eval.behavior` uses the plain Hugging Face model, because transcoders aren't needed yet. For each prompt it records the logits of the token for the answer the data supports and the token for the alternative. `eval.analyze` averages multiple-choice items over both letter orders, which cancels letter bias. It then writes `figures/*_phase.png` (heatmaps: rows of data x exponent shift, one per prior level and format), `figures/*_levels.png`, `results/*_cells.csv` and `results/*_summary.md`, and prints the go/no-go numbers.

**Made-up-law control.** `--levels fake_named` renders the law-named prompts with only the law's name swapped for an invented one (e.g. "A Varnek's law experiment"). Data tables depend only on the seed, never on the levels, so these prompts line up with earlier results, and `eval.analyze` can merge files from the same model:
```
python -m prompts.generate --out data/week1_fakelaw.jsonl --levels fake_named --formats mc
python -m eval.behavior --model Qwen/Qwen3-4B --prompts data/week1_fakelaw.jsonl
python -m eval.analyze results/week1_Qwen3-4B.jsonl results/week1_fakelaw_Qwen3-4B.jsonl --name week1_control_Qwen3-4B
```
`--levels setting_only` does the same with the law's name removed and the setup description kept word for word. With all five levels merged, the summary splits the textbook pull into steps between neighbouring levels: the law's name (real to made-up), generic "X's law" framing (made-up to setup only), the setup description (setup only to variable names), and the variable names (to anonymous).

## Layout
```
prompts/     prompt generator: laws, exponent shifts, prior-strength levels, A/B order swaps
eval/        batched logit differences → phase diagrams
graphs/      attribution graph outputs (gitignored; they're large)
notebooks/   exploration
figures/     figures for the write-up
tests/       environment checks that run anywhere
requirements-lock-cpu.txt   exact versions from the CPU test install (a record only; on a GPU machine install from requirements.txt)
LOG.md       research log: Monday hypothesis, Friday update
```
