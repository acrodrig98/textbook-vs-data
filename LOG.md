# Research log

Each Monday, write the week's hypothesis and a kill/pivot criterion. Each Friday, write a one-paragraph update and add your best figure.

## Week 1 (Nov 23–29, Thanksgiving week, light)
**Hypothesis:** For the same data, naming the law in the prompt (e.g., "Ohm's law experiment") shifts Gemma-2-2B toward the textbook answer, compared with anonymized x/y prompts.
**Kill/pivot:** If the model can't pick the formula that fits anonymized prompts above chance, simplify the task (linear vs. quadratic only, smaller numbers) or switch to Qwen3-4B.

**Friday update (written Nov 25):**
- **Setup:** works on Colab T4 with High-RAM. Free Colab ran out of RAM while circuit-tracer loaded Gemma. Gemma-2-2B tokenizes digits one at a time, and " A"/" B" are single tokens.
- **Runs:** 4,347 prompts from 483 data tables (seed 0), plus a multiple-choice set with 2 worked examples (`--shots 2`), on Gemma-2-2B and Qwen3-4B. All results are in `results/`, the figures in `figures/`.
- **Gemma-2-2B: no-go.** Multiple choice isn't driven by the data: the top token was " B" on all 966 anonymous prompts, and the gap between the two answers averaged ~0.04 logits. With 2 worked examples it answers in the right format, but it's still below chance on the anonymous control (37%). Numeric prediction works (~70% sides with the data), but naming the law makes no difference: pull +3 pts, 95% CI [-1, +7]. Its errors follow the exponent, not the wording.
- **Qwen3-4B, multiple choice, no examples: GO.** The anonymous control is 84% (congruent) / 82% (conflict). Naming the law pulls it toward the textbook: **+39 pts [34, 45]**, positive for every law (Coulomb +23, free fall +77, Hooke +22, Kepler +21, Ohm +49). For conflict prompts with the law named, the model sides with the data 13 / 41 / 52 / 61% with 1-4 rows (anonymous: 62 / 80 / 89 / 96%), so more data gradually overcomes the textbook. With variable names but no law named, conflict is 74%, so the pull comes mainly from the law's name.
- **Qwen3-4B, numeric:** small pull, +7 pts [3, 11], significant only for free fall and Ohm. The textbook pull is about choosing a formula, not about extending the table.
- **Qwen3-4B with 2 worked examples:** the pull drops to +16 [12, 21] and new biases appear, so prompts without examples are the cleaner setup.
- **Caveats:** Qwen gets Coulomb conflict prompts right only 34% of the time when anonymous (probably the large numbers), and Kepler's anonymous control is at 50%.
- **Decision:** the main model is now **Qwen3-4B with multiple-choice prompts and no examples**, focusing on free fall and Ohm. Gemma-2-2B becomes a negative result: the effect appears with scale.
- **Next:** a made-up-law control (e.g. "Henderson's law"), to tell framing apart from recall of the specific law; a first Neuronpedia graph on a free-fall conflict prompt; then pick ~10 matched sets (law named with 1-2 rows / the same table anonymized / law named with 4 rows) for attribution graphs.
- **Best figure:** `figures/week1_Qwen3-4B_phase.png` (top-left panel).

**Control, made-up law names (Nov 25):** same tables and wording as the law-named prompts, with only the law's name invented (Varnek, Tolliver, Corvel, Dresmond, Halvane). Qwen3-4B, multiple choice, conflict prompts, change in siding with the data: real name -> made-up name **+17 pts [13, 21]**; made-up name -> variable names only +13 [9, 18]; variable names -> anonymous +9 [5, 13]; total +39 [34, 45]. The made-up name also lowers accuracy on congruent prompts (94% -> 84%), so knowing the law helps when the data agrees and misleads when it doesn't. The source differs by law: Ohm's pull is mostly the name (+28); free fall's is mostly the setup description ("a dropped ball falls", +61), and linear data in a free-fall setup is never accepted, even with a made-up name; Hooke's is mostly the variable names (+19). Next: a setup-only level (name removed, description kept) to separate generic "X's law" framing from the description. Figures: `figures/week1_control_Qwen3-4B_*.png`.
