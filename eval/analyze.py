"""Turn Week 1 scores into phase diagrams and the go/no-go numbers.

Decisions are made per data table. For multiple choice, the logit difference is averaged over the
two letter orders, which cancels any preference for " A" or " B". The model "sides with the data"
when that average is above 0. For numeric prompts there is one decision per table.

Outputs:
  figures/<stem>_phase.png    - heatmaps: fraction of conflict tables where the model sides with the data,
                                by number of rows x exponent shift, per prior level and format
  figures/<stem>_levels.png   - siding-with-data rate for conflict tables vs accuracy on congruent tables,
                                by prior level
  results/<stem>_cells.csv    - the numbers behind the heatmaps
  printed summary             - letter bias, format compliance, prior pull with a 95% bootstrap CI, go/no-go

Usage:
  python -m eval.analyze results/week1_gemma-2-2b.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["named_law", "names_only", "anonymous"]
LEVEL_LABELS = {"named_law": "Law named", "names_only": "Variable names only", "anonymous": "Anonymous x, y"}
FORMAT_LABELS = {"mc": "Multiple choice", "numeric": "Numeric prediction"}

# Reference palette (dataviz skill): sequential blue ramp, chart chrome, categorical slots 1-2
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"

# Suggested Week 1 go/no-go thresholds (see the plan): the anonymous control must work,
# and naming the law must pull decisions toward the textbook by a clear margin.
GO_MIN_ANON_ACCURACY = 0.70
GO_MIN_PULL = 0.10


def load_results(path: Path) -> pd.DataFrame:
    with path.open() as f:
        df = pd.DataFrame([json.loads(line) for line in f])
    return df.drop(columns=["meta"], errors="ignore")


def letter_bias(df: pd.DataFrame) -> float:
    """Mean logit(" A") - logit(" B") over MC prompts. Positive = the model prefers " A"."""
    mc = df[df["format"] == "mc"]
    if mc.empty:
        return float("nan")
    logit_a = np.where(mc["target_data"] == " A", mc["logit_data"], mc["logit_alt"])
    logit_b = np.where(mc["target_data"] == " A", mc["logit_alt"], mc["logit_data"])
    return float(np.mean(logit_a - logit_b))


def table_decisions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (data table, prior level, format), with the order-averaged logit difference."""
    keys = ["base_id", "prior_level", "format"]
    agg = (df.groupby(keys)
             .agg(logit_diff=("logit_diff", "mean"), n_orders=("logit_diff", "size"),
                  law=("law", "first"), congruent=("congruent", "first"), shift=("shift", "first"),
                  n_rows=("n_rows", "first"), p_two_targets=("p_two_targets", "mean"))
             .reset_index())
    agg["sides_with_data"] = agg["logit_diff"] > 0
    return agg


def rates(dec: pd.DataFrame) -> pd.DataFrame:
    """Siding-with-data rate by format x prior level x condition."""
    out = (dec.assign(condition=np.where(dec["congruent"], "congruent", "conflict"))
              .groupby(["format", "prior_level", "condition"])["sides_with_data"]
              .agg(["mean", "size"]).reset_index())
    return out.rename(columns={"mean": "rate", "size": "n_tables"})


def prior_pull(dec: pd.DataFrame, fmt: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float, int]:
    """Anonymous minus law-named siding-with-data rate on conflict tables, with a paired bootstrap CI.

    Each table appears at both levels with identical data, so the difference is computed per table
    and the tables are resampled together.
    """
    d = dec[(dec["format"] == fmt) & (~dec["congruent"])]
    wide = d.pivot_table(index="base_id", columns="prior_level", values="sides_with_data", aggfunc="first")
    wide = wide.dropna(subset=["anonymous", "named_law"])
    if wide.empty:
        return float("nan"), float("nan"), float("nan"), 0
    per_table = wide["anonymous"].astype(float).to_numpy() - wide["named_law"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(per_table, size=(n_boot, len(per_table)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(per_table.mean()), float(lo), float(hi), len(per_table)


def phase_cells(dec: pd.DataFrame) -> pd.DataFrame:
    conflict = dec[~dec["congruent"]]
    return (conflict.groupby(["format", "prior_level", "shift", "n_rows"])["sides_with_data"]
                    .agg(["mean", "size"]).reset_index()
                    .rename(columns={"mean": "rate", "size": "n_tables"}))


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)


def plot_phase(cells: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cmap.set_bad("#f0efec")
    formats = [f for f in ("mc", "numeric") if f in set(cells["format"])]
    shifts = sorted(cells["shift"].unique())
    rows = sorted(cells["n_rows"].unique())

    fig, axes = plt.subplots(len(formats), 3, figsize=(11, 3.2 * len(formats) + 0.9),
                             squeeze=False, facecolor=SURFACE)
    for i, fmt in enumerate(formats):
        for j, level in enumerate(LEVELS):
            ax = axes[i][j]
            sub = cells[(cells["format"] == fmt) & (cells["prior_level"] == level)]
            grid = np.full((len(shifts), len(rows)), np.nan)
            counts = np.zeros_like(grid)
            for _, r in sub.iterrows():
                grid[shifts.index(r["shift"]), rows.index(r["n_rows"])] = r["rate"]
                counts[shifts.index(r["shift"]), rows.index(r["n_rows"])] = r["n_tables"]
            im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1,
                           origin="lower", aspect="auto")
            for (y, x), v in np.ndenumerate(grid):
                if not np.isnan(v):
                    ax.text(x, y, f"{v:.0%}", ha="center", va="center", fontsize=9,
                            color="#ffffff" if v > 0.6 else INK)
            # 2px surface gap between cells
            ax.set_xticks(np.arange(-0.5, len(rows)), minor=True)
            ax.set_yticks(np.arange(-0.5, len(shifts)), minor=True)
            ax.grid(which="minor", color=SURFACE, linewidth=2)
            ax.tick_params(which="minor", length=0)
            ax.set_xticks(range(len(rows)), [str(r) for r in rows])
            ax.set_yticks(range(len(shifts)), [f"{s:g}" for s in shifts])
            _style_axes(ax)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(LEVEL_LABELS[level], color=INK, fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(f"{FORMAT_LABELS[fmt]}\nexponent shift", color=INK_2, fontsize=10)
            if i == len(formats) - 1:
                ax.set_xlabel("rows of data", color=INK_2, fontsize=10)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("conflict tables where the model sides with the data", color=INK_2, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_levels(rate_df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    formats = [f for f in ("mc", "numeric") if f in set(rate_df["format"])]
    fig, axes = plt.subplots(1, len(formats), figsize=(5.2 * len(formats), 3.6), squeeze=False,
                             facecolor=SURFACE, sharey=True)
    x = np.arange(len(LEVELS))
    series = [("congruent", "Congruent control: picks the data's formula", SERIES_2),
              ("conflict", "Conflict: sides with the data over the textbook law", SERIES_1)]
    for ax, fmt in zip(axes[0], formats):
        _style_axes(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ys_by_cond = {}
        for cond, label, color in series:
            sub = rate_df[(rate_df["format"] == fmt) & (rate_df["condition"] == cond)].set_index("prior_level")
            ys_by_cond[cond] = [sub["rate"].get(level, np.nan) for level in LEVELS]
            ax.plot(x, ys_by_cond[cond], color=color, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        # Label the higher point of each pair above it and the lower one below, so labels never collide
        for xi in x:
            pair = {c: ys_by_cond[c][xi] for c in ys_by_cond if not np.isnan(ys_by_cond[c][xi])}
            top = max(pair, key=pair.get) if pair else None
            for cond, yi in pair.items():
                above = cond == top
                ax.annotate(f"{yi:.0%}", (xi, yi), textcoords="offset points",
                            xytext=(0, 9 if above else -16), ha="center", fontsize=9, color=INK_2)
        ax.margins(x=0.12)
        ax.axhline(0.5, color=AXIS, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xticks(x, [LEVEL_LABELS[level] for level in LEVELS])
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(FORMAT_LABELS[fmt], color=INK, fontsize=11, loc="left")
    axes[0][0].set_ylabel("share of data tables", color=INK_2, fontsize=10)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, -0.08), ncol=2,
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(title + "   (dashed line = chance)", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def summarize(df: pd.DataFrame, dec: pd.DataFrame, rate_df: pd.DataFrame, model: str) -> str:
    lines = [f"## Week 1 behavior: {model}", ""]
    lines.append(f"Prompts scored: {len(df)}  |  data tables: {dec['base_id'].nunique()}")
    lines.append(f"Letter bias (mean logit ' A' - ' B'): {letter_bias(df):+.2f}  "
                 "(cancelled out by averaging over both letter orders)")
    for fmt in ("mc", "numeric"):
        sub = dec[dec["format"] == fmt]
        if not sub.empty:
            lines.append(f"Format compliance, {fmt}: on average {sub['p_two_targets'].mean():.1%} of next-token "
                         "probability is on the two answer tokens")
    lines.append("")
    lines.append("Siding-with-data rate by prior level (conflict = data vs textbook law; congruent = control):")
    lines.append(f"{'format':9} {'condition':10} " + " ".join(f"{LEVEL_LABELS[lv]:>20}" for lv in LEVELS))
    for fmt in ("mc", "numeric"):
        for cond in ("congruent", "conflict"):
            sub = rate_df[(rate_df["format"] == fmt) & (rate_df["condition"] == cond)].set_index("prior_level")
            if sub.empty:
                continue
            cells = " ".join(f"{sub['rate'].get(lv, np.nan):>19.0%} " for lv in LEVELS)
            lines.append(f"{fmt:9} {cond:10} {cells}")
    lines.append("")
    lines.append("Go/no-go (suggested thresholds: anonymous accuracy >= "
                 f"{GO_MIN_ANON_ACCURACY:.0%} on both conditions; prior pull >= {GO_MIN_PULL:.0%} with CI above 0):")
    for fmt in ("mc", "numeric"):
        sub = rate_df[(rate_df["format"] == fmt) & (rate_df["prior_level"] == "anonymous")].set_index("condition")
        if sub.empty:
            continue
        acc_cong, acc_conf = sub["rate"].get("congruent", np.nan), sub["rate"].get("conflict", np.nan)
        pull, lo, hi, n = prior_pull(dec, fmt)
        competent = acc_cong >= GO_MIN_ANON_ACCURACY and acc_conf >= GO_MIN_ANON_ACCURACY
        effect = pull >= GO_MIN_PULL and lo > 0
        verdict = "GO" if competent and effect else ("NO-GO: model can't do the anonymous task" if not competent
                                                     else "NO-GO: no clear prior pull")
        lines.append(f"  {fmt:8} anonymous accuracy {acc_cong:.0%} (congruent) / {acc_conf:.0%} (conflict); "
                     f"prior pull {pull:+.0%} [95% CI {lo:+.0%}, {hi:+.0%}] over {n} tables -> {verdict}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = ap.parse_args(argv)

    df = load_results(args.results)
    with args.results.open() as f:
        model = json.loads(f.readline()).get("meta", {}).get("model", args.results.stem)
    dec = table_decisions(df)
    rate_df = rates(dec)
    cells = phase_cells(dec)
    stem = args.results.stem

    cells.to_csv(args.results.with_name(f"{stem}_cells.csv"), index=False)
    plot_phase(cells, args.figures_dir / f"{stem}_phase.png",
               f"When does {model} side with the data over the textbook law?")
    plot_levels(rate_df, args.figures_dir / f"{stem}_levels.png", f"{model}: effect of naming the law")
    report = summarize(df, dec, rate_df, model)
    args.results.with_name(f"{stem}_summary.md").write_text(report + "\n")
    print(report)
    print(f"\nFigures: {args.figures_dir}/{stem}_phase.png, {args.figures_dir}/{stem}_levels.png")


if __name__ == "__main__":
    main()
