# %%
"""
Figure of the learning curves (Reviewer 3, comment 6), from the outputs of
1-model_training/<model>_model/<model>_learning_curve.py.

One panel per model: AUC on the training subsample (in-sample) and on the
held-out fold against the number of training patients (mean and SD over the
outer folds). The x tick labels give the training patients and the events
per feature (EPV) of each training fraction.

Outputs (output/analysis/): learning_curves.png, learning_curves.pdf and
learning_curves.md (the plotted values as a table).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.learning_curve import summarize_learning_curve
from lib.paths import OUTPUT_DIR
from lib.utils import ensure_dir

save_dir = OUTPUT_DIR / "analysis"
ensure_dir(save_dir)

# Validated categorical pair (slots 1 and 2), text and grid inks
SERIES = {
    "test_AUC": ("Held-out fold", "#257de0"),
    "train_AUC": ("Training (in-sample)", "#f47230"),
}
TEXT, TEXT_SECONDARY, GRID = "#191919", "#666666", "#e6e6e6"

# %% Load
curves = {}
for model in ["Admission", "Full"]:
    file = OUTPUT_DIR / f"{model}_model" / f"{model}_learning_curve.csv"
    if file.exists():
        curves[model] = pd.read_csv(file)
    else:
        print(f"skipped {model}: missing {file}")
if not curves:
    raise SystemExit("No learning-curve results found")
summary = summarize_learning_curve(pd.concat(curves.values(), ignore_index=True))

# %% Figure
plt.rcParams.update({"font.size": 9, "axes.edgecolor": TEXT_SECONDARY, "text.color": TEXT,
                     "axes.labelcolor": TEXT, "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY})
fig, axes = plt.subplots(1, len(curves), figsize=(5.6 * len(curves), 4.4), sharey=True, squeeze=False)
low = min(0.5, float((summary["test_AUC_mean"] - summary["test_AUC_sd"].fillna(0)).min()) - 0.02)
for ax, model in zip(axes[0], curves):
    s = summary[summary["model"] == model].sort_values("fraction")
    n_features = int(curves[model]["n_features"].iloc[0])
    n_folds = int(s["n_folds"].max())
    x = s["n_train"]
    for metric, (label, color) in SERIES.items():
        mean, sd = s[f"{metric}_mean"], s[f"{metric}_sd"].fillna(0)
        ax.fill_between(x, mean - sd, (mean + sd).clip(upper=1), color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.5, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1, label=label)
        # Direct label at the end of the line
        ax.annotate(f"{label}\n{mean.iloc[-1]:.2f}", (x.iloc[-1], mean.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TEXT)
    ax.set_xticks(x, [f"{n:.0f}\n{e:.1f}" for n, e in zip(x, s["events_per_feature"])], fontsize=7.5)
    ax.set_xlim(x.min() - 0.03 * x.max(), x.max() * 1.22)
    ax.set_ylim(low, 1.0)
    ax.set_title(f"{model} model ({n_features} features, {n_folds} outer folds)", fontsize=10, loc="left")
    ax.set_xlabel("Training patients (top) and events per feature (bottom)", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0][0].set_ylabel("AUC")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(1, 0.995))
fig.suptitle("Learning curves: AUC by training-set size (mean and SD over the outer folds)",
             x=0.01, y=0.995, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))

# %% Saving
fig.savefig(save_dir / "learning_curves.png", dpi=200)
fig.savefig(save_dir / "learning_curves.pdf")

lines = ["# Learning curves (mean and SD over the outer folds)", "",
         "| model | fraction | training patients | events | events per feature | train AUC | held-out AUC | gap |",
         "|---|---|---|---|---|---|---|---|"]
for _, r in summary.iterrows():
    lines.append(
        f"| {r['model']} | {r['fraction']:.0%} | {r['n_train']:.0f} | {r['n_events']:.0f} | "
        f"{r['events_per_feature']:.1f} | {r['train_AUC_mean']:.3f} ({r['train_AUC_sd']:.3f}) | "
        f"{r['test_AUC_mean']:.3f} ({r['test_AUC_sd']:.3f}) | {r['AUC_gap_mean']:.3f} |"
    )
(save_dir / "learning_curves.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# %%
