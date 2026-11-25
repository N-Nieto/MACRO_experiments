# %%
"""
Distribution of every model feature in CULPRIT-SHOCK and eICU, as the models
see them (after unit harmonization, BNP -> NT-proBNP and troponin I -> T
proxies), to check that the two cohorts are comparable before validating.

Admission features: all patients of both cohorts. 24 h features: patients
alive after 24 h (cohort of the Full model). Continuous features are shown as
box plots (log scale for right-skewed labs), binary features as % positive.

Requires the preprocessed eICU cohort (eICU_extract_CS_patients.py).
Outputs (output/data_checks/, aggregate values only):
    culprit_vs_eicu_admission.png/.pdf, culprit_vs_eicu_24h.png/.pdf
    culprit_vs_eicu_features.md   n, % missing, median [IQR] or n (%), and the
        standardized mean difference (SMD) per feature
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from lib.cohort_comparison import compare_cohorts, is_binary, use_log_scale
from lib.data_load_utils import get_data_from_features, load_CULPRIT_data, load_eICU
from lib.data_processing import remove_low_variance_features
from lib.experiment_config import VARIANCE_THS, model_cohort
from lib.experiment_definitions import get_features
from lib.feature_names import feature_label
from lib.paths import CULPRIT_DIR, DATA_DIR, OUTPUT_DIR
from lib.utils import ensure_dir

# %%
save_dir = OUTPUT_DIR / "data_checks"
ensure_dir(save_dir)

COHORTS = ("CULPRIT-SHOCK", "eICU")
COLORS = {"CULPRIT-SHOCK": "#257de0", "eICU": "#f47230"}  # validated categorical pair
TEXT, MUTED, GRID = "#191919", "#666666", "#e6e6e6"
MIN_GROUP = 10  # groups with fewer values are not drawn


# %% Data
patient_info = load_CULPRIT_data(CULPRIT_DIR)
admission_features = get_features("Admission")
day1_features = [f for f in get_features("24hs") if f not in admission_features]

culprit = {
    "Admission": get_data_from_features(model_cohort(patient_info, "Admission"), admission_features),
    "Full": get_data_from_features(model_cohort(patient_info, "Full"), get_features("24hs")),
}
# eICU with the same proxies/alignment as in the validation (load_eICU)
eicu = {
    name: load_eICU(DATA_DIR, name, X.astype(float), outcome="in_hospital_death")[0]
    for name, X in culprit.items()
}
# Features that enter the models (variance filter on CULPRIT-SHOCK)
in_model = set(remove_low_variance_features(culprit["Admission"].astype(float), VARIANCE_THS).columns) | set(
    remove_low_variance_features(culprit["Full"].astype(float), VARIANCE_THS).columns
)
# Readable feature names (lib/feature_names.py, as for the SHAP plots)
labels = {f: feature_label(f) for f in get_features("24hs")}

# %% Table
tables = []
for name, features in [("Admission", admission_features), ("Full", day1_features)]:
    t = compare_cohorts(culprit[name], eicu[name], features, MIN_GROUP)
    t.insert(0, "cohort", "all patients" if name == "Admission" else "alive at 24 h")
    tables.append(t)
table = pd.concat(tables, ignore_index=True)
table.insert(2, "label", table["feature"].map(labels))
table.insert(3, "in model", table["feature"].isin(in_model).map({True: "yes", False: "no (low variance)"}))
table = table.rename(columns={
    "a_missing_%": "CULPRIT % missing", "a_summary": "CULPRIT", "b_missing_%": "eICU % missing", "b_summary": "eICU",
})


def markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [f"{v:.0f}" if isinstance(v, float) and str(c).endswith("missing") else
                 (f"{v:+.2f}" if c == "SMD" and pd.notna(v) else str(v)) for c, v in r.items()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


n_text = (f"CULPRIT-SHOCK: {len(culprit['Admission'])} patients ({len(culprit['Full'])} alive at 24 h); "
          f"eICU: {len(eicu['Admission'])} patients ({len(eicu['Full'])} alive at 24 h).")
(save_dir / "culprit_vs_eicu_features.md").write_text(
    "# Feature distributions: CULPRIT-SHOCK vs eICU\n\n" + n_text + "\n\n"
    "Binary: n positive (%); continuous: median [Q1, Q3]. SMD: standardized mean difference "
    "(log10 scale for skewed labs); |SMD| > 0.1 is a common threshold for imbalance, > 0.2 notable. "
    "eICU NT-proBNP is converted from BNP and troponin T includes the troponin-I proxy.\n\n"
    + markdown(table) + "\n"
)
print(n_text)
print(table.loc[:, ["feature", "in model", "CULPRIT % missing", "eICU % missing", "SMD"]].to_string())


# %% Figures
def draw_feature(ax, feature: str, a: pd.Series, b: pd.Series, smd: float):
    data = {"CULPRIT-SHOCK": a.dropna().astype(float), "eICU": b.dropna().astype(float)}
    binary = is_binary(pd.concat([a, b]))
    title = labels[feature] + ("" if feature in in_model else " (not in model)")
    ax.set_title(title, fontsize=8, color=TEXT, loc="left", pad=4)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, length=2)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    shown = [c for c in COHORTS if len(data[c]) >= MIN_GROUP]
    ticks = [f"{c}\nn={len(data[c])}" if len(data[c]) >= MIN_GROUP else f"{c}\nnot available"
             for c in COHORTS]
    if binary:
        values = [100 * data[c].mean() if c in shown else 0 for c in COHORTS]
        bars = ax.bar([0, 1], values, width=0.55, color=[COLORS[c] for c in COHORTS], edgecolor="white", linewidth=2)
        for bar, c, v in zip(bars, COHORTS, values):
            if c in shown:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 2, f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=7, color=TEXT)
        ax.set_ylim(0, 115)
        ax.set_ylabel("% positive", fontsize=7, color=MUTED)
    else:
        log = use_log_scale(a, b)
        positions = [i for i, c in enumerate(COHORTS) if c in shown]
        if positions:
            box = ax.boxplot(
                [data[COHORTS[i]] for i in positions], positions=positions, widths=0.5, patch_artist=True,
                showfliers=True, medianprops={"color": TEXT, "linewidth": 1.2},
                whiskerprops={"color": MUTED, "linewidth": 0.8}, capprops={"color": MUTED, "linewidth": 0.8},
                flierprops={"marker": "o", "markersize": 2, "markeredgewidth": 0, "alpha": 0.5},
            )
            for patch, flier, i in zip(box["boxes"], box["fliers"], positions):
                patch.set_facecolor(COLORS[COHORTS[i]])
                patch.set_alpha(0.85)
                patch.set_edgecolor("white")
                flier.set_markerfacecolor(COLORS[COHORTS[i]])
        if log:
            ax.set_yscale("log")
    ax.set_xticks([0, 1], ticks)
    ax.set_xlim(-0.6, 1.6)
    if np.isfinite(smd):
        large = abs(smd) > 0.2
        ax.set_xlabel(f"SMD {smd:+.2f}", fontsize=7, color=TEXT if large else MUTED,
                      fontweight="bold" if large else "normal", labelpad=3)


def figure(name: str, features: list[str], file_name: str, title: str):
    n_cols = 6
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols, 2.3 * n_rows), squeeze=False)
    smd = table[table["cohort"] == ("all patients" if name == "Admission" else "alive at 24 h")].set_index("feature")["SMD"]
    for ax, f in zip(axes.flat, features):
        draw_feature(ax, f, culprit[name][f], eicu[name][f] if f in eicu[name] else pd.Series(dtype=float), smd[f])
    for ax in axes.flat[len(features):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11, color=TEXT)
    fig.text(0.01, 0.995 - 0.35 / fig.get_figheight(), n_text + " Box plots: median, IQR, 1.5 IQR whiskers; "
             "log scale for skewed labs. SMD (below each panel) in bold when |SMD| > 0.2.",
             fontsize=8, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / fig.get_figheight()))
    for ext in ["png", "pdf"]:
        fig.savefig(save_dir / f"{file_name}.{ext}", dpi=200)
    plt.close(fig)


figure("Admission", admission_features, "culprit_vs_eicu_admission", "Admission features: CULPRIT-SHOCK vs eICU")
figure("Full", day1_features, "culprit_vs_eicu_24h", "24 h features (patients alive at 24 h): CULPRIT-SHOCK vs eICU")
print(f"Saved in {save_dir}")
# %%
