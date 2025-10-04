# aggregate_results.py
import os, json, glob, re
import pandas as pd

# -------- Paths (relative to this file) --------
ROOT        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
REPORT_DIR  = os.path.join(ROOT, "report")
REPORT_MD   = os.path.join(ROOT, "REPORT.md")
os.makedirs(REPORT_DIR, exist_ok=True)

# -------- Loading --------
def load_one(run_dir):
    """Load metrics_test.json from a run folder."""
    mt = os.path.join(run_dir, "metrics_test.json")
    if not os.path.exists(mt):
        return None
    with open(mt, "r") as f:
        m = json.load(f)
    return {
        "run": os.path.basename(run_dir),
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("classification_report", {}).get("macro avg", {}).get("f1-score"),
        "weighted_f1": m.get("classification_report", {}).get("weighted avg", {}).get("f1-score"),
        "top5": m.get("top5_accuracy"),
    }

# -------- Renderers --------
def md_table(df, title=None):
    lines = []
    if title:
        lines.append(f"**{title}**")
    lines += [
        "| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        fmt = lambda x: "-" if pd.isna(x) else f"{x:.3f}"
        lines.append(
            f"| {r['run']} | {fmt(r['accuracy'])} | {fmt(r['macro_f1'])} | "
            f"{fmt(r['weighted_f1'])} | {fmt(r['top5'])} |"
        )
    return "\n".join(lines)

def latex_table(df, caption, label):
    fmt = lambda x: "-" if pd.isna(x) else f"{x:.3f}"
    rows = "\n".join(
        [f"    {r['run']} & {fmt(r['accuracy'])} & {fmt(r['macro_f1'])} & "
         f"{fmt(r['weighted_f1'])} & {fmt(r['top5'])} \\\\"
         for _, r in df.iterrows()]
    )
    # RAW f-string: keep backslashes literal; double braces for LaTeX braces
    return fr"""
\begin{{table}}[h]
  \centering
  \begin{{tabular}}{{lcccc}}
    \toprule
    Run & Acc & Macro-F1 & Weighted-F1 & Top-5 \\
    \midrule
{rows}
    \bottomrule
  \end{{tabular}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{table}}
""".strip()

def write_pair(df, stem, caption, label, title=None):
    """Write Markdown + LaTeX to report/<stem>.md/.tex"""
    df = df.reset_index(drop=True)
    md = md_table(df, title=title)
    tex = latex_table(df, caption, label)
    md_path  = os.path.join(REPORT_DIR, f"{stem}.md")
    tex_path = os.path.join(REPORT_DIR, f"{stem}.tex")
    with open(md_path, "w") as f:  f.write(md + "\n")
    with open(tex_path, "w") as f: f.write(tex + "\n")
    return md_path, tex_path

# -------- Helpers --------
def pick(df, regexes):
    """Subset rows whose run name matches ANY regex."""
    patt = re.compile("|".join(regexes))
    return df[df["run"].str.contains(patt, regex=True)]

# -------- Main --------
def main():
    # Load all runs
    runs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    rows = []
    for d in runs:
        m = load_one(d)
        if m:
            rows.append(m)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics found in results/*/metrics_test.json")
        return
    df = df.sort_values(["accuracy", "macro_f1"], ascending=[False, False])

    # Overall summary
    g_md, g_tex = write_pair(
        df, "summary_table",
        caption="Overall comparison across all runs in \\texttt{results/}.",
        label="tab:summary_all",
        title="Overall results"
    )
    print("Wrote:", g_md, "\n      ", g_tex)

    # Ablation: ResNet18 input size
    size_df = pick(df, [r"^resnet18_64px$", r"^resnet18_128px$", r"^resnet18_224"])
    if not size_df.empty:
        p_md, p_tex = write_pair(
            size_df.sort_values("run"),
            "ablation_size_resnet18",
            caption="ResNet-18: effect of input resolution.",
            label="tab:ablation_size",
            title="Ablation: ResNet-18 input size"
        )
        print("Wrote:", p_md, "\n      ", p_tex)

    # Ablation: ResNet18 augmentation (aug vs noaug, any batch size)
    aug_df = pick(df, [r"^resnet18_aug", r"^resnet18_noaug"])
    if not aug_df.empty:
        p_md, p_tex = write_pair(
            aug_df.sort_values("run"),
            "ablation_aug_resnet18",
            caption="ResNet-18: augmentation on vs. off (various batch sizes).",
            label="tab:ablation_aug",
            title="Ablation: ResNet-18 augmentation"
        )
        print("Wrote:", p_md, "\n      ", p_tex)

    # Ablation: ViT freezing vs full fine-tuning
    vit_df = pick(df, [r"^vitb16_224_freeze$", r"^vitb16_224_fullft$"])
    if not vit_df.empty:
        p_md, p_tex = write_pair(
            vit_df.sort_values("run"),
            "ablation_vit_freeze_full",
            caption="ViT-B/16: frozen backbone vs. full fine-tuning.",
            label="tab:ablation_vit_freeze",
            title="Ablation: ViT freezing vs full FT"
        )
        print("Wrote:", p_md, "\n      ", p_tex)

    # Append quick block into REPORT.md for convenience
    try:
        with open(g_md, "r") as f: summary_md = f.read()
        block = "\n## Results (auto-generated)\n\n" + summary_md + "\n"
        for stem in ["ablation_size_resnet18", "ablation_aug_resnet18", "ablation_vit_freeze_full"]:
            path = os.path.join(REPORT_DIR, f"{stem}.md")
            if os.path.exists(path):
                with open(path, "r") as f:
                    block += "\n" + f"### {stem.replace('_',' ').title()}\n\n" + f.read() + "\n"
        if os.path.exists(REPORT_MD):
            with open(REPORT_MD, "a") as f: f.write(block)
        else:
            with open(REPORT_MD, "w") as f: f.write("# Report\n" + block)
        print("Updated:", REPORT_MD)
    except Exception as e:
        print("Could not update REPORT.md:", e)

if __name__ == "__main__":
    main()