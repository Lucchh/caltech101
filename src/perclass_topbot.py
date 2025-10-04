import os, argparse, pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_perclass(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)  # columns: class, accuracy
    # coerce missing/blank to NaN, cast to float
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    # drop classes with no samples on the test split (rare but possible)
    df = df.dropna(subset=["accuracy"]).reset_index(drop=True)
    return df

def top_bottom(df: pd.DataFrame, k: int = 5):
    df_sorted = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    topk = df_sorted.head(k).copy()
    botk = df_sorted.tail(k).copy()
    return topk, botk

def make_bar(df: pd.DataFrame, out_png: str, title: str = ""):
    plt.figure(figsize=(11, 3.6))
    srt = df.sort_values("accuracy", ascending=False).reset_index(drop=True)
    plt.bar(np.arange(len(srt)), srt["accuracy"].values)
    plt.xticks([])  # 102 labels won’t fit; keep axis clean for the appendix
    plt.ylabel("Per-class accuracy")
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def as_markdown(topk: pd.DataFrame, botk: pd.DataFrame, run_name: str) -> str:
    md = []
    md.append(f"### Per-class extremes — {run_name}")
    md.append("")
    md.append("| Top-5 class | Acc |  | Bottom-5 class | Acc |")
    md.append("|---|---:|---|---|---:|")
    for i in range(max(len(topk), len(botk))):
        row = []
        if i < len(topk):
            row += [topk.loc[i, "class"], f"{topk.loc[i, 'accuracy']:.3f}"]
        else:
            row += ["", ""]
        row.append("")  # spacer column
        if i < len(botk):
            row += [botk.loc[botk.index[i], "class"], f"{botk.loc[botk.index[i], 'accuracy']:.3f}"]
        else:
            row += ["", ""]
        md.append("| " + " | ".join(row) + " |")
    return "\n".join(md)

def as_latex(topk: pd.DataFrame, botk: pd.DataFrame, run_name: str, label: str) -> str:
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("  \\centering")
    lines.append("  \\begin{tabular}{l r c l r}")
    lines.append("    \\toprule")
    lines.append("    \\textbf{Top-5 class} & \\textbf{Acc} & & \\textbf{Bottom-5 class} & \\textbf{Acc}\\\\")
    lines.append("    \\midrule")
    for i in range(max(len(topk), len(botk))):
        left = (f"{topk.loc[i, 'class']} & {topk.loc[i, 'accuracy']:.3f}") if i < len(topk) else " & "
        right = (f"{botk.loc[botk.index[i], 'class']} & {botk.loc[botk.index[i], 'accuracy']:.3f}") if i < len(botk) else " & "
        lines.append(f"    {left} & & {right}\\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(f"  \\caption{{Top/Bottom-5 per-class accuracy for {run_name}.}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser(description="Make Top-5 / Bottom-5 per-class accuracy tables from a run folder")
    ap.add_argument("--run", required=True, help="results/<run_name> folder containing per_class_accuracy.csv")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bar", action="store_true", help="also write a sorted bar plot (for appendix)")
    ap.add_argument("--outdir", default="report", help="where to save markdown/tex outputs (and optional plot)")
    args = ap.parse_args()

    run_name = os.path.basename(os.path.normpath(args.run))
    csv_path = os.path.join(args.run, "per_class_accuracy.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = load_perclass(csv_path)
    topk, botk = top_bottom(df, k=args.k)

    os.makedirs(args.outdir, exist_ok=True)

    # Markdown
    md = as_markdown(topk, botk, run_name)
    md_path = os.path.join(args.outdir, f"perclass_{run_name}_top{args.k}_bot{args.k}.md")
    with open(md_path, "w") as f:
        f.write(md)

    # LaTeX
    tex = as_latex(topk, botk, run_name, label=f"tab:perclass_{run_name}")
    tex_path = os.path.join(args.outdir, f"perclass_{run_name}_top{args.k}_bot{args.k}.tex")
    with open(tex_path, "w") as f:
        f.write(tex)

    print(f"\nMarkdown saved to: {md_path}")
    print(f"LaTeX saved to   : {tex_path}")

    # Optional appendix plot
    if args.bar:
        png_path = os.path.join(args.run, "per_class_accuracy_bar.png")
        make_bar(df, png_path, title=f"{run_name}: per-class accuracy (sorted)")
        print(f"Bar plot saved to: {png_path}")

    # Also echo to console so you can copy quickly
    print("\n----- Markdown preview -----")
    print(md)

if __name__ == "__main__":
    main()