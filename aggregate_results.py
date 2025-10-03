import os, json, glob
import pandas as pd

# Always resolve relative to THIS file's folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OUT_DIR = os.path.join(SCRIPT_DIR, "report")
OUT_MD = os.path.join(OUT_DIR, "summary_table.md")
REPORT_MD = os.path.join(SCRIPT_DIR, "REPORT.md")

def load_metrics(run_dir):
    path = os.path.join(run_dir, "metrics_test.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        m = json.load(f)
    return {
        "run": os.path.basename(run_dir),
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("classification_report", {}).get("macro avg", {}).get("f1-score"),
        "weighted_f1": m.get("classification_report", {}).get("weighted avg", {}).get("f1-score"),
        "top5": m.get("top5_accuracy"),
    }

def main():
    print("Reading from:", RESULTS_DIR)
    runs = [d for d in glob.glob(os.path.join(RESULTS_DIR, "*")) if os.path.isdir(d)]
    print("Found run dirs:", [os.path.basename(d) for d in runs])

    rows = []
    for d in runs:
        metrics = load_metrics(d)
        if metrics: rows.append(metrics)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics found. Expected files like results/<run>/metrics_test.json")
        return

    # Build markdown
    lines = [
        "| Run | Acc | Macro-F1 | Weighted-F1 | Top-5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        acc  = f"{r['accuracy']:.3f}"     if pd.notna(r['accuracy'])     else "-"
        mf1  = f"{r['macro_f1']:.3f}"     if pd.notna(r['macro_f1'])     else "-"
        wf1  = f"{r['weighted_f1']:.3f}"  if pd.notna(r['weighted_f1'])  else "-"
        top5 = f"{r['top5']:.3f}"         if pd.notna(r['top5'])         else "-"
        lines.append(f"| {r['run']} | {acc} | {mf1} | {wf1} | {top5} |")

    table_md = "\n".join(lines)

    # Print + save
    print("\n" + table_md + "\n")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_MD, "w") as f: f.write(table_md)
    print("Saved table to:", OUT_MD)

    # Also append into REPORT.md for convenience
    block = "\n## Results Summary\n\n" + table_md + "\n"
    try:
        if os.path.exists(REPORT_MD):
            with open(REPORT_MD, "a") as f: f.write(block)
        else:
            with open(REPORT_MD, "w") as f: f.write("# Report\n" + block)
        print("Updated:", REPORT_MD)
    except Exception as e:
        print("Could not update REPORT.md:", e)

if __name__ == "__main__":
    main()