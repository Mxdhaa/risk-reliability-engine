from pathlib import Path
import json

def save_json(path: str, obj: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_text(path: str, text: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def make_results_latex(rows: list[dict], caption: str, label: str) -> str:
    cols = list(rows[0].keys())

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{" + "l" * len(cols) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(cols) + " \\\\")
    lines.append("\\midrule")

    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("NA" if v != v else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)
