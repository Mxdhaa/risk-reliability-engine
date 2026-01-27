import json, os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

fit_path = os.path.join("artifacts", "fit_metrics.json")
preds_path = os.path.join("artifacts", "preds.csv")

if not os.path.exists(fit_path) or not os.path.exists(preds_path):
    print("Missing files:")
    print("fit_metrics.json exists:", os.path.exists(fit_path))
    print("preds.csv exists:", os.path.exists(preds_path))
    raise SystemExit(1)

with open(fit_path, "r", encoding="utf-8") as f:
    fit = json.load(f)

preds = pd.read_csv(preds_path)

y_candidates = ["y_true", "y", "label", "target"]
p_candidates = ["p_hat", "p", "proba", "prob", "pred", "pred_proba", "score"]

y_col = next((c for c in y_candidates if c in preds.columns), None)
p_col = next((c for c in p_candidates if c in preds.columns), None)

if y_col is None or p_col is None:
    print("ERROR: required columns not found")
    print("Columns present:", list(preds.columns))
    raise SystemExit(1)

y = preds[y_col].astype(int)
p = preds[p_col].astype(float)

out = {
    "from_fit_metrics_json": fit,
    "recomputed_from_preds_csv": {
        "y_col": y_col,
        "p_col": p_col,
        "n": int(len(preds)),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "false_safe_rate@thr_0.5": float(((p < 0.5) & (y == 1)).mean())
    }
}

print(json.dumps(out, indent=2))
