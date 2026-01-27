# figure3.py  (PROJECT ROOT)
from src.figures import make_figure3

if __name__ == "__main__":
    out_path = "artifacts/figure3.png"
    make_figure3(
        preds_path="artifacts/preds.csv",
        out_path=out_path,
        vol_window=20,
        stress_quantile=0.90,
        use_abs_returns=True,
        show=True,   # <-- POPUP PLOT WINDOW
    )
