import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def make_labels_next_day(returns: pd.Series, reported_risk: pd.Series, gamma: float) -> pd.Series:
    L = -returns.astype(float)
    y_t = (L > gamma * reported_risk).astype(int)
    return y_t.shift(-1)

def chrono_split(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    i1 = int(n * train_ratio)
    i2 = int(n * (train_ratio + val_ratio))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]

def train_reliability_xgb(X: pd.DataFrame, y: pd.Series, cfg):
    data = X.copy()
    data["y"] = y
    data = data.dropna()

    y = data.pop("y").astype(int)
    X = data

    pack = pd.concat([X, y.rename("y")], axis=1)
    train_df, val_df, test_df = chrono_split(pack, cfg.train_ratio, cfg.val_ratio)

    X_tr, y_tr = train_df.drop(columns=["y"]), train_df["y"].astype(int)
    X_va, y_va = val_df.drop(columns=["y"]), val_df["y"].astype(int)
    X_te, y_te = test_df.drop(columns=["y"]), test_df["y"].astype(int)

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    scale_pos_weight = (neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample_bytree,
        reg_lambda=cfg.xgb_reg_lambda,
        min_child_weight=cfg.xgb_min_child_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_va = model.predict_proba(X_va)[:, 1]
    p_te = model.predict_proba(X_te)[:, 1]

    metrics = {
        "auc_val": float(roc_auc_score(y_va, p_va)) if y_va.nunique() > 1 else float("nan"),
        "auc_test": float(roc_auc_score(y_te, p_te)) if y_te.nunique() > 1 else float("nan"),
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_va)),
        "n_test": int(len(X_te)),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_test": float(y_te.mean()),
    }
    return model, metrics
