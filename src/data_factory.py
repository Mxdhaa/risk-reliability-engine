import pandas as pd
import numpy as np
import yfinance as yf

def fetch_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False)

    # Flatten MultiIndex columns (e.g., ('close','^gspc') -> 'close')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    out["ret"] = close.pct_change()
    out["logret"] = np.log(close / close.shift(1))

    return out.dropna(subset=["ret", "logret"])
