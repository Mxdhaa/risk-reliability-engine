import pandas as pd
import numpy as np
import yfinance as yf

import time

def fetch_ohlcv(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(symbol, start=start, end=end, progress=False, threads=False)
            if not df.empty:
                # Flatten MultiIndex columns (e.g., ('close','^gspc') -> 'close')
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [str(c).lower() for c in df.columns]

                df = df[["open", "high", "low", "close", "volume"]].dropna()
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {symbol}: {e}")
        time.sleep(2)
    raise ValueError(f"Failed to fetch data for {symbol} after 3 attempts")


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    out["ret"] = close.pct_change()
    out["logret"] = np.log(close / close.shift(1))

    return out.dropna(subset=["ret", "logret"])
