from data_factory import fetch_ohlcv, compute_returns

df = fetch_ohlcv("^GSPC", "2020-01-01", "2020-06-01")
print("Columns before returns:", list(df.columns))

df = compute_returns(df)
print(df.head())
print("Rows:", len(df))
