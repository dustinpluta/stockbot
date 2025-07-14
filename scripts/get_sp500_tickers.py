# scripts/get_sp500_tickers.py

import sys
import pandas as pd

def main(output_file="sp500_tickers.txt"):
    # Read table from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, header=0)
    df = tables[0]  # First table contains the list

    tickers = df["Symbol"].sort_values().tolist()

    with open(output_file, "w") as f:
        for ticker in tickers:
            f.write(ticker + "\n")

    print(f"✅ Saved {len(tickers)} tickers to {output_file}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) == 2 else "sp500_tickers.txt"
    main(out)
