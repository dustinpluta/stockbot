# src/core/feature_docs.py

FEATURE_DOCS = {
    "rsi": {
        "description": "Relative Strength Index over a 14-period window (default). Measures recent gains vs. losses.",
        "formula": r"100 - \left( \frac{100}{1 + RS} \right),\quad RS = \frac{\text{avg gain}}{\text{avg loss}}",
        "notes": [
            "Used to identify overbought or oversold conditions.",
            "Requires rolling average; produces NaNs at the start."
        ]
    },
    "macd": {
        "description": "Moving Average Convergence Divergence: difference between 12- and 26-period EMAs of closing price.",
        "formula": r"MACD_t = EMA_{12}(P_t) - EMA_{26}(P_t)",
        "notes": [
            "Highlights trend strength and momentum shifts."
        ]
    },
    "sma20": {
        "description": "20-period Simple Moving Average of closing price.",
        "formula": r"SMA_t = \frac{1}{20} \sum_{i=t-19}^t P_i",
        "notes": [
            "Smooths out short-term fluctuations in price.",
            "Needs 20 data points before producing values."
        ]
    },
    "return_1h": {
        "description": "One-period (1-hour) percentage return.",
        "formula": r"(P_t - P_{t-1}) / P_{t-1}",
        "notes": [
            "Can be noisy; used in momentum or volatility calculations."
        ]
    },
    "return_4h": {
        "description": "Four-period (4-hour) percentage return.",
        "formula": r"(P_t - P_{t-4}) / P_{t-4}",
        "notes": []
    },
    "log_return_1h": {
        "description": "Logarithmic return over one hour.",
        "formula": r"\log(P_t / P_{t-1})",
        "notes": [
            "May produce NaNs or errors for non-positive prices."
        ]
    },
    "volatility_5h": {
        "description": "Rolling standard deviation of 1-hour returns over 5 hours.",
        "formula": r"\text{std}(r_t, r_{t-1}, ..., r_{t-4})",
        "notes": [
            "Measures short-term variability in returns."
        ]
    },
    "momentum": {
        "description": "Price momentum over a 5-hour window (Close_t - Close_{t-5}).",
        "formula": r"P_t - P_{t-5}",
        "notes": []
    },
    "bollinger_upper": {
        "description": "Upper Bollinger Band: SMA plus 2 standard deviations.",
        "formula": r"SMA_t + 2 \cdot \text{std}_t",
        "notes": [
            "Used to signal overbought conditions or upper volatility bounds."
        ]
    },
    "bollinger_lower": {
        "description": "Lower Bollinger Band: SMA minus 2 standard deviations.",
        "formula": r"SMA_t - 2 \cdot \text{std}_t",
        "notes": [
            "Used to signal oversold conditions or lower volatility bounds."
        ]
    }
}
