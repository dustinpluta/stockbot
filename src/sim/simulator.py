# src/sim/simulator.py
import pandas as pd
from collections import defaultdict
from datetime import timedelta

class PortfolioSimulator:
    def __init__(self, initial_budget: float = 1000.0, cooldown_hours: int = 3):
        self.initial_budget = initial_budget
        self.cooldown = timedelta(hours=cooldown_hours)
        self.reset()

    def reset(self):
        self.budget = self.initial_budget
        self.holdings = {}  # {ticker: {'buy_price': ..., 'timestamp': ...}}
        self.trade_log = []
        self.current_time = None
        self.sold_today = set()

    def simulate(self, df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
        df = df.sort_values(by="timestamp").copy()
        for _, row in df.iterrows():
            self.current_time = pd.to_datetime(row["timestamp"])
            ticker = row["ticker"]
            price = row[price_col]
            action = row.get("action")

            if action == "buy":
                self._handle_buy(ticker, price)
            elif action == "sell":
                self._handle_sell(ticker, price)

        # Final liquidation
        for ticker, info in self.holdings.items():
            self._record_trade("final_sell", ticker, info["buy_price"], self.current_time)

        return pd.DataFrame(self.trade_log)

    def _handle_buy(self, ticker, price):
        if ticker in self.holdings or ticker in self.sold_today:
            return
        if self.budget < price:
            return

        self.holdings[ticker] = {"buy_price": price, "timestamp": self.current_time}
        self.budget -= price
        self._record_trade("buy", ticker, price, self.current_time)

    def _handle_sell(self, ticker, price):
        if ticker not in self.holdings:
            return
        buy_time = self.holdings[ticker]["timestamp"]
        if self.current_time - buy_time < self.cooldown:
            return

        self._record_trade("sell", ticker, price, self.current_time)
        del self.holdings[ticker]
        self.sold_today.add(ticker)

    def _record_trade(self, action, ticker, price, timestamp):
        self.trade_log.append({
            "timestamp": timestamp,
            "ticker": ticker,
            "action": action,
            "price": price,
            "budget": self.budget
        })

    def get_summary(self):
        trades = pd.DataFrame(self.trade_log).sort_values("timestamp")
        buy_prices = {}
        total_pnl = 0.0

        for _, row in trades.iterrows():
            action, ticker, price = row["action"], row["ticker"], row["price"]
            if action == "buy":
                buy_prices[ticker] = price
            elif action in ("sell", "final_sell"):
                bp = buy_prices.pop(ticker, None)
                if bp is not None:
                    total_pnl += price - bp

        return {
            "final_budget": self.budget,
            "pnl": total_pnl
        }
