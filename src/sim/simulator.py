# src/sim/simulator.py

import pandas as pd
from datetime import timedelta

class PortfolioSimulator:
    def __init__(self, initial_budget: float = 1000.0, cooldown_hours: int = 3):
        self.initial_budget = initial_budget
        self.cooldown = timedelta(hours=cooldown_hours)
        self.reset()

    def reset(self):
        """Reset the overall trade log; daily state is reset each day."""
        self.trade_log = []

    def simulate(self, df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
        """
        Backtest day-by-day, hour-by-hour:
        - Buys subtract from budget & pnl
        - Intraday sells add to pnl
        - EOD sells add to budget & pnl
        - fund_value = day_start_budget + daily_pnl
        - Next day's start budget = fund_value at EOD
        - Append an 'eod_summary' row with daily_pnl and fund_value
        """
        df = df.sort_values("timestamp").copy()
        overall_budget = self.initial_budget  # M_i

        for date, day_df in df.groupby(df["timestamp"].dt.date):
            # Daily initialization
            day_start_budget = overall_budget
            budget    = day_start_budget
            daily_pnl = 0.0
            holdings  = {}
            sold_today= set()

            # Hour-by-hour trades
            for _, row in day_df.iterrows():
                ts, ticker, price, action = (
                    row["timestamp"],
                    row["ticker"],
                    row[price_col],
                    row.get("action"),
                )

                if action == "buy":
                    if ticker not in holdings and ticker not in sold_today and budget >= price:
                        budget    -= price
                        daily_pnl -= price
                        holdings[ticker] = {"buy_price": price, "buy_time": ts}
                        self._record(
                            timestamp=ts,
                            action="buy",
                            ticker=ticker,
                            price=price,
                            budget=budget,
                            daily_pnl=daily_pnl,
                            fund_value=day_start_budget + daily_pnl
                        )

                elif action == "sell":
                    info = holdings.get(ticker)
                    if info and (ts - info["buy_time"] >= self.cooldown):
                        daily_pnl += price
                        del holdings[ticker]
                        sold_today.add(ticker)
                        self._record(
                            timestamp=ts,
                            action="sell_intraday",
                            ticker=ticker,
                            price=price,
                            budget=budget,
                            daily_pnl=daily_pnl,
                            fund_value=day_start_budget + daily_pnl
                        )

            # End-of-day liquidation
            last_ts = day_df["timestamp"].iloc[-1]
            for ticker, info in list(holdings.items()):
                prices = day_df.loc[day_df["ticker"] == ticker, price_col]
                if not prices.empty:
                    last_price = prices.iloc[-1]
                    budget    += last_price
                    daily_pnl += last_price
                    self._record(
                        timestamp=last_ts,
                        action="sell_eod",
                        ticker=ticker,
                        price=last_price,
                        budget=budget,
                        daily_pnl=daily_pnl,
                        fund_value=day_start_budget + daily_pnl
                    )
                del holdings[ticker]

            # EOD summary row
            fund_value = day_start_budget + daily_pnl
            self.trade_log.append({
                "timestamp":  pd.Timestamp(date),
                "action":     "eod_summary",
                "ticker":     None,
                "price":      None,
                "budget":     None,
                "daily_pnl":  daily_pnl,
                "fund_value": fund_value
            })

            # Roll forward for next day
            overall_budget = fund_value

        return pd.DataFrame(self.trade_log)

    def _record(
        self,
        timestamp: pd.Timestamp,
        action: str,
        ticker: str,
        price: float,
        budget: float,
        daily_pnl: float,
        fund_value: float
    ):
        """Log each trade or EOD sell with running budget, pnl, and fund_value."""
        self.trade_log.append({
            "timestamp":  timestamp,
            "action":     action,
            "ticker":     ticker,
            "price":      price,
            "budget":     budget,
            "daily_pnl":  daily_pnl,
            "fund_value": fund_value
        })

    def get_summary(self) -> dict:
        """Return final fund value and total P&L."""
        if not self.trade_log:
            return {"final_fund_value": self.initial_budget, "total_pnl": 0.0}
        final_value = self.trade_log[-1]["fund_value"]
        total_pnl   = final_value - self.initial_budget
        return {"final_fund_value": final_value, "total_pnl": total_pnl}
