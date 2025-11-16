# Made by Kristian Fatohi

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
import requests
import os


FINNHUB_KEY = os.getenv("FINNHUB_KEY")


def fetch_data(ticker: str):
    """Fetch data from yfinance + Finnhub (fallback to yf). Returns enhanced info for multi-year."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info
    hist = stock.history(period="1y")  # daily OHLCV

    if hist.empty:
        raise ValueError(f"No historical data for {ticker}. Check the ticker symbol.")

    current_price = info.get("currentPrice") #Try to get live price.
    if current_price is None:
        current_price = hist["Close"].iloc[-1] #If not live price, use last close.

    forward_eps = info.get("forwardEps", np.nan)
    forward_pe = info.get("forwardPE", 20)  # default 20x
    rev_growth = info.get("revenueGrowth", 0.15)  # default 15%

    lt_growth = rev_growth
    if FINNHUB_KEY:
        try:
            url = f"https://finnhub.io/api/v1/stock/metric?symbol={ticker}&metric=all&token={FINNHUB_KEY}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json().get("metric", {})
                lt_growth = data.get("ltGrowth", rev_growth)
                forward_eps = data.get("forwardEps", forward_eps)
        except Exception as e:
            print(f"Finnhub fetch failed (using yf fallback): {e}")

    if lt_growth == rev_growth:
        lt_growth = info.get("earningsGrowth", rev_growth)

    return hist, current_price, forward_eps, forward_pe, lt_growth


def generate_projections(current_price, forward_eps, forward_pe, lt_growth, years=[1, 3, 5]):
    """Generate multi-year projections with compounding. Default: 1Y, 3Y, 5Y."""
    projections = []
    for n in years:
        projected_eps = forward_eps * (1 + lt_growth) ** n
        projected_price = projected_eps * forward_pe
        upside = (projected_price - current_price) / current_price * 100
        projections.append({
            "Period": f"{n}Y",
            "Projected EPS": projected_eps,
            "Projected Price": projected_price,
            "CAGR %": lt_growth * 100,
            "Upside %": upside
        })

    df = pd.DataFrame(projections)
    return df


def technical_signals(hist: pd.DataFrame):
    """Calculate technical indicators and generate buy/sell signals."""
    df = hist.copy()

    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    macd = ta.macd(df["Close"])
    df = df.join(macd)

    df["Signal"] = 0

    buy_cond = (df["RSI"] < 30) & (df["Close"] > df["SMA_50"])
    sell_cond = (df["RSI"] > 70) & (df["Close"] < df["SMA_50"])

    df.loc[buy_cond, "Signal"] = 1
    df.loc[sell_cond, "Signal"] = -1

    if len(df) >= 2:
        macd_line = df["MACD_12_26_9"].iloc[-1]
        signal_line = df["MACDs_12_26_9"].iloc[-1]
        if macd_line > signal_line:
            cur = df["Signal"].iloc[-1]
            df.loc[df.index[-1], "Signal"] = max(cur, 1)

    latest_signal = int(df["Signal"].iloc[-1])
    return latest_signal, df[["Close", "RSI", "MACD_12_26_9", "Signal"]].tail(1)


def get_rating(proj_df: pd.DataFrame, tech_signal: int, current_pe: float) -> str:
    """Combine projections (now using 3Y upside), technicals, and valuation into a final rating."""

    proj_upside = proj_df[proj_df["Period"] == "3Y"]["Upside %"].iloc[0] if "3Y" in proj_df["Period"].values else \
    proj_df["Upside %"].iloc[0]

    proj_score = 1 if proj_upside > 10 else (-1 if proj_upside < -10 else 0)
    tech_score = tech_signal
    val_score = -1 if current_pe > 25 else (1 if current_pe < 15 else 0)

    total = 0.4 * proj_score + 0.4 * tech_score + 0.2 * val_score
    if total > 0.5:
        return "BUY"
    elif total < -0.5:
        return "SELL"
    else:
        return "HOLD"


def plot_summary(hist: pd.DataFrame, ticker: str):
    """Plot price and RSI chart for the given historical data."""
    # Ensure RSI & SMA exist (reuse the same logic as technical_signals)
    df = hist.copy()
    df["RSI"] = ta.rsi(df["Close"], length=14)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # ---- price + SMA -------------------------------------------------
    ax1.plot(df.index, df["Close"], label="Close", color="steelblue")
    ax1.plot(df.index, df["SMA_50"], label="SMA-50", color="orange")
    ax1.set_title(f"{ticker} – Price & SMA-50")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ---- RSI ---------------------------------------------------------
    ax2.plot(df.index, df["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--", linewidth=1)
    ax2.axhline(30, color="green", linestyle="--", linewidth=1)
    ax2.set_title("RSI (14)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticker = input("Enter ticker (e.g., SOFI): ").strip().upper()

    hist, price, fwd_eps, fwd_pe, lt_growth = fetch_data(ticker)  # Now lt_growth

    # projections (multi-year)
    proj_df = generate_projections(price, fwd_eps, fwd_pe, lt_growth)
    proj_upside_3y = proj_df[proj_df["Period"] == "3Y"]["Upside %"].iloc[0] if "3Y" in proj_df["Period"].values else \
    proj_df["Upside %"].iloc[0]

    # technicals
    tech_sig, tech_df = technical_signals(hist)

    # current P/E (approx)
    cur_pe = price / fwd_eps if not np.isnan(fwd_eps) and fwd_eps != 0 else fwd_pe

    # rating
    rating = get_rating(proj_df, tech_sig, cur_pe)  # Now passes proj_df

    # output
    print(f"\n{ticker} Snapshot – {pd.Timestamp.now().date()}")
    print(f"Current Price : ${price:,.2f}")
    print(f"Long-Term Growth Rate: {lt_growth * 100:.1f}% (Source: Finnhub/yf)")
    print("\nMulti-Year Projections")
    print(proj_df.round(2).to_string(index=False))
    print("\nLatest Technicals")
    print(tech_df.round(2).to_string(index=False))
    print(f"Technical Signal : {'Bullish' if tech_sig == 1 else 'Bearish' if tech_sig == -1 else 'Neutral'}")
    print(f"\nRecommendation : **{rating}**")
    print(f"(Projections 40% [3Y Upside: {proj_upside_3y:.1f}%], Technicals 40%, Valuation 20 %)\n")

    # optional chart
    if input("Show price + RSI chart? (y/n): ").strip().lower() == "y":
        plot_summary(hist, ticker)
