# stock_analyzer_fixed.py
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

def fetch_data(ticker: str):
    """Fetch data from yfinance and return relevant info."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info
    hist = stock.history(period="1y")          # daily OHLCV

    if hist.empty:
        raise ValueError(f"No historical data for {ticker}. Check the ticker or internet connection.")

    current_price = info.get("currentPrice")
    if current_price is None:
        current_price = hist["Close"].iloc[-1]

    forward_eps = info.get("forwardEps", np.nan)
    forward_pe  = info.get("forwardPE", 20)      # default 20×
    rev_growth  = info.get("revenueGrowth", 0.15)  # 15 % default

    return hist, current_price, forward_eps, forward_pe, rev_growth



def generate_projections(current_price, forward_eps, forward_pe, rev_growth, periods=1):
    """Generate simple 1yr projections based on EPS growth and P/E ratio."""
    projected_eps = forward_eps * (1 + rev_growth)
    projected_price = projected_eps * forward_pe
    upside = (projected_price - current_price) / current_price * 100

    df = pd.DataFrame({
        "Period": [f"FY+{i+1}" for i in range(periods)],
        "Projected EPS": [projected_eps],
        "Projected Price": [projected_price],
        "Upside %": [upside]
    })
    return df


def technical_signals(hist: pd.DataFrame):
    """Calculate technical indicators and generate buy/sell signals."""
    df = hist.copy()

    df["RSI"]    = ta.rsi(df["Close"], length=14)
    df["SMA_50"] = ta.sma(df["Close"], length=50)

    macd = ta.macd(df["Close"])
    df = df.join(macd)

    df["Signal"] = 0                       

    buy_cond  = (df["RSI"] < 30) & (df["Close"] > df["SMA_50"])
    sell_cond = (df["RSI"] > 70) & (df["Close"] < df["SMA_50"])

    df.loc[buy_cond,  "Signal"] = 1
    df.loc[sell_cond, "Signal"] = -1

    # MACD bullish boost (only on the latest row)
    if len(df) >= 2:
        macd_line   = df["MACD_12_26_9"].iloc[-1]
        signal_line = df["MACDs_12_26_9"].iloc[-1]
        if macd_line > signal_line:
            cur = df["Signal"].iloc[-1]
            df.loc[df.index[-1], "Signal"] = max(cur, 1)

    latest_signal = int(df["Signal"].iloc[-1])
    return latest_signal, df[["Close", "RSI", "MACD_12_26_9", "Signal"]].tail(1)


def get_rating(proj_upside: float, tech_signal: int, current_pe: float) -> str:
    """Combine projections, technicals, and valuation into a final rating."""
    proj_score = 1 if proj_upside > 10 else (-1 if proj_upside < -10 else 0)
    tech_score = tech_signal
    val_score  = -1 if current_pe > 25 else (1 if current_pe < 15 else 0)

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
    df["RSI"]    = ta.rsi(df["Close"], length=14)
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
    ax2.axhline(70, color="red",   linestyle="--", linewidth=1)
    ax2.axhline(30, color="green", linestyle="--", linewidth=1)
    ax2.set_title("RSI (14)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticker = input("Enter ticker (e.g., SOFI): ").strip().upper()

    hist, price, fwd_eps, fwd_pe, rev_g = fetch_data(ticker)

    # projections
    proj_df = generate_projections(price, fwd_eps, fwd_pe, rev_g)
    upside = proj_df["Upside %"].iloc[0]

    # technicals
    tech_sig, tech_df = technical_signals(hist)

    # current P/E (approx)
    cur_pe = price / fwd_eps if not np.isnan(fwd_eps) and fwd_eps != 0 else fwd_pe

    # rating
    rating = get_rating(upside, tech_sig, cur_pe)

    # output 
    print(f"\n{ticker} Snapshot – {pd.Timestamp.now().date()}")
    print(f"Current Price : ${price:,.2f}")
    print("\n1-Year Projection")
    print(proj_df.round(2).to_string(index=False))
    print("\nLatest Technicals")
    print(tech_df.round(2).to_string(index=False))
    print(f"Technical Signal : {'Bullish' if tech_sig == 1 else 'Bearish' if tech_sig == -1 else 'Neutral'}")
    print(f"\nRecommendation : **{rating}**")
    print("(Projections 40 %, Technicals 40 %, Valuation 20 %)\n")

    # optional chart
    if input("Show price + RSI chart? (y/n): ").strip().lower() == "y":
        plot_summary(hist, ticker)
