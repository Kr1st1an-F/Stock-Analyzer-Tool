# Stock Analyzer Tool  
![Python](https://img.shields.io/badge/python-3.9%2B-blue)  
![License](https://img.shields.io/badge/license-MIT-green)  
![Status](https://img.shields.io/badge/status-active-success)

> **A lightweight, educational stock analysis tool** that fetches real-time data, runs simple 1-year projections, and gives a **BUY / HOLD / SELL** rating using projections, technicals, and valuation.

---

## Features

- **No API keys needed** – uses `yfinance` (free Yahoo Finance data)
- **1-3-5-Year EPS-based price target** using analyst forward estimates
- **Technical indicators**: RSI(14), SMA-50, MACD
- **Rule-based rating** (40% Projections, 40% Technicals, 20% Valuation)
- **Interactive price + RSI chart** (Matplotlib)
- **Fully offline-capable** after first run (caches data if you add caching)

---

## How to Use

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/stock-analyzer-tool.git
cd stock-analyzer-tool
```
### 2. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
```
### or
```bash
.venv\Scripts\activate       # Windows
```
### 3. Install dependencies
```bash
pip install yfinance pandas numpy pandas_ta matplotlib
```
### 4. Enter a Ticker
```
Enter ticker (e.g., SOFI):
```
### You’ll get:
* Current price
* 1-year projected price & upside
* RSI, MACD, SMA signal
* BUY / HOLD / SELL recommendation
* Option to view price + RSI chart

## Example Output (Using $SOFI as ticker) - Nov 13, 2025)
```
SOFI Snapshot – 2025-11-13
Current Price : $32.21

1-Year Projection
 Period  Projected EPS  Projected Price  Upside %
   FY+1           0.40            44.39      37.8

Latest Technicals
   Close   RSI  MACD_12_26_9  Signal
 32.21  60.87          0.84       0

Technical Signal : Neutral

Recommendation : **BUY**
```
## Limitations

| Limitation             | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| **Simplified Model**   | Assumes EPS grows at revenue rate; no margin modeling                   |
| **Free Data**          | `yfinance` can lag or miss fields (e.g., forward EPS)                   |
| **Rule-Based Only**    | No machine learning or sentiment analysis                               |
| **No Risk Metrics**    | No beta, volatility, or drawdown analysis                               |
| **U.S. Stocks Only**   | `yfinance` works best with NYSE/NASDAQ tickers                          |

## DISCLAIMER!!!
> **THIS TOOL IS FOR EDUCATIONAL PURPOSES ONLY.
It is not financial advice, investment recommendation, or a trading signal.
The projections and ratings are based on simplified assumptions and public data that may be delayed, incomplete, or inaccurate. You are solely responsible for your investment decisions.
Always do your own research (DYOR) and consider consulting a licensed financial advisor.**
