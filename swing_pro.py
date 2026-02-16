import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import datetime
import json
import os
import sys
import io
import joblib  # 🧠 AI Library
import sklearn # Required for the AI model to run

# ================= CONFIGURATION =================
# These look for the NEW keys in your new Repo's Secrets
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MAX_ALERTS_PER_DAY = 3
MIN_SCORE = 7 # 8.5 High threshold for quality

# File paths
HISTORY_FILE = "alert_history.json"
TRADES_FILE = "trades.json"
MODEL_FILE = "swing_ai_model.pkl"

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ Error: Telegram tokens not found. Check GitHub Secrets.")
    sys.exit(1)

# ================= DATA HELPERS =================
def load_json(filename):
    if not os.path.exists(filename): 
        return {} if "history" in filename else []
    try:
        with open(filename, 'r') as f: return json.load(f)
    except: 
        return {} if "history" in filename else []

def save_json(filename, data):
    with open(filename, 'w') as f: json.dump(data, f, indent=4)

def update_history(ticker):
    history = load_json(HISTORY_FILE)
    history[ticker] = datetime.date.today().strftime("%Y-%m-%d")
    save_json(HISTORY_FILE, history)

# ================= DUPLICATE & COOL-DOWN CHECKER =================
def is_duplicate_alert(ticker):
    clean_symbol = ticker.replace('.NS', '')
    history = load_json(HISTORY_FILE)
    
    # 1. Alert Cooldown (10 Days for this bot)
    if ticker in history:
        try:
            last = datetime.datetime.strptime(history[ticker], "%Y-%m-%d").date()
            if (datetime.date.today() - last).days < 10: return True
        except: pass

    # 2. Open Position Block
    trades = load_json(TRADES_FILE)
    for t in trades:
        if t.get('symbol') == clean_symbol and t.get('status') == 'OPEN': return True
            
    return False

# ================= TELEGRAM SENDER =================
def send_telegram_alert(message):
    chat_ids = [x.strip() for x in TELEGRAM_CHAT_ID.split(',')]
    for chat_id in chat_ids:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"Telegram Error: {e}")

# ================= 🧠 AI ANALYSIS ENGINE =================
def analyze_stock(ticker, sector, nifty_trend, ai_model):
    try:
        # 1. FETCH DATA (1 Year)
        df = yf.download(ticker, period="1y", progress=False)
        
        # Validation
        if df.empty or len(df) < 200: return None
        
        # Fix MultiIndex columns (yfinance generic fix)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. CALCULATE INDICATORS (Must match training data EXACTLY)
        # Technicals
        df["EMA20"] = ta.ema(df["Close"], 20)
        df["EMA200"] = ta.ema(df["Close"], 200)
        df["RSI"] = ta.rsi(df["Close"], 14)
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], 14)
        
        # ADX Check
        adx_df = ta.adx(df["High"], df["Low"], df["Close"], 14)
        if adx_df is not None and not adx_df.empty:
            df["ADX"] = adx_df['ADX_14']
        else:
            df["ADX"] = 0

        # Custom AI Features
        df["RVOL"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["Dist_EMA20"] = (df["Close"] - df["EMA20"]) / df["EMA20"]

        df = df.dropna()
        curr = df.iloc[-1]

        # 3. PRE-FILTER (Don't waste AI time on garbage)
        if curr["Close"] < curr["EMA200"]: return None # Trend is down
        if curr["ADX"] < 20: return None # No momentum

        # 4. 🤖 ASK THE AI
        ai_score = 50.0 # Default neutral
        ai_msg = "🤖 AI: OFF"
        
        if ai_model:
            # Prepare single row of features
            # Order MUST be: ['RSI', 'ADX', 'RVOL', 'Dist_EMA20', 'ATR']
            features = pd.DataFrame([[
                curr['RSI'], 
                curr['ADX'], 
                curr['RVOL'], 
                curr['Dist_EMA20'], 
                curr['ATR']
            ]], columns=['RSI', 'ADX', 'RVOL', 'Dist_EMA20', 'ATR'])
            
            # Get Probability (Returns [Prob_Loss, Prob_Win])
            probability = ai_model.predict_proba(features)[0][1]
            ai_score = round(probability * 100, 1)
            
            if ai_score > 65: ai_msg = f"🚀 AI: BULLISH ({ai_score}%)"
            elif ai_score < 40: ai_msg = f"🐻 AI: BEARISH ({ai_score}%)"
            else: ai_msg = f"🤷 AI: NEUTRAL ({ai_score}%)"

        # 5. SCORING SYSTEM
        base_score = 5.0
        reasons = [ai_msg]
        
        # AI Weighted Influence
        if ai_score > 75: base_score += 3.0
        elif ai_score > 60: base_score += 1.5
        elif ai_score < 40: return None # AI Reject (Hard Filter)

        # Technical Bonus
        if curr["RVOL"] > 2.0: 
            base_score += 1.0
            reasons.append(f"🐳 Huge Volume ({round(curr['RVOL'],1)}x)")
        
        # Setup Detection
        setup = "Setup"
        range_high = df.iloc[-11:-1]["High"].max()
        
        if curr["Close"] > range_high:
            setup = "Breakout"
            base_score += 1.5
        elif abs(curr["Close"] - curr["EMA20"]) / curr["Close"] < 0.03:
            setup = "Pullback"
            base_score += 1.0

        if base_score < MIN_SCORE: return None

        # 6. CALCULATE TRADE LEVELS
        sl = max(curr["Close"] - 2 * curr["ATR"], curr["Close"] * 0.92) # Max 8% risk
        target = curr["Close"] + (curr["Close"] - sl) * 2 # 1:2 Risk/Reward

        return {
            "symbol": ticker.replace(".NS", ""),
            "sector": sector,
            "setup": setup,
            "entry": round(curr["Close"], 1),
            "sl": round(sl, 1),
            "t1": round(target, 1),
            "score": round(base_score, 1),
            "reasons": reasons,
            "ai_conf": ai_score
        }

    except Exception as e:
        # print(f"Error on {ticker}: {e}") # Uncomment for debugging
        return None

# ================= MAIN RUNNER =================
def run_scan():
    print("--- 🧠 Starting Experimental AI Scan ---")
    
    # 1. Load the Brain
    ai_model = None
    if os.path.exists(MODEL_FILE):
        try:
            ai_model = joblib.load(MODEL_FILE)
            print("✅ AI Brain Loaded Successfully.")
        except Exception as e:
            print(f"⚠️ AI Load Failed: {e}")
    else:
        print("⚠️ Warning: 'swing_ai_model.pkl' not found. Running in Manual Mode.")

    # 2. Get Nifty List (Dynamic)
    try:
        url = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
        df = pd.read_csv(io.StringIO(requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).content.decode('utf-8')))
        STOCKS = {f"{row['Symbol']}.NS": row.get("Industry", "Unknown") for _, row in df.iterrows()}
    except:
        print("⚠️ NSE Download failed, using backup list.")
        STOCKS = {"RELIANCE.NS": "Energy", "HDFCBANK.NS": "Bank", "INFY.NS": "IT", "TCS.NS": "IT"}

    # 3. Check Market Trend
    nifty_trend = "NEUTRAL"
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)["Close"]
        if isinstance(nifty, pd.DataFrame): nifty = nifty.iloc[:, 0]
        nifty_trend = "BULLISH" if nifty.iloc[-1] > ta.ema(nifty, 50).iloc[-1] else "BEARISH"
    except: pass
    
    market_icon = "🟢" if nifty_trend == "BULLISH" else "🔴"
    
    # 4. Scan Loop
    signals = []
    print(f"Scanning {len(STOCKS)} stocks...")
    
    for ticker, sector in STOCKS.items():
        if not is_duplicate_alert(ticker):
            data = analyze_stock(ticker, sector, nifty_trend, ai_model)
            if data: signals.append(data)

    # 5. Sort & Filter
    signals.sort(key=lambda x: x["score"], reverse=True)

    # 6. Send Alerts
    if not signals:
        print("No setups found.")
        # Optional: Send "No trades" message to keep bot active
        # send_telegram_alert(f"📉 **AI Scan Complete**\nMarket: {market_icon} {nifty_trend}\nNo high-confidence setups.")
        return

    print(f"Found {len(signals)} signals. Sending top {MAX_ALERTS_PER_DAY}...")

    for s in signals[:MAX_ALERTS_PER_DAY]:
        risk = round(s["entry"] - s["sl"], 1)
        reasoning = "\n".join([f"• {r}" for r in s["reasons"]])
        
        msg = f"""
🧪 **EXPERIMENTAL AI ALERT**

📌 **Stock:** {s['symbol']}
🏢 **Sector:** {s['sector']}
🚦 **Market:** {market_icon} {nifty_trend}
📊 **Score:** {s['score']}/10
🤖 **AI Confidence:** {s['ai_conf']}%

🔍 **Why this trade?**
{reasoning}

📍 **Entry:** {s['entry']}
⛔ **Stop:** {s['sl']}
🎯 **Target:** {s['t1']}
⚠️ **Risk:** ₹{risk}

_This is a test signal from Bot B_
"""
        send_telegram_alert(msg)
        update_history(s["symbol"] + ".NS")
        
        # Log Trade
        trade = {"symbol": s['symbol'], "entry": s['entry'], "status": "OPEN", "date": str(datetime.date.today())}
        trades = load_json(TRADES_FILE)
        trades.append(trade)
        save_json(TRADES_FILE, trades)

if __name__ == "__main__":
    run_scan()
