import yfinance as yf
import pandas as pd
import json
import os
import requests
import datetime
import sys

# --- CONFIG ---
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TRADES_FILE = "trades.json"

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def track_trades():
    print("--- 💼 STARTING TRADE MANAGER ---")
    
    if not os.path.exists(TRADES_FILE):
        print("No trades file found yet.")
        return

    with open(TRADES_FILE, 'r') as f:
        try: trades = json.load(f)
        except: trades = []

    active_trades = [t for t in trades if t['status'] == 'OPEN']
    if not active_trades:
        print("No open positions to check.")
        return

    print(f"🔍 Checking {len(active_trades)} open positions...")
    updated = False

    for t in active_trades:
        symbol = t['symbol'] + ".NS"
        try:
            # 1. Parse the Entry Date
            entry_date_str = t['date']
            entry_date = datetime.datetime.strptime(entry_date_str, "%Y-%m-%d").date()

            # 2. Download Data
            df = yf.download(symbol, period="5d", progress=False)
            
            if df.empty: continue
            
            # Fix MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                temp = pd.DataFrame()
                temp['High'] = df.xs('High', level=0, axis=1)[symbol]
                temp['Low'] = df.xs('Low', level=0, axis=1)[symbol]
                temp['Close'] = df.xs('Close', level=0, axis=1)[symbol]
                df = temp

            # 3. CRITICAL FIX: Filter for Future Dates Only
            # We only look at rows where the Date is AFTER the Entry Date
            future_data = df[df.index.date > entry_date]

            if future_data.empty:
                # This means the trade was taken today, and there is no "tomorrow" data yet.
                # So we skip checking it.
                print(f"⏳ {t['symbol']}: Trade is new. Waiting for future data.")
                continue

            # 4. Check Win/Loss on FUTURE data only
            high_price = future_data['High'].max()
            low_price = future_data['Low'].min()

            # CHECK WIN
            if high_price >= t['target']:
                t['status'] = "WIN"
                msg = f"✅ **TRADE WON: {t['symbol']}**\n\n🎯 Target Hit: {t['target']}\n💰 Profit: +10% (Virtual)"
                send_telegram(msg)
                updated = True

            # CHECK LOSS
            elif low_price <= t['sl']:
                t['status'] = "LOSS"
                msg = f"❌ **TRADE LOST: {t['symbol']}**\n\n⛔ Stop Hit: {t['sl']}\n💸 Loss: -5% (Virtual)"
                send_telegram(msg)
                updated = True
                
        except Exception as e:
            print(f"Error checking {symbol}: {e}")

    if updated:
        with open(TRADES_FILE, 'w') as f:
            json.dump(trades, f, indent=4)
        print("💾 Ledger updated successfully.")

if __name__ == "__main__":
    track_trades()
