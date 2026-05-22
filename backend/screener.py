"""
S&P 500 Screener for the 8/21 EMA Swing Trading Strategy (Enhanced with ML features).

This version captures all technical indicators and market context for ML training.
"""

from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import json
from datetime import datetime, timedelta
import logging
import os
import concurrent.futures
from tqdm import tqdm
import time
import warnings
import functools
import threading

warnings.filterwarnings('ignore')

_DATA_DIR = Path(__file__).parent / 'data'

# Setup Logging
os.makedirs(Path(__file__).parent / 'logs', exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = str(Path(__file__).parent / 'logs' / f'sp500_8_21_screener_enhanced_ml_{timestamp}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

HISTORY_FILE = str(_DATA_DIR / 'screener_history.csv')

def load_history():
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        # Add new columns if they don't exist (for backward compatibility)
        new_columns = {
            'Exit Price': None,
            'PnL %': None,
            'PnL $': None,
            'Hold Days': None,
            'Hit Target': False,
            'Hit Stop': False,
            # Technical indicators at entry
            'RSI_Entry': None,
            'ADX_Entry': None,
            'DeMarker_Entry': None,
            'DeMarker_Min_14d': None,
            'RS_vs_SPY': None,
            'Volume_Ratio': None,
            'Price_to_8EMA_%': None,
            'Price_to_21EMA_%': None,
            'EMA_Stack_Gap_%': None,
            'Pullback_Depth_%': None,
            'Pullback_Days': None,
            # Market context
            'SPY_RSI': None,
            'SPY_Trend': None,
            'VIX_Level': None,
            'Sector': None
        }
        for col, default in new_columns.items():
            if col not in df.columns:
                df[col] = default
        return df
    return pd.DataFrame(columns=['Ticker', 'Entry Date', 'Entry Price', 'Current Price', 'Stop Loss', 'Fib Target', 'Risk/Reward', 'Status', 'Exit Date', 'Pullback Date', 'Market Cap', 'Reason',
                                 'Exit Price', 'PnL %', 'PnL $', 'Hold Days', 'Hit Target', 'Hit Stop',
                                 'RSI_Entry', 'ADX_Entry', 'DeMarker_Entry', 'DeMarker_Min_14d', 'RS_vs_SPY', 'Volume_Ratio',
                                 'Price_to_8EMA_%', 'Price_to_21EMA_%', 'EMA_Stack_Gap_%', 'Pullback_Depth_%', 'Pullback_Days',
                                 'SPY_RSI', 'SPY_Trend', 'VIX_Level', 'Sector'])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

_TICKER_CACHE_FILE = _DATA_DIR / 'sp1500_tickers.json'

def _fetch_wikipedia_tickers(headers):
    """Fetch tickers from Wikipedia using io.StringIO to avoid lxml path confusion."""
    categories = {}
    sources = [
        ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'Symbol', 'Large'),
        ('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', 'Symbol', 'Mid'),
        ('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', 'Symbol', 'Small'),
    ]
    for url, col, cap in sources:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        df = pd.read_html(io.StringIO(resp.text))[0]
        for t in df[col].tolist():
            categories[t.replace('.', '-')] = cap
    return categories

def get_market_tickers():
    """Fetch S&P 1500 tickers (500 + 400 + 600) from Wikipedia, with JSON cache fallback."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        logger.info("Fetching S&P 500, 400, and 600 tickers from Wikipedia...")
        categories = _fetch_wikipedia_tickers(headers)
        tickers = sorted(list(categories.keys()))
        logger.info(f"Successfully loaded {len(tickers)} tickers from Wikipedia.")
        # Update cache for future fallback
        cache_data = [{'ticker': t, 'cap': categories[t]} for t in tickers]
        try:
            with open(_TICKER_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
        except Exception:
            pass
        return tickers, categories
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed: {e}. Trying cached ticker file...")
        if _TICKER_CACHE_FILE.exists():
            try:
                with open(_TICKER_CACHE_FILE) as f:
                    data = json.load(f)
                categories = {d['ticker']: d['cap'] for d in data}
                tickers = sorted(list(categories.keys()))
                logger.info(f"Loaded {len(tickers)} tickers from cache.")
                return tickers, categories
            except Exception as ce:
                logger.error(f"Cache load failed: {ce}")
        logger.error("Falling back to hardcoded ticker list.")
        fallback = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'V', 'UNH']
        return fallback, {t: 'Large' for t in fallback}

def calculate_indicators(df):
    """Calculate EMAs, SMAs, RSI, DeMarker, ADX"""
    if df.empty: return df
    df = df.copy()

    # EMAs
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # SMAs
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # DeMarker (13 period)
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()
    df['DeMax'] = np.where(high_diff > 0, high_diff, 0)
    df['DeMin'] = np.where(low_diff < 0, -low_diff, 0)
    df['DeMax_MA'] = df['DeMax'].rolling(window=13).mean()
    df['DeMin_MA'] = df['DeMin'].rolling(window=13).mean()
    denom = df['DeMax_MA'] + df['DeMin_MA']
    df['DeMarker'] = np.where(denom != 0, df['DeMax_MA'] / denom, 0.5)

    # ADX 14
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up_move = high - high.shift()
    down_move = low.shift() - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()

    return df

def get_stock_sector(ticker):
    """Get sector information for a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('sector', 'Unknown')
    except:
        return 'Unknown'

def is_market_closed():
    """Check if US market is closed (after 4 PM ET)"""
    from datetime import datetime
    import pytz

    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)

    # Check if it's a weekday
    if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return True

    # Check if after 4 PM ET
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return now_et >= market_close

def analyze_stock_enhanced(ticker, spy_df=None, vix_level=None, market_context=None):
    """
    Enhanced analysis that returns all ML features even when strategy conditions fail.
    Returns either a detailed dict with all metrics or a reason string on data error.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if len(hist) < 200:
            return "Insufficient Data"

        # Only exclude today's candle if market is still open
        if not is_market_closed():
            today = datetime.now().date()
            last_candle_date = hist.index[-1].date()

            if last_candle_date == today:
                hist = hist.iloc[:-1]
                if len(hist) < 200:
                    return "Insufficient Data (excluding today)"

        df = calculate_indicators(hist)

        # Get latest values
        current_price = df['Close'].iloc[-1]

        ema_8 = df['EMA_8'].iloc[-1]
        ema_21 = df['EMA_21'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        sma_200 = df['SMA_200'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        demarker = df['DeMarker'].iloc[-1]

        # Calculate all ML features regardless of pass/fail
        ml_features = {}

        # Core indicator values
        ml_features['RSI_Entry'] = round(rsi, 2) if not pd.isna(rsi) else None
        ml_features['ADX_Entry'] = round(adx, 2) if not pd.isna(adx) else None
        ml_features['DeMarker_Entry'] = round(demarker, 3) if not pd.isna(demarker) else None

        # DeMarker min in last 14 days
        demarker_14d = df['DeMarker'].iloc[-14:]
        ml_features['DeMarker_Min_14d'] = round(demarker_14d.min(), 3) if len(demarker_14d) > 0 else None

        # Relative strength vs SPY
        rs_ratio = None
        if spy_df is not None:
            try:
                if len(spy_df) > 21 and len(df) > 21:
                    spy_close = spy_df['Close'].iloc[-1]
                    spy_close_20 = spy_df['Close'].iloc[-21]
                    spy_perf = (spy_close - spy_close_20) / spy_close_20 * 100  # Convert to %

                    stock_close = df['Close'].iloc[-1]
                    stock_close_20 = df['Close'].iloc[-21]
                    stock_perf = (stock_close - stock_close_20) / stock_close_20 * 100  # Convert to %

                    # FIXED: Calculate relative performance (difference, not ratio)
                    rs_ratio = round(stock_perf - spy_perf, 2)  # Stock outperformance in %
            except:
                rs_ratio = None
        ml_features['RS_vs_SPY'] = rs_ratio

        # Volume analysis
        avg_vol_20 = df['Volume'].iloc[-21:-1].mean()
        recent_slice = df.iloc[-10:]  # Extended from 5 to 10 days for better pullback detection
        pullback_mask = (recent_slice['Low'] <= recent_slice['EMA_8'] * 1.005)

        if pullback_mask.any():
            pullback_vol_avg = recent_slice.loc[pullback_mask, 'Volume'].mean()
            volume_ratio = round(pullback_vol_avg / avg_vol_20, 2) if avg_vol_20 > 0 else None
        else:
            volume_ratio = None
        ml_features['Volume_Ratio'] = volume_ratio

        # Price distances from EMAs
        ml_features['Price_to_8EMA_%'] = round((current_price - ema_8) / ema_8 * 100, 2)
        ml_features['Price_to_21EMA_%'] = round((current_price - ema_21) / ema_21 * 100, 2)

        # EMA stack gap
        ml_features['EMA_Stack_Gap_%'] = round((ema_21 - sma_50) / current_price * 100, 2)

        # Pullback analysis
        swing_high = df['High'].iloc[-20:].max()
        swing_low = recent_slice['Low'].min()
        ml_features['Pullback_Depth_%'] = round((swing_high - swing_low) / swing_high * 100, 2)

        # Pullback duration
        if pullback_mask.any():
            pullback_start_idx = pullback_mask.idxmax()
            pullback_days = (df.index[-1] - pullback_start_idx).days
            ml_features['Pullback_Days'] = pullback_days
        else:
            ml_features['Pullback_Days'] = 0

        # Market context
        ml_features['SPY_RSI'] = market_context.get('SPY_RSI') if market_context else None
        ml_features['SPY_Trend'] = market_context.get('SPY_Trend') if market_context else None
        ml_features['VIX_Level'] = vix_level

        # Now perform strategy checks
        failure_reason = None

        # 1. Trend Filter: Price > 200 SMA
        if current_price <= sma_200:
            failure_reason = "Below 200 SMA"

        # 2. Stacked Moving Averages
        elif ema_21 <= sma_50:
            failure_reason = "Weak Trend (21<50)"

        # 3. Power Zone
        elif ema_8 <= ema_21:
            failure_reason = "Lost Power Zone"

        # 4. Support Hold
        elif current_price < ema_21:
            failure_reason = "Price < 21 EMA"

        # 5. RSI Filter
        elif rsi < 45 or rsi > 75:
            failure_reason = f"RSI {rsi:.0f} (Invalid)"

        # 6. ADX Filter (lowered from 25 to 20 - catches earlier trends)
        elif pd.isna(adx):
            failure_reason = "ADX N/A (Insufficient Data)"
        elif adx < 20:
            failure_reason = f"ADX {adx:.0f} (Weak)"

        # 7. Relative Strength - must outperform SPY (positive difference)
        elif rs_ratio is not None and rs_ratio < -2.0:  # Allow 2% underperformance
            failure_reason = "RS < SPY"

        # 8. Pullback check
        elif not pullback_mask.any():
            failure_reason = "No Pullback (Moved)"

        # 9. DeMarker Oversold History - REMOVED from exit validation
        # Once entry is valid, this shouldn't re-invalidate (+0.96% avg PnL on exits)
        # Keep for entry: require prior oversold to establish bounce setup
        # elif not (demarker_14d < 0.30).any():
        #     failure_reason = "No DeMarker Setup"

        # 10. DeMarker Bounce - Keep as soft validation (entry timing)
        elif demarker <= 0.30:
            failure_reason = "DeMarker <= 0.30"

        # 11. Choppy check - REMOVED: +0.91% avg PnL on exits (closing winners)
        # This filter was too restrictive for exit validation
        # else:
        #     prior_slice = df.iloc[-29:-14]
        #     oversold_days_count = (prior_slice['DeMarker'] < 0.30).sum()
        #     if oversold_days_count > 3:
        #         failure_reason = f"Choppy Market ({oversold_days_count} oversold days)"

        # 12. Volume dry up - REMOVED: Data showed 54% win rate on these exits
        # Volume spike on bounce can be confirmation, not rejection
        # if not failure_reason and volume_ratio and volume_ratio > 1.0:
        #     failure_reason = "High Pullback Vol"

        # If failed, return the reason but keep ML features for closed positions
        if failure_reason:
            return {'failure_reason': failure_reason, 'ml_features': ml_features}

        # Perfect setup - calculate targets
        pullback_date = pullback_mask[pullback_mask].index[-1].strftime('%Y-%m-%d')
        stop_loss = min(ema_21 * 0.99, swing_low * 0.99)
        target = swing_low + ((swing_high - swing_low) * 1.618)

        # Calculate Risk/Reward Ratio
        risk = current_price - stop_loss
        reward = target - current_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

        # FILTER: Reject setups with poor risk/reward (< 1.5)
        # Require at least 1.5:1 reward-to-risk ratio for quality setups
        if rr_ratio < 1.5:
            return {'failure_reason': f'Poor R/R ({rr_ratio})', 'ml_features': ml_features}

        return {
            'Ticker': ticker,
            'Signal': 'PERFECT SETUP',
            'Price': round(current_price, 2),
            'Pullback Date': pullback_date,
            'DeMarker': round(demarker, 2),
            'RSI': round(rsi, 2),
            'ADX': round(adx, 2),
            'Stop Loss': round(stop_loss, 2),
            'Fib Target': round(target, 2),
            'Risk/Reward': rr_ratio,
            '8 EMA': round(ema_8, 2),
            '21 EMA': round(ema_21, 2),
            'Volume': int(df['Volume'].iloc[-1]),
            'ml_features': ml_features
        }

    except Exception as e:
        return f"Error: {str(e)}"

def validate_exit_conditions(ticker, entry_date, entry_price, stop_loss, target_price=None, spy_df=None):
    """
    Separate exit validation with relaxed criteria.
    Only exits on major structure breaks, not minor technical violations.

    Args:
        ticker: Stock ticker symbol
        entry_date: Date position was entered
        entry_price: Price at entry
        stop_loss: Stop loss price
        target_price: Target price (optional)
        spy_df: SPY data (optional, unused but kept for compatibility)

    Returns: (should_exit, reason, exit_price)
    """
    try:
        # Calculate hold days
        entry_dt = pd.to_datetime(entry_date)
        current_dt = pd.to_datetime(datetime.now().date())
        hold_days = (current_dt - entry_dt).days

        # Minimum hold period - don't check exits for 3 days to avoid whipsaws
        if hold_days < 3:
            return False, None, None

        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if len(hist) < 200:
            return False, None, None

        # Get current price (including today if market is closed)
        if is_market_closed():
            current_price = hist['Close'].iloc[-1]
            current_low = hist['Low'].iloc[-1]
        else:
            # Market open - use previous close
            if len(hist) > 1:
                current_price = hist['Close'].iloc[-2]
                current_low = hist['Low'].iloc[-2]
                hist = hist.iloc[:-1]  # Exclude today's incomplete candle
            else:
                return False, None, None

        df = calculate_indicators(hist)

        # Get key levels
        sma_200 = df['SMA_200'].iloc[-1]
        ema_21 = df['EMA_21'].iloc[-1]

        # EXIT CONDITION 1: Target Hit (take profits)
        if target_price and current_price >= target_price:
            return True, "Target Hit", current_price

        # EXIT CONDITION 2: Stop Loss Hit (most important)
        if current_low <= stop_loss:
            return True, "Stop Loss Hit", min(current_price, stop_loss)

        # EXIT CONDITION 3: Major trend break - close below 200 SMA
        if current_price < sma_200 * 0.98:  # 2% buffer to avoid whipsaws
            return True, "Below 200 SMA", current_price

        # EXIT CONDITION 4: Prolonged weakness - below 21 EMA for 3+ days
        if hold_days >= 5:  # Only check after 5 days
            recent_3days = df['Close'].iloc[-3:]
            ema_21_3days = df['EMA_21'].iloc[-3:]
            below_21ema_count = (recent_3days < ema_21_3days).sum()

            if below_21ema_count >= 3:
                return True, "Sustained break of 21 EMA", current_price

        # EXIT CONDITION 5: Time stop - no progress after 20 days
        if hold_days > 20:
            # Check if we've made any progress
            profit_pct = (current_price - entry_price) / entry_price * 100
            if profit_pct < 2.0:  # Less than 2% gain after 20 days
                return True, "Time Stop (No Progress)", current_price

        # Otherwise, hold the position
        return False, None, None

    except Exception as e:
        logger.error(f"Error in exit validation for {ticker}: {e}")
        return False, None, None

def get_vix_level():
    """Get current VIX level"""
    try:
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="5d")
        if not vix_hist.empty:
            return round(vix_hist['Close'].iloc[-1], 2)
    except:
        pass
    return None

def run_screener(limit=None):
    """
    Run the enhanced screener with ML feature capture
    """
    logger.info("Starting enhanced screener run with ML features...")

    # Load history
    df_history = load_history()

    # Ensure columns exist
    for col in ['Pullback Date', 'Market Cap', 'Reason', 'Risk/Reward']:
        if col not in df_history.columns:
            df_history[col] = ""
    df_history['Pullback Date'] = df_history['Pullback Date'].fillna('')
    df_history['Market Cap'] = df_history['Market Cap'].fillna('')
    df_history['Reason'] = df_history['Reason'].fillna('')

    # Clean up duplicate active entries
    if not df_history.empty:
        active_mask = df_history['Status'] == 'Active'
        active_df = df_history[active_mask]

        duplicate_tickers = active_df['Ticker'].value_counts()
        duplicate_tickers = duplicate_tickers[duplicate_tickers > 1].index.tolist()

        if duplicate_tickers:
            logger.warning(f"Found duplicate active entries for: {duplicate_tickers}. Consolidating...")
            for ticker in duplicate_tickers:
                ticker_mask = (df_history['Ticker'] == ticker) & (df_history['Status'] == 'Active')
                active_indices = df_history.index[ticker_mask].tolist()
                for old_idx in active_indices[:-1]:
                    df_history.loc[old_idx, 'Status'] = 'Closed'
                    df_history.loc[old_idx, 'Exit Date'] = datetime.now().strftime("%Y-%m-%d")
                    df_history.loc[old_idx, 'Reason'] = 'Duplicate Consolidated'
            save_history(df_history)

    # Get tickers
    tickers, categories = get_market_tickers()

    # Backfill Market Cap
    if not df_history.empty:
        for idx, row in df_history.iterrows():
            if (pd.isna(row['Market Cap']) or row['Market Cap'] == '') and row['Ticker'] in categories:
                df_history.at[idx, 'Market Cap'] = categories[row['Ticker']]

    if limit:
        tickers = tickers[:limit]
        logger.info(f"Limiting scan to first {len(tickers)} tickers...")
        print(f"Limiting scan to first {len(tickers)} tickers...")

    # Fetch SPY data and calculate market context
    print("\nFetching SPY data and market context...")
    market_context = {}
    spy_df = None

    try:
        spy = yf.Ticker("SPY")
        spy_hist = spy.history(period="1y")

        # Only exclude today's candle if market is still open
        if not is_market_closed():
            today = datetime.now().date()
            if spy_hist.index[-1].date() == today:
                spy_hist = spy_hist.iloc[:-1]

        spy_df = calculate_indicators(spy_hist)

        # Calculate SPY RSI
        market_context['SPY_RSI'] = round(spy_df['RSI'].iloc[-1], 2) if not pd.isna(spy_df['RSI'].iloc[-1]) else None

        # SPY trend (distance from 200 SMA)
        spy_price = spy_df['Close'].iloc[-1]
        spy_200sma = spy_df['SMA_200'].iloc[-1]
        market_context['SPY_Trend'] = round((spy_price - spy_200sma) / spy_200sma * 100, 2)

        if spy_price < spy_200sma:
            print("\nWARNING: SPY is below 200 SMA. Market is Bearish.")
            print("Strict filtering is enabled. Expect few or no results.")
    except Exception as e:
        logger.error(f"Failed to fetch SPY data: {e}")

    # Get VIX level
    vix_level = get_vix_level()
    print(f"VIX Level: {vix_level}")

    results = []
    all_analysis = {}

    # Scan in parallel with enhanced analysis
    print("\nStarting enhanced scan with ML features...")
    start_time = time.time()

    analyze_func = functools.partial(analyze_stock_enhanced,
                                    spy_df=spy_df,
                                    vix_level=vix_level,
                                    market_context=market_context)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(analyze_func, t): t for t in tickers}

        with tqdm(total=len(tickers), unit="stock") as pbar:
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                result = future.result()
                all_analysis[ticker] = result

                # Check if it's a valid setup (not a failure reason string)
                if isinstance(result, dict):
                    if 'failure_reason' not in result:
                        # Valid setup
                        result['Market Cap'] = categories.get(result['Ticker'], 'Unknown')
                        results.append(result)
                    else:
                        # Failed but we have ML features for later use
                        all_analysis[ticker] = result['failure_reason']

                pbar.update(1)

    elapsed = time.time() - start_time
    print(f"\nScan complete in {elapsed:.2f} seconds.")

    # Process results
    if results:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values(['Volume'], ascending=[False])
    else:
        df_results = pd.DataFrame()

    # Update history with enhanced data
    current_date = datetime.now().strftime("%Y-%m-%d")

    new_tickers_this_run = []
    closed_tickers_this_run = []

    # Handle closed positions WITH NEW EXIT VALIDATION
    if not df_history.empty:
        current_tickers = df_results['Ticker'].tolist() if not df_results.empty else []
        active_mask = df_history['Status'] == 'Active'

        # Use new exit validation for all active positions
        closed_tickers_this_run = []

        for idx, row in df_history[active_mask].iterrows():
            t = row['Ticker']
            entry_date = row['Entry Date']
            entry_price = row['Entry Price']
            stop_loss = row['Stop Loss']
            target_price = row.get('Fib Target', None)

            # Use the new exit validation function
            should_exit, exit_reason, exit_price_suggested = validate_exit_conditions(
                t, entry_date, entry_price, stop_loss, target_price, spy_df
            )

            if should_exit:
                closed_tickers_this_run.append(t)

                # Update the history with exit information
                df_history.loc[idx, 'Status'] = 'Closed'
                df_history.loc[idx, 'Exit Date'] = current_date
                df_history.loc[idx, 'Exit Price'] = exit_price_suggested
                df_history.loc[idx, 'Reason'] = exit_reason

                # Calculate P&L
                if exit_price_suggested and entry_price and entry_price > 0:
                    pnl_pct = round((exit_price_suggested - entry_price) / entry_price * 100, 2)
                    df_history.loc[idx, 'PnL %'] = pnl_pct

                    # Calculate hold days
                    entry_dt = pd.to_datetime(entry_date)
                    exit_dt = pd.to_datetime(current_date)
                    hold_days = (exit_dt - entry_dt).days
                    df_history.loc[idx, 'Hold Days'] = hold_days

                    # Mark if hit stop or target
                    if exit_reason == "Stop Loss Hit":
                        df_history.loc[idx, 'Hit Stop'] = True
                    elif exit_price_suggested >= row.get('Fib Target', float('inf')):
                        df_history.loc[idx, 'Hit Target'] = True

    # Add new setups with ML features
    if not df_results.empty:
        for _, row in df_results.iterrows():
            ticker = row['Ticker']
            ml_features = row.get('ml_features', {})

            mask = (df_history['Ticker'] == ticker) & (df_history['Status'] == 'Active')
            active_count = mask.sum()

            if active_count > 0:
                # Update existing
                if active_count > 1:
                    logger.warning(f"Found {active_count} active rows for {ticker}, consolidating...")
                    active_indices = df_history.index[mask].tolist()
                    for old_idx in active_indices[:-1]:
                        df_history.loc[old_idx, 'Status'] = 'Closed'
                        df_history.loc[old_idx, 'Exit Date'] = current_date
                        df_history.loc[old_idx, 'Reason'] = 'Duplicate Consolidated'

                mask = (df_history['Ticker'] == ticker) & (df_history['Status'] == 'Active')
                idx = df_history.index[mask]
                df_history.loc[idx, 'Current Price'] = row['Price']
                df_history.loc[idx, 'Stop Loss'] = row['Stop Loss']
                df_history.loc[idx, 'Fib Target'] = row['Fib Target']
                df_history.loc[idx, 'Risk/Reward'] = row['Risk/Reward']
                df_history.loc[idx, 'Pullback Date'] = row['Pullback Date']
                df_history.loc[idx, 'Market Cap'] = row['Market Cap']

                # Update ML features (but NEVER overwrite ML_Confidence, ML_Win_Prob, ML_Expected_PnL)
                # These are frozen at entry for validation purposes
                frozen_ml_cols = ['ML_Confidence', 'ML_Win_Prob', 'ML_Expected_PnL']
                for key, value in ml_features.items():
                    if key not in frozen_ml_cols:
                        df_history.loc[idx, key] = value
            else:
                # Add new with ML features
                new_row = {
                    'Ticker': ticker,
                    'Entry Date': current_date,
                    'Pullback Date': row['Pullback Date'],
                    'Entry Price': row['Price'],
                    'Current Price': row['Price'],
                    'Stop Loss': row['Stop Loss'],
                    'Fib Target': row['Fib Target'],
                    'Risk/Reward': row['Risk/Reward'],
                    'Status': 'Active',
                    'Exit Date': '',
                    'Market Cap': row['Market Cap'],
                    'Reason': '',
                    # Add sector
                    'Sector': get_stock_sector(ticker),
                    # Add all ML features
                    **ml_features
                }
                df_history = pd.concat([df_history, pd.DataFrame([new_row])], ignore_index=True)
                new_tickers_this_run.append(ticker)

    # CLEANUP: Remove active positions with R/R < 1.5
    # Close them with reason "Poor R/R" instead of keeping them active
    if 'Risk/Reward' in df_history.columns:
        poor_rr_mask = (df_history['Status'] == 'Active') & (df_history['Risk/Reward'] < 1.5)
        poor_rr_count = poor_rr_mask.sum()

        if poor_rr_count > 0:
            logger.info(f"Cleaning up {poor_rr_count} active positions with R/R < 1.5")
            df_history.loc[poor_rr_mask, 'Status'] = 'Closed'
            df_history.loc[poor_rr_mask, 'Exit Date'] = current_date
            df_history.loc[poor_rr_mask, 'Exit Price'] = df_history.loc[poor_rr_mask, 'Current Price']
            df_history.loc[poor_rr_mask, 'Reason'] = 'Poor R/R (< 1.5)'

            # Calculate PnL for these closed positions
            for idx in df_history.index[poor_rr_mask]:
                entry_price = df_history.loc[idx, 'Entry Price']
                exit_price = df_history.loc[idx, 'Exit Price']
                if pd.notna(entry_price) and pd.notna(exit_price) and entry_price > 0:
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    df_history.loc[idx, 'PnL %'] = round(pnl_pct, 2)
                    df_history.loc[idx, 'PnL $'] = round(pnl_pct * 50, 2)  # Assuming $5000 position

    # Save enhanced history
    save_history(df_history)

    # AUTO-RETRAIN ML MODELS if we have enough new closed trades
    try:
        from ml_confidence_scorer import TradingConfidenceScorer

        # Check if we should retrain
        closed_with_ml = df_history[(df_history['Status'] == 'Closed') & df_history['RSI_Entry'].notna()]

        # Check last training time
        model_path = str(_DATA_DIR / 'ml_models' / 'win_classifier_latest.pkl')
        should_retrain = False

        if os.path.exists(model_path):
            # Get model age
            model_age_days = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))).days

            # Retrain if: model is older than 7 days OR we have 10+ new closed trades since last training
            if model_age_days > 7:
                logger.info(f"ML model is {model_age_days} days old. Retraining...")
                should_retrain = True
            elif len(closed_with_ml) >= 30:  # Need minimum data to train
                # Count trades closed in last 7 days
                recent_closed = closed_with_ml[pd.to_datetime(closed_with_ml['Exit Date']) > (datetime.now() - timedelta(days=7))]
                if len(recent_closed) >= 10:
                    logger.info(f"Found {len(recent_closed)} new closed trades in last 7 days. Retraining ML models...")
                    should_retrain = True
        else:
            # No model exists, train for first time
            if len(closed_with_ml) >= 30:
                logger.info("No ML models found. Training for first time...")
                should_retrain = True

        if should_retrain:
            scorer = TradingConfidenceScorer()
            if scorer.train_models():
                logger.info("ML models retrained successfully!")
            else:
                logger.warning("ML model retraining failed. Continuing without ML scores.")
    except Exception as e:
        logger.error(f"Error during ML auto-retrain: {e}")

    # Prepare return dataframes
    df_new_run = df_history[
        (df_history['Ticker'].isin(new_tickers_this_run)) &
        (df_history['Entry Date'] == current_date) &
        (df_history['Status'] == 'Active')
    ]
    df_closed_run = df_history[df_history['Ticker'].isin(closed_tickers_this_run)]

    # ADD ML CONFIDENCE SCORES to new setups (after df_new_run is created)
    try:
        from ml_confidence_scorer import TradingConfidenceScorer

        if not df_new_run.empty:
            scorer = TradingConfidenceScorer()
            if scorer.load_models():
                ml_scores = []
                for _, row in df_new_run.iterrows():
                    features = row.to_dict()
                    prediction = scorer.predict_confidence(features)
                    if prediction:
                        ml_scores.append({
                            'Ticker': row['Ticker'],
                            'ML_Confidence': prediction['confidence'],
                            'ML_Win_Prob': prediction['win_probability'],
                            'ML_Expected_PnL': prediction['expected_pnl']
                        })

                if ml_scores:
                    df_ml = pd.DataFrame(ml_scores)
                    # Merge ML scores into df_new_run
                    df_new_run = df_new_run.merge(df_ml, on='Ticker', how='left')
                    logger.info(f"Added ML confidence scores to {len(ml_scores)} new setups")

                    # CRITICAL: Save ML confidence to history file (ONCE at entry, never change)
                    # This ensures we can validate predictions against actual outcomes
                    for _, ml_row in df_ml.iterrows():
                        ticker = ml_row['Ticker']
                        mask = (df_history['Ticker'] == ticker) & \
                               (df_history['Status'] == 'Active') & \
                               (df_history['Entry Date'] == current_date)

                        if mask.any():
                            df_history.loc[mask, 'ML_Confidence'] = ml_row['ML_Confidence']
                            df_history.loc[mask, 'ML_Win_Prob'] = ml_row['ML_Win_Prob']
                            df_history.loc[mask, 'ML_Expected_PnL'] = ml_row['ML_Expected_PnL']

                    # Save updated history with ML scores
                    df_history.to_csv(HISTORY_FILE, index=False)
                    logger.info(f"Saved ML confidence scores to history file")
    except Exception as e:
        logger.error(f"Error adding ML confidence scores: {e}")

    return df_new_run, df_closed_run, df_history

# Insider buying cache
INSIDER_CACHE_FILE = str(_DATA_DIR / 'insider_cache.json')
_insider_cache = None
_cache_lock = threading.Lock()  # Thread safety for concurrent access

def load_insider_cache():
    """Load insider buying cache from file (thread-safe)"""
    global _insider_cache

    with _cache_lock:
        if _insider_cache is not None:
            return _insider_cache

        if os.path.exists(INSIDER_CACHE_FILE):
            try:
                import json
                with open(INSIDER_CACHE_FILE, 'r') as f:
                    _insider_cache = json.load(f)
                    logger.info(f"Loaded insider cache with {len(_insider_cache)} entries")
            except Exception as e:
                logger.warning(f"Could not load insider cache: {e}")
                _insider_cache = {}
        else:
            _insider_cache = {}

        return _insider_cache

def save_insider_cache():
    """Save insider buying cache to file (thread-safe)"""
    global _insider_cache

    with _cache_lock:
        if _insider_cache is None:
            return

        try:
            import json
            # Create a copy to avoid iteration issues
            cache_copy = dict(_insider_cache)
            with open(INSIDER_CACHE_FILE, 'w') as f:
                json.dump(cache_copy, f, indent=2)
            logger.info(f"Saved insider cache with {len(cache_copy)} entries")
        except Exception as e:
            logger.error(f"Could not save insider cache: {e}")

def get_insider_shares_from_finviz(ticker):
    """
    Fallback: Fetch insider trading summary from FinViz.
    Returns approximate insider buying activity (simplified boolean indicator).
    """
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        # Look for insider trading table
        # FinViz shows recent insider transactions
        if 'Insider Trading' in response.text and 'Buy' in response.text:
            # Simple heuristic: if we see "Buy" in insider section, return positive indicator
            # This is a simplified approach - returns 1 to indicate buying activity detected
            return 1
        return 0

    except Exception as e:
        logger.debug(f"FinViz fetch failed for {ticker}: {e}")
        return None

def get_insider_shares_from_yahoo(ticker):
    """
    Fallback: Use yfinance to get insider transactions.
    Returns NET insider activity (buys - sells) by key insiders in last 3 months.

    Positive number = Net buying
    Negative number = Net selling

    Note: Yahoo Finance doesn't distinguish between purchases and stock awards.
    We use heuristics to filter for likely purchases:
    - Exclude very large transactions (>100k shares) likely to be stock awards
    - Focus on smaller acquisitions more likely to be open-market purchases
    - Only count transactions with associated value (actual purchases have cost)
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)

        # Get insider transactions
        insider_txns = stock.insider_transactions

        if insider_txns is None or insider_txns.empty:
            return 0

        # Filter to last 3 months (to match OpenInsider timeframe)
        three_months_ago = datetime.now() - timedelta(days=90)

        # Convert Start Date to datetime
        if 'Start Date' in insider_txns.columns:
            insider_txns['Start Date'] = pd.to_datetime(insider_txns['Start Date'], errors='coerce')
            insider_txns = insider_txns[insider_txns['Start Date'] >= three_months_ago]

        if insider_txns.empty:
            return 0

        # Filter by key positions (CEO, CFO, Directors, Officers, etc.)
        if 'Position' in insider_txns.columns:
            def is_key_insider(pos):
                if pd.isna(pos) or not isinstance(pos, str):
                    return False
                pos_upper = pos.upper()
                # Focus on C-suite, Directors, and Officers (more meaningful signals)
                key_titles = ['CEO', 'CFO', 'COO', 'PRESIDENT', 'CHAIRMAN', 'DIRECTOR', 'CHIEF', 'OFFICER']
                return any(title in pos_upper for title in key_titles)

            insider_txns = insider_txns[insider_txns['Position'].apply(is_key_insider)]

        if insider_txns.empty:
            return 0

        # Calculate NET insider activity (buys - sells)
        # IMPORTANT: Yahoo Finance stores Shares as POSITIVE even for sales!
        # We need to check the 'Text' column to determine if it's a buy or sell
        if 'Shares' in insider_txns.columns and 'Text' in insider_txns.columns:
            net_shares = 0

            # Only count transactions with values (actual market transactions)
            if 'Value' in insider_txns.columns:
                market_txns = insider_txns[insider_txns['Value'].notna() & (insider_txns['Value'] != 0)].copy()

                if not market_txns.empty:
                    # Exclude extremely large transactions (>100k shares) - likely stock awards
                    reasonable_txns = market_txns[market_txns['Shares'] <= 100000].copy()

                    if not reasonable_txns.empty:
                        # Determine buy vs sell from Text column
                        def get_signed_shares(row):
                            shares = row['Shares']
                            text = str(row['Text']).lower() if pd.notna(row['Text']) else ''

                            # Check if it's a sale (negate the shares)
                            if 'sale' in text or 'sell' in text:
                                return -shares
                            # Check if it's a purchase (keep positive)
                            elif 'purchase' in text or 'buy' in text or 'option exercise' in text:
                                return shares
                            # Stock awards have Value=0, so shouldn't be in market_txns
                            # But if somehow here, treat as neither buy nor sell
                            else:
                                return 0

                        reasonable_txns['Signed_Shares'] = reasonable_txns.apply(get_signed_shares, axis=1)
                        net_shares = reasonable_txns['Signed_Shares'].sum()
                        return int(net_shares) if not pd.isna(net_shares) else 0

        return 0

    except Exception as e:
        logger.debug(f"Yahoo Finance insider fetch failed for {ticker}: {e}")
        return None

def get_insider_shares_from_openinsider(ticker):
    """
    Primary: Fetch NET insider activity from OpenInsider.com.
    Returns NET shares (purchases - sales) by key insiders in last 3 months.

    Positive number = Net buying
    Negative number = Net selling
    """
    try:
        def fetch_transactions(transaction_type):
            """Fetch purchases ('p') or sales ('s') from OpenInsider"""
            url = "https://openinsider.com/screener"
            params = {
                's': ticker,
                'fd': 90,  # 3 months
                't': transaction_type,  # 'p' for Purchase, 's' for Sale
                'cnt': 100,
                'page': 1
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.debug(f"OpenInsider returned status {response.status_code} for {ticker} ({transaction_type})")
                return None

            dfs = pd.read_html(response.text)
            if not dfs:
                return 0

            df = None
            for d in dfs:
                if 'Ticker' in d.columns and 'Qty' in d.columns and 'Title' in d.columns:
                    df = d
                    break
            if df is None:
                return 0

            def is_target_title(t):
                if not isinstance(t, str): return False
                t = t.upper()
                targets = ['CEO', 'CFO', 'COO', 'VP', 'VICE PRESIDENT', 'PRESIDENT', 'CHIEF EXECUTIVE', 'CHIEF FINANCIAL', 'CHIEF OPERATING', 'DIR', 'DIRECTOR', 'CHAIRMAN']
                return any(x in t for x in targets)

            relevant = df[df['Title'].apply(is_target_title)]

            total = 0
            for q in relevant['Qty']:
                try:
                    q_str = str(q).replace('+', '').replace(',', '').strip()
                    total += float(q_str)
                except (ValueError, TypeError, AttributeError):
                    continue

            return int(total)

        # Fetch both purchases and sales
        purchases = fetch_transactions('p')
        sales = fetch_transactions('s')

        # If either fetch failed completely, return None (will trigger fallback)
        if purchases is None or sales is None:
            return None

        # Return net activity (positive = buying, negative = selling)
        net_activity = purchases - sales
        return net_activity

    except (requests.RequestException, ValueError, KeyError, IndexError) as e:
        logger.debug(f"OpenInsider fetch failed for {ticker}: {type(e).__name__}")
        return None

def get_insider_shares_purchased(ticker, force_refresh=False):
    """
    Fetches insider purchases for a specific ticker in the last 3-6 months.
    Uses cache if data is less than 10 days old.

    Tries multiple data sources in order:
    1. OpenInsider.com (most detailed)
    2. Yahoo Finance via yfinance (reliable fallback)
    3. FinViz (simple indicator)

    Args:
        ticker: Stock ticker symbol
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        Number of shares purchased by insiders (or 1 for FinViz indicator)
    """
    global _insider_cache

    # Ensure cache is loaded
    load_insider_cache()

    # Check cache unless force refresh (thread-safe read)
    with _cache_lock:
        if not force_refresh and ticker in _insider_cache:
            cached_data = _insider_cache[ticker]
            cache_date = datetime.strptime(cached_data['date'], '%Y-%m-%d')
            days_old = (datetime.now() - cache_date).days

            if days_old <= 10:
                logger.debug(f"Using cached insider data for {ticker} ({days_old} days old)")
                return cached_data['shares']

    # Try multiple data sources with fallback
    shares = None
    source = None

    # Try OpenInsider first (most detailed)
    shares = get_insider_shares_from_openinsider(ticker)
    if shares is not None:
        source = "OpenInsider"
    else:
        # Fallback to Yahoo Finance
        shares = get_insider_shares_from_yahoo(ticker)
        if shares is not None:
            source = "YahooFinance"
        else:
            # Final fallback to FinViz
            shares = get_insider_shares_from_finviz(ticker)
            if shares is not None:
                source = "FinViz"
            else:
                # All sources failed
                shares = 0
                source = "None"

    if source != "None":
        logger.debug(f"Fetched insider data for {ticker} from {source}: {shares} shares")

    # Cache successful results (thread-safe)
    if shares is not None and shares >= 0:
        with _cache_lock:
            _insider_cache[ticker] = {
                'shares': shares,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'source': source
            }

    return shares if shares is not None else 0

def main():
    print("\n" + "="*60)
    print(" S&P 1500 SCREENER: ENHANCED WITH ML FEATURES ")
    print("="*60)

    # Default to scanning ALL tickers (pass limit=None to run_screener)
    # To test with a smaller sample, modify the limit parameter below
    limit = None  # Set to a number (e.g., 50) for quick testing

    # DYNAMIC POSITION MANAGEMENT - Re-evaluate active positions before screening
    print("\n" + "="*60)
    print(" STEP 1: DYNAMIC POSITION MANAGEMENT ")
    print("="*60)
    position_actions = {}
    try:
        from dynamic_position_manager import DynamicPositionManager
        manager = DynamicPositionManager()
        position_actions = manager.process_all_positions()
    except Exception as e:
        logger.error(f"Error in dynamic position management: {e}")
        print(f"Skipping dynamic position management due to error: {e}")

    # Run Screener
    print("\n" + "="*60)
    print(" STEP 2: SCREENING FOR NEW SETUPS ")
    print("="*60)
    df_new_run, df_closed_run, df_history = run_screener(limit)

    # Load AO Portfolio for cross-reference
    ao_tickers = set()
    if os.path.exists('ao_saucer_portfolio.csv'):
        try:
            ao_df = pd.read_csv('ao_saucer_portfolio.csv')
            if 'Ticker' in ao_df.columns:
                ao_tickers = set(ao_df['Ticker'].tolist())
        except Exception:
            pass

    # Fetch Insider Buying Data (smart caching)
    # Only fetch for: 1) New setups, 2) Existing setups with stale cache (>10 days old)
    global _insider_cache
    insider_data = {}
    active_tickers = df_history[df_history['Status'] == 'Active']['Ticker'].unique().tolist()
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Load cache to check which tickers need refresh
    load_insider_cache()

    tickers_to_fetch = []
    for ticker in active_tickers:
        # Check if it's a new setup (entered today)
        is_new = ((df_history['Ticker'] == ticker) &
                  (df_history['Status'] == 'Active') &
                  (df_history['Entry Date'] == current_date)).any()

        if is_new:
            tickers_to_fetch.append(ticker)
        else:
            # Check cache for existing setup (thread-safe read)
            with _cache_lock:
                if ticker in _insider_cache:
                    # Check cache age
                    cache_date = datetime.strptime(_insider_cache[ticker]['date'], '%Y-%m-%d')
                    days_old = (datetime.now() - cache_date).days
                    if days_old > 10:
                        tickers_to_fetch.append(ticker)
                    else:
                        # Use cached value
                        insider_data[ticker] = _insider_cache[ticker]['shares']
                else:
                    # Not in cache, fetch it
                    tickers_to_fetch.append(ticker)

    if tickers_to_fetch:
        print(f"\nFetching insider buying data for {len(tickers_to_fetch)} tickers ({len(active_tickers) - len(tickers_to_fetch)} from cache)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(get_insider_shares_purchased, t): t for t in tickers_to_fetch}
            for future in concurrent.futures.as_completed(future_to_ticker):
                t = future_to_ticker[future]
                result = future.result()
                insider_data[t] = result if result is not None else 0

        # Save cache once after all fetches complete (instead of after each fetch)
        # The individual fetches already updated the cache dict, just need one save
        if _insider_cache:
            logger.info(f"Completed fetching {len(tickers_to_fetch)} tickers, saving cache...")
            save_insider_cache()

        # Check if OpenInsider is accessible
        successful_fetches = sum(1 for v in insider_data.values() if v >= 0)
        if successful_fetches == 0 and len(tickers_to_fetch) > 0:
            print("  WARNING: OpenInsider.com appears to be inaccessible. Insider data unavailable.")
    else:
        print(f"\nUsing cached insider data for all {len(active_tickers)} active tickers")

    # Debug: Log insider data summary
    total_with_buys = sum(1 for v in insider_data.values() if v > 0)
    logger.info(f"Insider data collected: {len(insider_data)} tickers, {total_with_buys} with insider buys")

    # Display Results with ML feature indicators
    print("\n" + "="*60)
    print(" SCREENER RESULTS WITH ML FEATURES ")
    print("="*60)

    # Show cleanup summary if any positions were removed
    if 'Risk/Reward' in df_history.columns:
        recent_cleanup = df_history[
            (df_history['Status'] == 'Closed') &
            (df_history['Reason'] == 'Poor R/R (< 1.5)') &
            (df_history['Exit Date'] == current_date)
        ]
        if not recent_cleanup.empty:
            print(f"\nCleaned up {len(recent_cleanup)} position(s) with R/R < 1.5 (minimum required: 1.5)")
            print(f"  These positions have been closed and won't appear in active results.")

    # Define Market Cap Sort Order
    mc_order = {'Large': 0, 'Mid': 1, 'Small': 2}

    # Display New
    df_new = df_history[(df_history['Status'] == 'Active') & (df_history['Entry Date'] == current_date)].copy()
    if not df_new.empty:
        df_new['Has Entry'] = df_new['Ticker'].apply(lambda x: 'Yes' if x in ao_tickers else 'No')
        # Format insider data: positive for net buying, negative for net selling
        df_new['Insider Net'] = df_new['Ticker'].map(insider_data).fillna(0).astype(int).apply(lambda x: f"{x:+,}" if x != 0 else "0")

        # Sort by Market Cap
        df_new['MC_Rank'] = df_new['Market Cap'].map(mc_order).fillna(3)
        df_new = df_new.sort_values(['MC_Rank', 'Ticker'])

        # Sort by ML Confidence if available
        if 'ML_Confidence' in df_new.columns:
            conf_order_map = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            df_new['ML_Conf_Order'] = df_new['ML_Confidence'].map(conf_order_map).fillna(3)
            df_new = df_new.sort_values(['ML_Conf_Order', 'MC_Rank', 'Ticker'])

            print(f"\n[NEW SETUPS] ({len(df_new)}) - with ML Confidence Scores")
            display_cols = ['Ticker', 'Market Cap', 'Entry Price', 'RSI_Entry', 'ADX_Entry', 'RS_vs_SPY', 'Stop Loss', 'Fib Target', 'Risk/Reward', 'ML_Confidence', 'ML_Win_Prob', 'ML_Expected_PnL', 'Has Entry', 'Insider Net']
            # Filter to only existing columns
            display_cols = [col for col in display_cols if col in df_new.columns]
            print(df_new[display_cols].to_string(index=False))

            # Summary by confidence
            high_count = len(df_new[df_new['ML_Confidence'] == 'HIGH'])
            med_count = len(df_new[df_new['ML_Confidence'] == 'MEDIUM'])
            low_count = len(df_new[df_new['ML_Confidence'] == 'LOW'])
            print(f"\nML Confidence Summary: HIGH={high_count}, MEDIUM={med_count}, LOW={low_count}")
        else:
            print(f"\n[NEW SETUPS] ({len(df_new)})")
            print(df_new[['Ticker', 'Market Cap', 'Entry Price', 'RSI_Entry', 'ADX_Entry', 'RS_vs_SPY', 'Stop Loss', 'Fib Target', 'Risk/Reward', 'Has Entry', 'Insider Net']].to_string(index=False))

    # Display Existing
    df_existing = df_history[(df_history['Status'] == 'Active') & (df_history['Entry Date'] != current_date)].copy()
    if not df_existing.empty:
        df_existing['Has Entry'] = df_existing['Ticker'].apply(lambda x: 'Yes' if x in ao_tickers else 'No')
        # Format insider data: positive for net buying, negative for net selling
        df_existing['Insider Net'] = df_existing['Ticker'].map(insider_data).fillna(0).astype(int).apply(lambda x: f"{x:+,}" if x != 0 else "0")

        # Sort by Market Cap
        df_existing['MC_Rank'] = df_existing['Market Cap'].map(mc_order).fillna(3)
        df_existing = df_existing.sort_values(['MC_Rank', 'Ticker'])

        print(f"\n[EXISTING SETUPS] ({len(df_existing)})")
        print(df_existing[['Ticker', 'Market Cap', 'Entry Date', 'Entry Price', 'Current Price', 'RSI_Entry', 'ADX_Entry', 'Risk/Reward', 'Has Entry', 'Insider Net']].to_string(index=False))

    # Display Closed with P&L
    df_closed = df_history[df_history['Status'] == 'Closed']
    if not df_closed.empty:
        # Show recent closed positions
        recent_closed = df_closed.sort_values('Exit Date', ascending=False).head(20)
        print(f"\n[RECENT CLOSED SETUPS] (Last 20 of {len(df_closed)} total)")
        display_cols = ['Ticker', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price', 'PnL %', 'Hold Days', 'Reason']
        # Filter to only show columns that exist
        display_cols = [col for col in display_cols if col in recent_closed.columns]
        print(recent_closed[display_cols].to_string(index=False))

    if df_new.empty and df_existing.empty and df_closed.empty:
        print("\nNo active or closed setups found.")

    # Display ML feature summary
    print("\n" + "="*60)
    print(" ML FEATURE CAPTURE SUMMARY ")
    print("="*60)

    # Count how many rows have ML features
    ml_columns = ['RSI_Entry', 'ADX_Entry', 'DeMarker_Entry', 'RS_vs_SPY', 'Volume_Ratio']
    active_with_ml = df_history[(df_history['Status'] == 'Active') & (df_history['RSI_Entry'].notna())].shape[0]
    closed_with_ml = df_history[(df_history['Status'] == 'Closed') & (df_history['RSI_Entry'].notna())].shape[0]

    print(f"Active positions with ML features: {active_with_ml}")
    print(f"Closed positions with ML features: {closed_with_ml}")
    print(f"Total positions with ML features: {active_with_ml + closed_with_ml}")

    print(f"\nHistory updated in {HISTORY_FILE}")

    # AUTO-RUN ML CONFIDENCE SCORER for all active positions
    try:
        from ml_confidence_scorer import TradingConfidenceScorer

        active_positions = df_history[df_history['Status'] == 'Active']
        if not active_positions.empty:
            print("\n" + "="*60)
            print(" AUTO-RUNNING ML CONFIDENCE SCORER ")
            print("="*60)

            scorer = TradingConfidenceScorer()

            # Prepare screener data for export (combine new and existing)
            df_screener_export = pd.concat([df_new, df_existing], ignore_index=True) if not df_new.empty or not df_existing.empty else None

            # Add position actions to insider_data for passing to ML scorer
            # Use a special key that won't conflict with ticker names
            insider_data_with_actions = insider_data.copy()
            insider_data_with_actions['__position_actions__'] = position_actions

            # Pass the insider_data (with position actions), and screener_data to avoid refetching
            scorer.evaluate_active_positions(insider_data=insider_data_with_actions, screener_data=df_screener_export)
    except Exception as e:
        logger.error(f"Error running ML confidence scorer: {e}")

if __name__ == "__main__":
    main()
