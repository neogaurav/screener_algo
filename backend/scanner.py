#!/usr/bin/env python3
"""
8/21 EMA Screener - Main entry point for scheduled GitHub Actions runs.
Writes JSON data files consumed by the frontend.
"""
import json
import sys
import math
import logging
import concurrent.futures
from pathlib import Path
from datetime import datetime

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from screener import (
    run_screener, get_vix_level, get_insider_shares_purchased,
    load_insider_cache, save_insider_cache, calculate_indicators
)
from dynamic_position_manager import DynamicPositionManager
from ml_confidence_scorer import TradingConfidenceScorer

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _safe(val):
    """Convert NaN/inf to None for JSON serialization."""
    if val is None:
        return None
    try:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, (pd.Timestamp, datetime)):
        return str(val)[:10]
    return val


def _row_to_setup(row, insider_data):
    ticker = str(row.get('Ticker', ''))
    entry_date = str(row.get('Entry Date', ''))
    hold_days = 0
    try:
        hold_days = (pd.Timestamp.now() - pd.to_datetime(entry_date)).days
    except Exception:
        pass
    return {
        'ticker': ticker,
        'entry_date': entry_date,
        'entry_price': _safe(row.get('Entry Price')),
        'current_price': _safe(row.get('Current Price')),
        'stop_loss': _safe(row.get('Stop Loss')),
        'fib_target': _safe(row.get('Fib Target')),
        'risk_reward': _safe(row.get('Risk/Reward')),
        'market_cap': str(row.get('Market Cap', '') or ''),
        'sector': str(row.get('Sector', '') or ''),
        'pullback_date': str(row.get('Pullback Date', '') or ''),
        'rsi': _safe(row.get('RSI_Entry')),
        'adx': _safe(row.get('ADX_Entry')),
        'demarker': _safe(row.get('DeMarker_Entry')),
        'rs_vs_spy': _safe(row.get('RS_vs_SPY')),
        'volume_ratio': _safe(row.get('Volume_Ratio')),
        'price_to_8ema_pct': _safe(row.get('Price_to_8EMA_%')),
        'price_to_21ema_pct': _safe(row.get('Price_to_21EMA_%')),
        'ml_confidence': str(row.get('ML_Confidence') or 'N/A'),
        'ml_win_prob': _safe(row.get('ML_Win_Prob')),
        'ml_expected_pnl': _safe(row.get('ML_Expected_PnL')),
        'insider_net': int(insider_data.get(ticker, 0) or 0),
        'hold_days': hold_days,
    }


def write_screener_json(df_new, df_existing, spy_info, vix_level, insider_data):
    new_setups = [_row_to_setup(row, insider_data) for _, row in df_new.iterrows()] if not df_new.empty else []
    existing_setups = [_row_to_setup(row, insider_data) for _, row in df_existing.iterrows()] if not df_existing.empty else []
    data = {
        'new_setups': new_setups,
        'existing_setups': existing_setups,
        'summary': {
            'new_count': len(new_setups),
            'existing_count': len(existing_setups),
            'total_active': len(new_setups) + len(existing_setups),
            'spy_price': spy_info.get('price'),
            'spy_above_200sma': spy_info.get('above_200sma'),
            'spy_rsi': spy_info.get('rsi'),
            'vix': vix_level,
        },
        'last_updated': datetime.utcnow().isoformat() + 'Z',
    }
    with open(DATA_DIR / 'screener.json', 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Wrote screener.json: {len(new_setups)} new, {len(existing_setups)} existing")


def write_closed_json(df_history):
    df_closed = df_history[df_history['Status'] == 'Closed'].copy()
    positions = []
    for _, row in df_closed.iterrows():
        entry_price = _safe(row.get('Entry Price'))
        exit_price = _safe(row.get('Exit Price'))
        pnl_pct = _safe(row.get('PnL %'))
        pnl_dollars = _safe(row.get('PnL $'))
        if pnl_dollars is None and pnl_pct is not None and entry_price:
            pnl_dollars = round((pnl_pct / 100) * 5000, 2)
        positions.append({
            'ticker': str(row.get('Ticker', '')),
            'entry_date': str(row.get('Entry Date', '') or ''),
            'exit_date': str(row.get('Exit Date', '') or ''),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'hold_days': _safe(row.get('Hold Days')),
            'exit_reason': str(row.get('Reason', '') or ''),
            'market_cap': str(row.get('Market Cap', '') or ''),
            'sector': str(row.get('Sector', '') or ''),
            'ml_confidence': str(row.get('ML_Confidence') or 'N/A'),
            'risk_reward': _safe(row.get('Risk/Reward')),
        })
    positions.sort(key=lambda x: x.get('exit_date', '') or '', reverse=True)
    winners = [p for p in positions if p.get('pnl_pct') is not None and p['pnl_pct'] > 0]
    losers = [p for p in positions if p.get('pnl_pct') is not None and p['pnl_pct'] <= 0]
    total_pnl = sum(p['pnl_dollars'] or 0 for p in positions if p.get('pnl_dollars') is not None)
    gross_profit = sum(p['pnl_dollars'] or 0 for p in winners if p.get('pnl_dollars') is not None)
    gross_loss = abs(sum(p['pnl_dollars'] or 0 for p in losers if p.get('pnl_dollars') is not None))
    perf = {
        'total_trades': len(positions),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': round(len(winners) / len(positions) * 100, 1) if positions else 0.0,
        'total_pnl_dollars': round(total_pnl, 2),
        'avg_pnl_pct': round(sum(p['pnl_pct'] for p in positions if p.get('pnl_pct') is not None) / len(positions), 2) if positions else 0.0,
        'avg_win_pct': round(sum(p['pnl_pct'] for p in winners if p.get('pnl_pct') is not None) / len(winners), 2) if winners else 0.0,
        'avg_loss_pct': round(sum(p['pnl_pct'] for p in losers if p.get('pnl_pct') is not None) / len(losers), 2) if losers else 0.0,
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0.0,
    }
    with open(DATA_DIR / 'closed.json', 'w') as f:
        json.dump({'closed_positions': positions, 'performance': perf}, f, indent=2, default=str)
    logger.info(f"Wrote closed.json: {len(positions)} closed trades")


def get_spy_info():
    try:
        spy = yf.Ticker('SPY')
        hist = spy.history(period='1y')
        if hist.empty:
            return {}
        df = calculate_indicators(hist)
        price = float(df['Close'].iloc[-1])
        sma200 = float(df['SMA_200'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None
        return {'price': round(price, 2), 'above_200sma': price > sma200, 'rsi': round(rsi, 1) if rsi else None}
    except Exception as e:
        logger.error(f'Error fetching SPY info: {e}')
        return {}


def main():
    logger.info('=' * 60)
    logger.info(' 8/21 EMA Screener - GitHub Actions Run ')
    logger.info('=' * 60)

    # Step 1: Fetch SPY and VIX first, before the main scan exhausts the rate limit
    spy_info = get_spy_info()
    vix_level = get_vix_level()
    logger.info(f"SPY: {spy_info}, VIX: {vix_level}")

    # Step 2: Dynamic position management
    position_actions = {}
    try:
        manager = DynamicPositionManager()
        position_actions = manager.process_all_positions()
        logger.info(f'Position manager: {len(position_actions)} positions evaluated')
    except Exception as e:
        logger.error(f'Error in position management: {e}')

    # Step 3: Run core screener (saves screener_history.csv, returns DataFrames)
    df_new_run, df_closed_run, df_history = run_screener()

    current_date = datetime.now().strftime('%Y-%m-%d')
    df_new = df_history[(df_history['Status'] == 'Active') & (df_history['Entry Date'] == current_date)].copy()
    df_existing = df_history[(df_history['Status'] == 'Active') & (df_history['Entry Date'] != current_date)].copy()

    # Step 4: Insider data
    insider_data = {}
    active_tickers = df_history[df_history['Status'] == 'Active']['Ticker'].unique().tolist()
    load_insider_cache()
    new_tickers = df_new['Ticker'].tolist() if not df_new.empty else []
    tickers_to_fetch = new_tickers  # fetch for new only; use cache for existing
    if tickers_to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(get_insider_shares_purchased, t): t for t in tickers_to_fetch}
            for fut in concurrent.futures.as_completed(futures):
                t = futures[fut]
                insider_data[t] = fut.result() or 0
        save_insider_cache()

    # Step 4: ML scoring
    df_screener_export = pd.concat([df_new, df_existing], ignore_index=True) if (not df_new.empty or not df_existing.empty) else None
    try:
        scorer = TradingConfidenceScorer()
        insider_with_actions = {**insider_data, '__position_actions__': position_actions}
        scorer.evaluate_active_positions(insider_data=insider_with_actions, screener_data=df_screener_export)
        logger.info('ML scoring complete')
    except Exception as e:
        logger.error(f'Error in ML scoring: {e}')

    # Step 6: Write JSON files
    write_screener_json(df_new, df_existing, spy_info, vix_level, insider_data)
    write_closed_json(df_history)
    logger.info('All JSON files written successfully')


if __name__ == '__main__':
    main()
