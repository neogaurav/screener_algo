"""
Dynamic Position Management System

This module re-evaluates active positions daily and makes intelligent exit decisions:
1. Early exit if setup deteriorates (broken EMAs, bearish structure)
2. Trailing stops for profitable positions
3. Target adjustments based on momentum
4. Risk management based on market volatility
"""

from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import logging
from screener import POSITION_SIZE

logger = logging.getLogger(__name__)


class DynamicPositionManager:
    def __init__(self, history_file=str(Path(__file__).parent / 'data' / 'screener_history.csv')):
        self.history_file = history_file

    def evaluate_position(self, ticker, entry_date, entry_price, stop_loss, fib_target):
        """
        Evaluate a single position and determine if action is needed.

        Returns:
            dict with keys: 'action', 'reason', 'exit_price', 'new_stop', 'new_target'
            action can be: 'HOLD', 'EXIT', 'TRAIL_STOP', 'EXTEND_TARGET'
        """
        try:
            # Fetch recent data (need enough for 200 SMA calculation)
            stock = yf.Ticker(ticker)
            df = stock.history(period='1y')

            if df.empty or len(df) < 50:
                return {'action': 'HOLD', 'reason': 'Insufficient data'}

            current_price = df['Close'].iloc[-1]
            current_high = df['High'].iloc[-1]
            current_low = df['Low'].iloc[-1]

            # Calculate EMAs
            df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
            df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()

            ema_8 = df['EMA_8'].iloc[-1]
            ema_21 = df['EMA_21'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]

            # Calculate P&L
            pnl_pct = ((current_price - entry_price) / entry_price) * 100

            # Get market context (SPY)
            spy = yf.Ticker('SPY')
            spy_df = spy.history(period='1mo')
            spy_trend = 1 if spy_df['Close'].iloc[-1] > spy_df['Close'].iloc[-20] else -1

            # === DECISION LOGIC ===

            # 1. CHECK FOR EARLY EXIT CONDITIONS (Setup Deterioration)
            if self._should_exit_early(df, current_price, entry_price, ema_8, ema_21, sma_50, sma_200, pnl_pct):
                reason = self._get_exit_reason(df, current_price, ema_8, ema_21, sma_50, pnl_pct)
                return {
                    'action': 'EXIT',
                    'reason': reason,
                    'exit_price': current_price,
                    'new_stop': None,
                    'new_target': None
                }

            # 2. CHECK FOR TRAILING STOP (Profitable Position)
            if pnl_pct > 5.0:  # Position is at least 5% profitable
                trailing_stop = self._calculate_trailing_stop(
                    current_price, entry_price, ema_8, ema_21, pnl_pct
                )

                if trailing_stop > stop_loss:  # Only raise the stop, never lower it
                    return {
                        'action': 'TRAIL_STOP',
                        'reason': f'Trailing stop to lock in profit ({pnl_pct:.1f}%)',
                        'exit_price': None,
                        'new_stop': trailing_stop,
                        'new_target': fib_target
                    }

            # 3. CHECK FOR TARGET EXTENSION (Strong Momentum)
            if self._should_extend_target(df, current_price, fib_target, pnl_pct, spy_trend):
                new_target = self._calculate_extended_target(current_price, fib_target)
                return {
                    'action': 'EXTEND_TARGET',
                    'reason': 'Strong momentum, extending target',
                    'exit_price': None,
                    'new_stop': stop_loss,
                    'new_target': new_target
                }

            # 4. DEFAULT: HOLD
            return {
                'action': 'HOLD',
                'reason': 'Setup intact, holding position',
                'exit_price': None,
                'new_stop': None,
                'new_target': None
            }

        except Exception as e:
            logger.error(f"Error evaluating {ticker}: {e}")
            return {'action': 'HOLD', 'reason': f'Error: {str(e)}'}

    def _should_exit_early(self, df, current_price, entry_price, ema_8, ema_21, sma_50, sma_200, pnl_pct):
        """Determine if position should be exited early due to setup deterioration"""

        # 1. Price broke below 21 EMA with momentum
        if current_price < ema_21 * 0.98:  # 2% below 21 EMA
            return True

        # 2. EMAs broke down (8 EMA crossed below 21 EMA)
        if ema_8 < ema_21:
            return True

        # 3. Price broke below 50 SMA (major support)
        if current_price < sma_50 * 0.97:
            return True

        # 4. Losing position approaching stop loss and showing weakness
        if pnl_pct < -3.0:  # Down 3%+
            # Check for recent bearish candles
            last_5_candles = df.iloc[-5:]
            bearish_count = sum(last_5_candles['Close'] < last_5_candles['Open'])
            if bearish_count >= 4:  # 4 out of 5 bearish candles
                return True

        # 5. Long holding period with no progress (>30 days, <3% gain)
        days_held = len(df) - df.index.get_loc(df[df['Close'] >= entry_price].index[0] if len(df[df['Close'] >= entry_price]) > 0 else df.index[0])
        if days_held > 30 and pnl_pct < 3.0:
            return True

        return False

    def _get_exit_reason(self, df, current_price, ema_8, ema_21, sma_50, pnl_pct):
        """Get specific reason for early exit"""
        if current_price < ema_21 * 0.98:
            return "Broke below 21 EMA"
        elif ema_8 < ema_21:
            return "EMA death cross (8<21)"
        elif current_price < sma_50 * 0.97:
            return "Broke below 50 SMA"
        elif pnl_pct < -3.0:
            return "Persistent weakness"
        else:
            return "Setup deteriorated"

    def _calculate_trailing_stop(self, current_price, entry_price, ema_8, ema_21, pnl_pct):
        """Calculate appropriate trailing stop based on profit level"""

        # Strategy: Use EMAs as dynamic support, but lock in more profit as position grows

        if pnl_pct >= 15.0:  # Large profit
            # Trail at 75% of profit (lock in most gains)
            return entry_price + (current_price - entry_price) * 0.75

        elif pnl_pct >= 10.0:  # Good profit
            # Trail at 50% of profit or 8 EMA, whichever is higher
            fifty_pct_stop = entry_price + (current_price - entry_price) * 0.50
            return max(fifty_pct_stop, ema_8 * 0.98)

        elif pnl_pct >= 5.0:  # Modest profit
            # Trail at breakeven or 21 EMA
            return max(entry_price * 1.01, ema_21 * 0.99)

        else:
            # Not enough profit to trail
            return None

    def _should_extend_target(self, df, current_price, fib_target, pnl_pct, spy_trend):
        """Determine if target should be extended due to strong momentum"""

        # Only consider if position is profitable and near target
        if pnl_pct < 8.0:
            return False

        distance_to_target = ((fib_target - current_price) / current_price) * 100
        if distance_to_target > 5.0:  # Still far from target
            return False

        # Check for strong momentum indicators
        df['RSI'] = self._calculate_rsi(df['Close'], 14)
        rsi = df['RSI'].iloc[-1]

        # Strong uptrend conditions
        last_10 = df.iloc[-10:]
        higher_highs = (last_10['High'].iloc[-1] > last_10['High'].iloc[-5] and
                       last_10['High'].iloc[-5] > last_10['High'].iloc[-10])

        # Extension criteria: Strong RSI, higher highs, bullish market
        if rsi > 60 and higher_highs and spy_trend > 0:
            return True

        return False

    def _calculate_extended_target(self, current_price, original_target):
        """Calculate new extended target"""
        # Extend by 1.382 Fibonacci ratio from current position
        remaining_distance = original_target - current_price
        extension = remaining_distance * 1.382
        return current_price + extension

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def process_all_positions(self):
        """Process all active positions and update history file

        Returns:
            dict: Dictionary mapping tickers to their position actions
                  Format: {ticker: {'action': str, 'reason': str}}
        """
        try:
            df_history = pd.read_csv(self.history_file)
            active_positions = df_history[df_history['Status'] == 'Active'].copy()

            if active_positions.empty:
                print("No active positions to evaluate")
                return {}

            print("\n" + "="*60)
            print(" DYNAMIC POSITION MANAGEMENT ")
            print("="*60)

            actions_taken = {
                'EXIT': [],
                'TRAIL_STOP': [],
                'EXTEND_TARGET': [],
                'HOLD': []
            }

            # Store position actions for export
            position_actions = {}

            for idx, position in active_positions.iterrows():
                ticker = position['Ticker']
                entry_date = position['Entry Date']
                entry_price = position['Entry Price']
                stop_loss = position['Stop Loss']
                fib_target = position['Fib Target']

                print(f"\nEvaluating {ticker}...")

                decision = self.evaluate_position(
                    ticker, entry_date, entry_price, stop_loss, fib_target
                )

                action = decision['action']
                reason = decision['reason']

                print(f"  Action: {action} - {reason}")

                # Store for export
                position_actions[ticker] = {'action': action, 'reason': reason}

                # Update history based on decision
                mask = (df_history['Ticker'] == ticker) & (df_history['Status'] == 'Active')

                if action == 'EXIT':
                    df_history.loc[mask, 'Status'] = 'Closed'
                    df_history.loc[mask, 'Exit Date'] = datetime.now().strftime('%Y-%m-%d')
                    df_history.loc[mask, 'Exit Price'] = decision['exit_price']
                    df_history.loc[mask, 'Reason'] = reason

                    # Calculate PnL
                    pnl_pct = ((decision['exit_price'] - entry_price) / entry_price) * 100
                    df_history.loc[mask, 'PnL %'] = round(pnl_pct, 2)
                    df_history.loc[mask, 'PnL $'] = round((pnl_pct / 100) * POSITION_SIZE, 2)

                    actions_taken['EXIT'].append((ticker, reason))

                elif action == 'TRAIL_STOP':
                    df_history.loc[mask, 'Stop Loss'] = decision['new_stop']
                    actions_taken['TRAIL_STOP'].append((ticker, reason))

                elif action == 'EXTEND_TARGET':
                    df_history.loc[mask, 'Fib Target'] = decision['new_target']
                    actions_taken['EXTEND_TARGET'].append((ticker, reason))

                else:  # HOLD
                    actions_taken['HOLD'].append(ticker)

            # Save updated history
            df_history.to_csv(self.history_file, index=False)

            # Summary
            print("\n" + "="*60)
            print(" SUMMARY ")
            print("="*60)
            print(f"Positions Exited: {len(actions_taken['EXIT'])}")
            for ticker, reason in actions_taken['EXIT']:
                print(f"  - {ticker}: {reason}")

            print(f"\nTrailing Stops Applied: {len(actions_taken['TRAIL_STOP'])}")
            for ticker, reason in actions_taken['TRAIL_STOP']:
                print(f"  - {ticker}: {reason}")

            print(f"\nTargets Extended: {len(actions_taken['EXTEND_TARGET'])}")
            for ticker, reason in actions_taken['EXTEND_TARGET']:
                print(f"  - {ticker}: {reason}")

            print(f"\nPositions Held: {len(actions_taken['HOLD'])}")

            return position_actions

        except Exception as e:
            logger.error(f"Error processing positions: {e}")
            raise


def main():
    """Standalone execution for dynamic position management"""
    manager = DynamicPositionManager()
    manager.process_all_positions()


if __name__ == "__main__":
    main()
