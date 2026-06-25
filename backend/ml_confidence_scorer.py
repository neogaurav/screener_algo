"""
Machine Learning Confidence Scoring Module for 8/21 EMA Strategy

This module trains ML models on historical data to predict:
1. Win/Loss probability
2. Expected P&L
3. Risk of hitting stop loss
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error, r2_score, roc_auc_score
import joblib
import warnings
from datetime import datetime
import os
import math
import json

warnings.filterwarnings('ignore')

class TradingConfidenceScorer:
    def __init__(self, history_file=str(Path(__file__).parent / 'data' / 'screener_history.csv')):
        """Initialize the ML confidence scorer"""
        self.history_file = history_file
        self.win_classifier = None
        self.pnl_regressor = None
        self.stop_classifier = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_dir = str(Path(__file__).parent / 'data' / 'ml_models')

        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def load_and_prepare_data(self):
        """Load historical data and prepare features"""
        df = pd.read_csv(self.history_file)

        # Filter to closed positions with P&L data
        df_closed = df[(df['Status'] == 'Closed') & df['PnL %'].notna()].copy()

        # FILTER: Use only recent high-quality data
        # Combine filters: recent + decent win rate to train on best patterns
        print(f"\nTotal closed trades: {len(df_closed)}")
        print(f"Overall win rate: {(df_closed['PnL %'] > 0).mean() * 100:.1f}%")
        print(f"Overall avg PnL: {df_closed['PnL %'].mean():.2f}%")

        # Recency handling: cap by age (keep a long lookback, never collapse to 30 days).
        # Newer trades are later up-weighted via recency sample weights (see below).
        if 'Exit Date' in df_closed.columns:
            df_closed['Exit Date'] = pd.to_datetime(df_closed['Exit Date'], errors='coerce')
            max_lookback = pd.Timestamp.now() - pd.Timedelta(days=548)  # ~18 months
            capped = df_closed[df_closed['Exit Date'] >= max_lookback].copy()
            if len(capped) >= 100:
                print(f"\nUsing last 18 months: {len(capped)} of {len(df_closed)} closed trades")
                print(f"  Win rate: {(capped['PnL %'] > 0).mean() * 100:.1f}%")
                df_closed = capped
            else:
                print(f"\nKeeping all history ({len(df_closed)} trades); <100 in last 18 months")

        # CRITICAL: Adjust model expectations based on actual performance
        # If historical performance is poor, model will be pessimistic (which is correct!)
        # To improve predictions, improve the trading strategy, not the model

        # Select ML features (technical indicators)
        feature_cols = [
            'RSI_Entry', 'ADX_Entry', 'DeMarker_Entry', 'DeMarker_Min_14d',
            'RS_vs_SPY', 'Volume_Ratio', 'Price_to_8EMA_%', 'Price_to_21EMA_%',
            'EMA_Stack_Gap_%', 'Pullback_Depth_%', 'Pullback_Days',
            'Risk/Reward', 'SPY_RSI', 'SPY_Trend', 'VIX_Level',
            # New signals
            'ATR_Pct', 'Bounce_Vol_Ratio', 'VIX_Regime', 'Sector_RS', 'Days_To_Earnings'
        ]

        # Filter to features that exist in columns
        available_features = [col for col in feature_cols if col in df_closed.columns]

        # IMPORTANT: Filter to only entries that have ML features populated
        # (entries created with ML-enhanced screener have RSI_Entry populated)
        ml_feature_indicator = 'RSI_Entry'  # Use this as indicator that ML features exist
        if ml_feature_indicator in df_closed.columns:
            df_with_ml = df_closed[df_closed[ml_feature_indicator].notna()].copy()
            print(f"Total closed positions: {len(df_closed)}")
            print(f"Closed positions with ML features: {len(df_with_ml)}")

            # FIXED: Clean corrupted RS_vs_SPY values (from old bug)
            if 'RS_vs_SPY' in df_with_ml.columns:
                # RS should be a percentage difference, typically -20% to +20%
                # Values > 100 are from the old division bug
                rs_values = pd.to_numeric(df_with_ml['RS_vs_SPY'], errors='coerce')
                bad_rs_mask = (rs_values.abs() > 100) | rs_values.isna()
                if bad_rs_mask.sum() > 0:
                    print(f"Filtering out {bad_rs_mask.sum()} trades with corrupted RS_vs_SPY values")
                    df_with_ml.loc[bad_rs_mask, 'RS_vs_SPY'] = 0  # Set to neutral

            df_closed = df_with_ml

        if len(df_closed) < 30:
            print(f"Warning: Only {len(df_closed)} closed positions with ML features. Need at least 30 for reliable training.")
            return None

        # Create target variables
        df_closed['Win'] = (df_closed['PnL %'] > 0).astype(int)
        df_closed['Hit_Stop_Binary'] = df_closed['Hit Stop'].fillna(False).astype(int)

        # Add categorical encoding for Market Cap
        if 'Market Cap' in df_closed.columns:
            df_closed['MC_Large'] = (df_closed['Market Cap'] == 'Large').astype(int)
            df_closed['MC_Mid'] = (df_closed['Market Cap'] == 'Mid').astype(int)
            available_features.extend(['MC_Large', 'MC_Mid'])

        # Remove features with too many missing values (within ML-feature entries)
        feature_availability = {}
        for col in available_features:
            non_null_pct = df_closed[col].notna().sum() / len(df_closed)
            feature_availability[col] = non_null_pct

        # Keep features with at least 50% data availability
        self.feature_columns = [col for col, pct in feature_availability.items() if pct >= 0.5]

        if len(self.feature_columns) < 5:
            print(f"Warning: Only {len(self.feature_columns)} features have sufficient data.")
            print("Features with data:", self.feature_columns)
            return None

        # Prepare feature matrix
        X = df_closed[self.feature_columns].copy()

        # Fill missing values with median
        for col in self.feature_columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())

        # Prepare target variables
        y_win = df_closed['Win']
        y_pnl = df_closed['PnL %']
        y_stop = df_closed['Hit_Stop_Binary']

        # Recency sample weights: newer closed trades count more (exp decay, ~180d half-life)
        sample_weights = None
        if 'Exit Date' in df_closed.columns and df_closed['Exit Date'].notna().any():
            age_days = (pd.Timestamp.now() - df_closed['Exit Date']).dt.days
            age_days = age_days.fillna(age_days.median()).clip(lower=0)
            sample_weights = np.power(0.5, age_days / 180.0).to_numpy()

        return X, y_win, y_pnl, y_stop, df_closed, sample_weights

    def train_models(self):
        """Train the ML models"""
        print("\n" + "="*60)
        print(" TRAINING ML CONFIDENCE MODELS ")
        print("="*60)

        # Load and prepare data
        result = self.load_and_prepare_data()
        if result is None:
            print("Insufficient data for training. Need more closed trades with ML features.")
            return False

        X, y_win, y_pnl, y_stop, df_closed, sample_weights = result

        if sample_weights is None:
            sample_weights = np.ones(len(X))

        print(f"\nTraining on {len(X)} closed trades")
        print(f"Features used ({len(self.feature_columns)}): {', '.join(self.feature_columns)}")

        # --- Leakage-free evaluation: split FIRST, then scale on train only ---
        X_arr = X.to_numpy()
        rf_params = dict(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)

        (X_tr, X_te,
         yw_tr, yw_te,
         yp_tr, yp_te,
         ys_tr, ys_te,
         w_tr, w_te) = train_test_split(
            X_arr, y_win, y_pnl, y_stop, sample_weights,
            test_size=0.2, random_state=42
        )

        eval_scaler = StandardScaler()
        X_tr_s = eval_scaler.fit_transform(X_tr)
        X_te_s = eval_scaler.transform(X_te)

        # 1. Win/Loss Classifier
        print("\n1. Training Win/Loss Classifier...")
        eval_win = RandomForestClassifier(**rf_params)
        eval_win.fit(X_tr_s, yw_tr, sample_weight=w_tr)
        win_train_acc = eval_win.score(X_tr_s, yw_tr)
        win_test_acc = eval_win.score(X_te_s, yw_te)
        print(f"   Train Accuracy: {win_train_acc:.2%}")
        print(f"   Test Accuracy:  {win_test_acc:.2%}")
        print(f"   Train-Test Gap: {(win_train_acc - win_test_acc):.2%}")
        try:
            win_proba_te = eval_win.predict_proba(X_te_s)
            if win_proba_te.shape[1] > 1:
                print(f"   Test ROC-AUC:   {roc_auc_score(yw_te, win_proba_te[:, 1]):.3f}")
        except Exception as e:
            print(f"   (ROC-AUC unavailable: {e})")
        print("   Classification report (test):")
        print(classification_report(yw_te, eval_win.predict(X_te_s), zero_division=0))

        # Leakage-free cross-validation: scaler refit inside each fold via Pipeline
        n_splits = max(2, min(5, len(X_arr) // 20))
        cv_pipe = Pipeline([('scaler', StandardScaler()),
                            ('rf', RandomForestClassifier(**rf_params))])
        cv_scores = cross_val_score(cv_pipe, X_arr, y_win, cv=n_splits)
        print(f"   CV Accuracy ({n_splits}-fold): {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

        # 2. P&L Regressor
        print("\n2. Training P&L Regressor...")
        eval_pnl = RandomForestRegressor(**rf_params)
        eval_pnl.fit(X_tr_s, yp_tr, sample_weight=w_tr)
        pnl_pred = eval_pnl.predict(X_te_s)
        rmse = np.sqrt(mean_squared_error(yp_te, pnl_pred))
        test_r2 = r2_score(yp_te, pnl_pred)
        train_r2 = r2_score(yp_tr, eval_pnl.predict(X_tr_s))
        print(f"   Test RMSE: {rmse:.2f}%")
        print(f"   Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f} | Gap: {(train_r2 - test_r2):.3f}")

        # 3. Stop Loss Risk Classifier
        print("\n3. Training Stop Loss Risk Classifier...")
        eval_stop = RandomForestClassifier(**rf_params)
        eval_stop.fit(X_tr_s, ys_tr, sample_weight=w_tr)
        stop_train_acc = eval_stop.score(X_tr_s, ys_tr)
        stop_test_acc = eval_stop.score(X_te_s, ys_te)
        print(f"   Train Accuracy: {stop_train_acc:.2%}")
        print(f"   Test Accuracy:  {stop_test_acc:.2%}")
        print(f"   Train-Test Gap: {(stop_train_acc - stop_test_acc):.2%}")

        # --- Fit PRODUCTION models on ALL data (scaler + models), recency-weighted ---
        self.scaler = StandardScaler()
        X_full_s = self.scaler.fit_transform(X_arr)
        self.win_classifier = RandomForestClassifier(**rf_params)
        self.win_classifier.fit(X_full_s, y_win, sample_weight=sample_weights)
        self.pnl_regressor = RandomForestRegressor(**rf_params)
        self.pnl_regressor.fit(X_full_s, y_pnl, sample_weight=sample_weights)
        self.stop_classifier = RandomForestClassifier(**rf_params)
        self.stop_classifier.fit(X_full_s, y_stop, sample_weight=sample_weights)

        # Feature importance (from production win model)
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.win_classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n   Top 5 Most Important Features for Win/Loss:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   - {row['feature']}: {row['importance']:.3f}")

        # Summary statistics
        print("\n" + "="*60)
        print(" TRAINING DATA STATISTICS ")
        print("="*60)

        win_rate = y_win.mean()
        avg_win = df_closed[df_closed['Win'] == 1]['PnL %'].mean()
        avg_loss = df_closed[df_closed['Win'] == 0]['PnL %'].mean()
        stop_rate = y_stop.mean()

        print(f"\nHistorical Win Rate: {win_rate:.1%}")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Stop Loss Hit Rate: {stop_rate:.1%}")

        # Add warning if win rate is very low
        if win_rate < 0.30:
            print("\nWARNING: Training data has very low win rate (<30%)")
            print("   This likely reflects the OLD algorithm's performance.")
            print("   Models will improve as new data from the FIXED algorithm accumulates.")
            print("   Current confidence scores may be pessimistic.")

        # Save models
        self.save_models()

        return True

    def save_models(self):
        """Save trained models to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        joblib.dump(self.win_classifier, f"{self.model_dir}/win_classifier_{timestamp}.pkl")
        joblib.dump(self.pnl_regressor, f"{self.model_dir}/pnl_regressor_{timestamp}.pkl")
        joblib.dump(self.stop_classifier, f"{self.model_dir}/stop_classifier_{timestamp}.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/scaler_{timestamp}.pkl")
        joblib.dump(self.feature_columns, f"{self.model_dir}/features_{timestamp}.pkl")

        # Also save as latest
        joblib.dump(self.win_classifier, f"{self.model_dir}/win_classifier_latest.pkl")
        joblib.dump(self.pnl_regressor, f"{self.model_dir}/pnl_regressor_latest.pkl")
        joblib.dump(self.stop_classifier, f"{self.model_dir}/stop_classifier_latest.pkl")
        joblib.dump(self.scaler, f"{self.model_dir}/scaler_latest.pkl")
        joblib.dump(self.feature_columns, f"{self.model_dir}/features_latest.pkl")

        print(f"\nModels saved to {self.model_dir}/")

    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            self.win_classifier = joblib.load(f"{self.model_dir}/win_classifier_latest.pkl")
            self.pnl_regressor = joblib.load(f"{self.model_dir}/pnl_regressor_latest.pkl")
            self.stop_classifier = joblib.load(f"{self.model_dir}/stop_classifier_latest.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler_latest.pkl")
            self.feature_columns = joblib.load(f"{self.model_dir}/features_latest.pkl")
            return True
        except FileNotFoundError:
            print("No pre-trained models found. Please train models first.")
            return False

    def predict_confidence(self, features_dict):
        """
        Predict confidence scores for a new setup

        Args:
            features_dict: Dictionary with feature values for the setup

        Returns:
            Dictionary with confidence scores and predictions
        """
        if self.win_classifier is None:
            if not self.load_models():
                return None

        # Prepare feature vector
        feature_values = []
        for col in self.feature_columns:
            if col in features_dict:
                feature_values.append(features_dict[col])
            elif col == 'MC_Large':
                feature_values.append(1 if features_dict.get('Market Cap') == 'Large' else 0)
            elif col == 'MC_Mid':
                feature_values.append(1 if features_dict.get('Market Cap') == 'Mid' else 0)
            else:
                feature_values.append(0)  # Use 0 for missing features

        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Get predictions
        # Robust to single-class models: take P(label==1) via classes_, default 0 if unseen.
        def _prob_positive(clf, Xs):
            proba = clf.predict_proba(Xs)[0]
            classes = list(clf.classes_)
            return float(proba[classes.index(1)]) if 1 in classes else 0.0

        win_prob = _prob_positive(self.win_classifier, X_scaled)
        expected_pnl = self.pnl_regressor.predict(X_scaled)[0]
        stop_risk = _prob_positive(self.stop_classifier, X_scaled)

        # Confidence scoring based on win probability, expected PnL, and stop risk
        # A good trade needs: decent win prob + positive expected returns + manageable risk

        # Calculate composite score (0-100)
        # Weight: 40% win probability, 40% expected PnL, 20% stop risk
        win_score = min(win_prob * 100, 100)  # Scale 0-100
        pnl_score = min(max(expected_pnl * 5, 0), 100)  # Expected PnL: 0-20% maps to 0-100
        risk_score = max(100 - (stop_risk * 100), 0)  # Lower stop risk = higher score

        composite_score = (win_score * 0.4) + (pnl_score * 0.4) + (risk_score * 0.2)

        # Assign confidence based on composite score AND individual thresholds
        # HIGH: Strong across all metrics
        if composite_score >= 60 and win_prob > 0.35 and expected_pnl > 2.0 and stop_risk < 0.35:
            confidence = 'HIGH'
            confidence_emoji = '[H]'

        # MEDIUM: Good overall but not exceptional
        elif composite_score >= 40 and win_prob > 0.28 and expected_pnl > 0.5:
            confidence = 'MEDIUM'
            confidence_emoji = '[M]'

        # LOW: Below threshold or negative expected returns
        else:
            confidence = 'LOW'
            confidence_emoji = '[L]'

        return {
            'confidence': confidence,
            'confidence_emoji': confidence_emoji,
            'win_probability': round(win_prob * 100, 1),
            'expected_pnl': round(expected_pnl, 2),
            'stop_risk': round(stop_risk * 100, 1)
        }

    def _safe_val(self, val):
        """Convert NaN/inf to None for JSON serialization."""
        if val is None:
            return None
        try:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(val, 'item'):
            # numpy scalar
            val = val.item()
        try:
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
        except (TypeError, ValueError):
            pass
        return val

    def export_to_json(self, output_path):
        """
        Export ML summary to JSON for web frontend consumption.

        Writes a ml_summary.json file with:
        - active_with_scores: active positions with ML scores
        - summary_by_confidence: aggregated stats by confidence level
        - model_info: training metadata
        - last_updated: ISO timestamp

        Args:
            output_path: Path to write the JSON file
        """
        try:
            df_history = pd.read_csv(self.history_file)

            # Active positions with ML scores
            active_positions = df_history[df_history['Status'] == 'Active'].copy()
            active_with_scores = []
            for _, row in active_positions.iterrows():
                entry = {
                    'ticker': str(row.get('Ticker', '') or ''),
                    'entry_date': str(row.get('Entry Date', '') or ''),
                    'current_price': self._safe_val(row.get('Current Price')),
                    'entry_price': self._safe_val(row.get('Entry Price')),
                    'ml_confidence': str(row.get('ML_Confidence') or 'N/A'),
                    'ml_win_prob': self._safe_val(row.get('ML_Win_Prob')),
                    'ml_expected_pnl': self._safe_val(row.get('ML_Expected_PnL')),
                    'rsi': self._safe_val(row.get('RSI_Entry')),
                    'adx': self._safe_val(row.get('ADX_Entry')),
                    'rs_vs_spy': self._safe_val(row.get('RS_vs_SPY')),
                    'market_cap': str(row.get('Market Cap', '') or ''),
                    'sector': str(row.get('Sector', '') or ''),
                }
                active_with_scores.append(entry)

            # Summary by confidence from closed trades
            df_closed = df_history[df_history['Status'] == 'Closed'].copy()
            summary_by_confidence = {}
            for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
                if 'ML_Confidence' in df_closed.columns:
                    conf_group = df_closed[df_closed['ML_Confidence'] == conf_level]
                else:
                    conf_group = pd.DataFrame()

                if not conf_group.empty and 'PnL %' in conf_group.columns:
                    pnl_vals = pd.to_numeric(conf_group['PnL %'], errors='coerce').dropna()
                    win_count = (pnl_vals > 0).sum()
                    total_count = len(pnl_vals)
                    avg_win_rate = round(win_count / total_count * 100, 1) if total_count > 0 else 0.0
                    avg_pnl = round(float(pnl_vals.mean()), 2) if len(pnl_vals) > 0 else 0.0
                    summary_by_confidence[conf_level] = {
                        'count': int(total_count),
                        'avg_win_rate': avg_win_rate,
                        'avg_pnl_pct': avg_pnl,
                    }
                else:
                    summary_by_confidence[conf_level] = {
                        'count': 0,
                        'avg_win_rate': 0.0,
                        'avg_pnl_pct': 0.0,
                    }

            # Model info
            model_pkl = os.path.join(self.model_dir, 'win_classifier_latest.pkl')
            is_trained = os.path.exists(model_pkl)
            last_trained = None
            if is_trained:
                try:
                    mtime = os.path.getmtime(model_pkl)
                    last_trained = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    last_trained = None

            output = {
                'active_with_scores': active_with_scores,
                'summary_by_confidence': summary_by_confidence,
                'model_info': {
                    'last_trained': last_trained,
                    'is_trained': is_trained,
                },
                'last_updated': datetime.utcnow().isoformat() + 'Z',
            }

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2, default=str)

            print(f"\n[SUCCESS] ML summary exported to: {output_path}")

        except Exception as e:
            print(f"\n[WARNING] Could not export ML summary to JSON: {e}")

    def export_to_excel(self, df_ml_results, df_summary, df_screener=None, position_actions=None):
        """
        Export ML confidence results to Excel with multiple sheets.

        Args:
            df_ml_results: DataFrame with ML confidence scores for all positions
            df_summary: DataFrame with summary by confidence level
            df_screener: Optional DataFrame with screener results (new/existing setups)
            position_actions: Optional dict of position actions from dynamic position manager
                             Format: {ticker: {'action': str, 'reason': str}}
        """
        try:
            from datetime import datetime

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'ml_confidence_results_{timestamp}.xlsx'

            # Create Excel writer
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Sheet 1: ML Confidence Scores
                df_export = df_ml_results.copy()

                # Add position actions if provided
                if position_actions:
                    print(f"  -> Adding position management actions for {len(position_actions)} positions")
                    df_export['Position Action'] = df_export['Ticker'].map(
                        lambda t: position_actions.get(t, {}).get('action', 'N/A')
                    )
                    df_export['Action Reason'] = df_export['Ticker'].map(
                        lambda t: position_actions.get(t, {}).get('reason', 'N/A')
                    )

                # Remove helper columns
                if 'conf_order' in df_export.columns:
                    df_export = df_export.drop(columns=['conf_order'])
                if 'Status_Category' in df_export.columns:
                    df_export = df_export.drop(columns=['Status_Category'])

                df_export.to_excel(writer, sheet_name='ML Confidence Scores', index=False)

                # Sheet 2: Summary by Confidence
                df_summary.to_excel(writer, sheet_name='Summary', index=False)

                # Sheet 3: Screener Results (if provided)
                if df_screener is not None and not df_screener.empty:
                    df_screener_export = df_screener.copy()
                    # Remove helper columns
                    helper_cols = ['MC_Rank', 'ML_Conf_Order']
                    for col in helper_cols:
                        if col in df_screener_export.columns:
                            df_screener_export = df_screener_export.drop(columns=[col])

                    df_screener_export.to_excel(writer, sheet_name='Screener Results', index=False)

                # Sheet 4: Recent Closed Positions (last 30 days)
                try:
                    df_history = pd.read_csv(self.history_file)
                    df_closed = df_history[df_history['Status'] == 'Closed'].copy()

                    if not df_closed.empty:
                        # Filter to last 30 days
                        df_closed['Exit Date'] = pd.to_datetime(df_closed['Exit Date'], errors='coerce')
                        thirty_days_ago = pd.Timestamp.now() - pd.Timedelta(days=30)
                        df_recent_closed = df_closed[df_closed['Exit Date'] >= thirty_days_ago].copy()

                        if not df_recent_closed.empty:
                            # Calculate days held
                            df_recent_closed['Entry Date'] = pd.to_datetime(df_recent_closed['Entry Date'], errors='coerce')
                            df_recent_closed['Days Held'] = (df_recent_closed['Exit Date'] - df_recent_closed['Entry Date']).dt.days

                            # Categorize exit reasons
                            def categorize_exit(reason):
                                if pd.isna(reason):
                                    return 'Unknown'
                                reason_str = str(reason).lower()
                                if 'target' in reason_str or 'fib' in reason_str:
                                    return 'Hit Target'
                                elif 'stop' in reason_str:
                                    return 'Hit Stop'
                                elif 'broke' in reason_str or 'ema' in reason_str or 'weakness' in reason_str or 'deteriorat' in reason_str:
                                    return 'Early Exit (Setup Failed)'
                                elif 'poor r/r' in reason_str:
                                    return 'Poor R/R'
                                else:
                                    return 'Other'

                            df_recent_closed['Exit Category'] = df_recent_closed['Reason'].apply(categorize_exit)

                            # Select and order columns for display
                            closed_cols = [
                                'Ticker', 'Market Cap', 'Entry Date', 'Entry Price',
                                'Exit Date', 'Exit Price', 'Days Held',
                                'Fib Target', 'Stop Loss', 'Risk/Reward',
                                'PnL %', 'PnL $', 'Exit Category', 'Reason'
                            ]
                            # Filter to only existing columns
                            closed_cols = [col for col in closed_cols if col in df_recent_closed.columns]

                            # Sort by exit date (most recent first)
                            df_recent_closed = df_recent_closed.sort_values('Exit Date', ascending=False)

                            # Format dates for display
                            df_recent_closed['Entry Date'] = df_recent_closed['Entry Date'].dt.strftime('%Y-%m-%d')
                            df_recent_closed['Exit Date'] = df_recent_closed['Exit Date'].dt.strftime('%Y-%m-%d')

                            df_recent_closed[closed_cols].to_excel(writer, sheet_name='Recent Closed (30d)', index=False)

                            # Calculate summary stats for this period
                            total_trades = len(df_recent_closed)
                            winners = df_recent_closed[df_recent_closed['PnL %'] > 0]
                            losers = df_recent_closed[df_recent_closed['PnL %'] <= 0]

                            win_rate = (len(winners) / total_trades * 100) if total_trades > 0 else 0
                            avg_win = winners['PnL %'].mean() if len(winners) > 0 else 0
                            avg_loss = losers['PnL %'].mean() if len(losers) > 0 else 0
                            avg_days = df_recent_closed['Days Held'].mean()

                            print(f"  -> Added {len(df_recent_closed)} recently closed positions to Excel")
                            print(f"     Last 30 days: {len(winners)}W-{len(losers)}L ({win_rate:.1f}% win rate)")

                except Exception as e:
                    print(f"  -> Could not add closed positions sheet: {e}")

                # Sheet 5: ML Performance Tracking (skipped — ml_confidence_tracker not used)

                # Auto-adjust column widths
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width

            print(f"\n[SUCCESS] Results exported to: {filename}")

        except Exception as e:
            print(f"\n[WARNING] Could not export to Excel: {e}")
            print("  Make sure 'openpyxl' is installed: pip install openpyxl")

    def calculate_avg_days_to_target(self):
        """Calculate average days to reach target or stop from historical data"""
        try:
            df = pd.read_csv(self.history_file)
            closed = df[df['Status'] == 'Closed'].copy()

            if closed.empty:
                return {'avg_days_win': 15, 'avg_days_loss': 10}  # Default estimates

            # Calculate days held for closed positions
            closed['Entry Date'] = pd.to_datetime(closed['Entry Date'], errors='coerce')
            closed['Exit Date'] = pd.to_datetime(closed['Exit Date'], errors='coerce')
            closed['Days Held'] = (closed['Exit Date'] - closed['Entry Date']).dt.days

            # Filter to valid data
            valid_closed = closed[closed['Days Held'].notna() & (closed['Days Held'] > 0)]

            if valid_closed.empty:
                return {'avg_days_win': 15, 'avg_days_loss': 10}

            # Separate wins and losses
            winners = valid_closed[valid_closed['PnL %'] > 0]
            losers = valid_closed[valid_closed['PnL %'] <= 0]

            avg_days_win = int(winners['Days Held'].median()) if not winners.empty else 15
            avg_days_loss = int(losers['Days Held'].median()) if not losers.empty else 10

            return {
                'avg_days_win': avg_days_win,
                'avg_days_loss': avg_days_loss
            }
        except Exception as e:
            # Return defaults if calculation fails
            return {'avg_days_win': 15, 'avg_days_loss': 10}

    def evaluate_active_positions(self, insider_data=None, screener_data=None):
        """Evaluate all active positions and add confidence scores

        Args:
            insider_data: Optional dict of {ticker: shares} for insider buying data.
                         If not provided, will fetch it.
                         May contain special key '__position_actions__' with position management data.
            screener_data: Optional DataFrame with screener results to include in export.
        """
        import yfinance as yf
        import concurrent.futures
        from datetime import datetime

        df = pd.read_csv(self.history_file)
        active_positions = df[df['Status'] == 'Active'].copy()

        # Extract position actions if present in insider_data
        position_actions = None
        if insider_data and '__position_actions__' in insider_data:
            position_actions = insider_data['__position_actions__']
            # Remove it so it doesn't interfere with ticker lookups
            insider_data = {k: v for k, v in insider_data.items() if k != '__position_actions__'}

        # Filter to only positions with good Risk/Reward (>= 1.5)
        if 'Risk/Reward' in active_positions.columns:
            before_count = len(active_positions)
            active_positions = active_positions[active_positions['Risk/Reward'] >= 1.5].copy()
            filtered_count = before_count - len(active_positions)
            if filtered_count > 0:
                print(f"\nFiltered out {filtered_count} positions with R/R < 1.5")

        if active_positions.empty:
            print("No active positions to evaluate (after R/R filter).")
            return

        print("\n" + "="*60)
        print(" ML CONFIDENCE SCORES FOR ACTIVE POSITIONS ")
        print("="*60)

        # Calculate average days to target from historical data
        days_stats = self.calculate_avg_days_to_target()
        print(f"\nHistorical timing: Winners avg {days_stats['avg_days_win']} days, Losers avg {days_stats['avg_days_loss']} days")

        # Fetch insider buying data if not provided
        if insider_data is None:
            from screener import get_insider_shares_purchased
            tickers = active_positions['Ticker'].unique().tolist()
            insider_data = {}

            if tickers:
                print(f"\nFetching insider data for {len(tickers)} tickers...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_ticker = {executor.submit(get_insider_shares_purchased, t): t for t in tickers}
                    for future in concurrent.futures.as_completed(future_to_ticker):
                        t = future_to_ticker[future]
                        try:
                            insider_data[t] = future.result()
                        except:
                            insider_data[t] = 0
        else:
            print(f"\nUsing cached insider data for {len(insider_data)} tickers")
            # Debug: Show how many have actual buys
            total_with_buys = sum(1 for v in insider_data.values() if v > 0)
            print(f"  -> {total_with_buys} tickers have insider buying activity")

        results = []
        for idx, row in active_positions.iterrows():
            # DUAL CONFIDENCE SYSTEM:
            # 1. Entry Confidence (frozen) - for validation
            # 2. Current Confidence (dynamic) - for position management

            # Get ENTRY confidence (frozen at entry)
            if pd.notna(row.get('ML_Confidence')) and pd.notna(row.get('ML_Win_Prob')):
                # Use stored values from entry (NEVER changes)
                entry_prediction = {
                    'confidence': row['ML_Confidence'],
                    'win_probability': row['ML_Win_Prob'],
                    'expected_pnl': row.get('ML_Expected_PnL', 0),
                    'stop_risk': 0
                }
            else:
                # Fallback: Calculate for old positions
                features = row.to_dict()
                entry_prediction = self.predict_confidence(features)

            # Get CURRENT confidence (recalculated based on current conditions)
            # Fetch CURRENT technical indicators for recalculation
            current_features = row.to_dict()
            current_prediction = self.predict_confidence(current_features)

            # For backward compatibility, use entry prediction as primary
            prediction = entry_prediction

            # Fetch live current price
            try:
                ticker = yf.Ticker(row['Ticker'])
                hist = ticker.history(period='2d')
                if not hist.empty:
                    live_price = hist['Close'].iloc[-1]
                else:
                    live_price = row['Current Price']  # Fallback to stale price
            except:
                live_price = row['Current Price']  # Fallback to stale price

            if prediction:
                # Get insider buying data
                insider_shares = insider_data.get(row['Ticker'], 0)

                # Calculate days in trade
                try:
                    entry_date = pd.to_datetime(row['Entry Date'])
                    days_in_trade = (datetime.now() - entry_date).days
                except:
                    days_in_trade = 0

                # Estimate days to target based on win probability and historical data
                # Use weighted average: higher win prob = use winner avg, lower = use loser avg
                win_prob = prediction['win_probability'] / 100.0
                est_days_to_target = int(
                    (win_prob * days_stats['avg_days_win']) +
                    ((1 - win_prob) * days_stats['avg_days_loss'])
                )

                # Remaining days = estimated total - days already held
                remaining_days = max(est_days_to_target - days_in_trade, 1)

                # Format insider data: positive for net buying, negative for net selling
                insider_fmt = f"{insider_shares:+,}" if insider_shares != 0 else "0"

                # Calculate percentage to Fib target
                fib_target = row.get('Fib Target')
                to_target_pct = None
                if pd.notna(fib_target) and fib_target > 0 and live_price > 0:
                    to_target_pct = ((fib_target - live_price) / live_price) * 100

                # Determine confidence change indicator
                conf_change = ''
                if current_prediction:
                    entry_conf = prediction['confidence']
                    current_conf = current_prediction['confidence']

                    if entry_conf != current_conf:
                        conf_order_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
                        if conf_order_map.get(current_conf, 0) > conf_order_map.get(entry_conf, 0):
                            conf_change = '↑ Upgraded'
                        else:
                            conf_change = '↓ Downgraded'
                    else:
                        conf_change = '→ Stable'

                results.append({
                    'Ticker': row['Ticker'],
                    'Entry Date': row['Entry Date'],
                    'Days in Trade': days_in_trade,
                    'Entry Price': row['Entry Price'],
                    'Current Price': round(live_price, 2),
                    'Fib Target': row.get('Fib Target', None),
                    'To Target %': round(to_target_pct, 1) if to_target_pct is not None else None,
                    'Stop Loss': row.get('Stop Loss', None),
                    'Risk/Reward': row['Risk/Reward'],
                    'Est. Days to Target': remaining_days,
                    'Insider Net': insider_fmt,
                    # Entry Confidence (frozen)
                    'Entry Confidence': prediction['confidence'],
                    'Entry Win Prob %': prediction['win_probability'],
                    'Entry Exp P&L %': prediction['expected_pnl'],
                    # Current Confidence (dynamic)
                    'Current Confidence': current_prediction['confidence'] if current_prediction else prediction['confidence'],
                    'Current Win Prob %': current_prediction['win_probability'] if current_prediction else prediction['win_probability'],
                    'Current Exp P&L %': current_prediction['expected_pnl'] if current_prediction else prediction['expected_pnl'],
                    'Confidence Change': conf_change,
                    # Legacy columns for compatibility
                    'Confidence': prediction['confidence'],
                    'Win Prob %': prediction['win_probability'],
                    'Expected P&L %': prediction['expected_pnl'],
                    'Stop Risk %': prediction['stop_risk']
                })

        if results:
            df_results = pd.DataFrame(results)

            # Add "Has Entry" column to show if ticker is also in AO Saucer portfolio
            try:
                import os
                ao_tickers = set()
                if os.path.exists('ao_saucer_portfolio.csv'):
                    ao_df = pd.read_csv('ao_saucer_portfolio.csv')
                    if 'Ticker' in ao_df.columns:
                        ao_tickers = set(ao_df['Ticker'].unique())

                df_results['Has AO Entry'] = df_results['Ticker'].apply(lambda x: 'Yes' if x in ao_tickers else 'No')
            except Exception as e:
                print(f"Could not add AO entry info: {e}")

            # Sort by Entry Confidence (for validation grouping)
            confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            df_results['conf_order'] = df_results['Entry Confidence'].map(confidence_order)
            df_results = df_results.sort_values('conf_order')

            print("\n" + "="*80)
            print(" DUAL CONFIDENCE SYSTEM EXPLAINED ")
            print("="*80)
            print("Entry Confidence:   Frozen at entry - used for validation")
            print("Current Confidence: Recalculated daily - used for position management")
            print("Confidence Change:  Shows if position upgraded/downgraded/stable")
            print("="*80)

            # Select key columns for display
            display_cols = [
                'Ticker', 'Days in Trade', 'Entry Price', 'Current Price',
                'Fib Target', 'To Target %',
                'Entry Confidence', 'Current Confidence', 'Confidence Change',
                'Entry Win Prob %', 'Current Win Prob %',
                'Insider Net', 'Has AO Entry'
            ]
            display_cols = [col for col in display_cols if col in df_results.columns]

            print("\n" + df_results[display_cols].to_string(index=False))

            # Calculate P&L breakdown by confidence level
            def categorize_position(row):
                """Categorize position as winning/losing/new"""
                entry_price = row['Entry Price']
                current_price = row['Current Price']

                if abs(current_price - entry_price) < 0.01:
                    return 'new'
                elif current_price > entry_price:
                    return 'winning'
                else:
                    return 'losing'

            df_results['Status_Category'] = df_results.apply(categorize_position, axis=1)

            # Summary by ENTRY confidence level with win/loss breakdown
            # (Entry Confidence is used for validation - frozen at entry)
            summary_data = []

            for conf_level in ['HIGH', 'MEDIUM', 'LOW']:
                conf_positions = df_results[df_results['Entry Confidence'] == conf_level]
                total = len(conf_positions)

                if total > 0:
                    winning = len(conf_positions[conf_positions['Status_Category'] == 'winning'])
                    losing = len(conf_positions[conf_positions['Status_Category'] == 'losing'])
                    new = len(conf_positions[conf_positions['Status_Category'] == 'new'])

                    winning_pct = (winning / total * 100)
                    losing_pct = (losing / total * 100)
                    new_pct = (new / total * 100)

                    summary_data.append({
                        'Confidence': conf_level,
                        'Total': total,
                        'Winning': winning,
                        'Winning %': f"{winning_pct:.1f}%",
                        'Losing': losing,
                        'Losing %': f"{losing_pct:.1f}%",
                        'New': new,
                        'New %': f"{new_pct:.1f}%"
                    })

            # Overall summary
            total_positions = len(df_results)
            total_winning = len(df_results[df_results['Status_Category'] == 'winning'])
            total_losing = len(df_results[df_results['Status_Category'] == 'losing'])
            total_new = len(df_results[df_results['Status_Category'] == 'new'])

            total_winning_pct = (total_winning / total_positions * 100) if total_positions > 0 else 0
            total_losing_pct = (total_losing / total_positions * 100) if total_positions > 0 else 0
            total_new_pct = (total_new / total_positions * 100) if total_positions > 0 else 0

            summary_data.append({
                'Confidence': 'TOTAL',
                'Total': total_positions,
                'Winning': total_winning,
                'Winning %': f"{total_winning_pct:.1f}%",
                'Losing': total_losing,
                'Losing %': f"{total_losing_pct:.1f}%",
                'New': total_new,
                'New %': f"{total_new_pct:.1f}%"
            })

            df_summary = pd.DataFrame(summary_data)

            print(f"\n" + "="*80)
            print(" SUMMARY BY ENTRY CONFIDENCE LEVEL ")
            print(" (Grouped by frozen entry predictions for validation) ")
            print("="*80)
            print(df_summary.to_string(index=False))


            # Export to JSON (replaces Excel export for web deployment)
            self.export_to_json(Path(__file__).parent / 'data' / 'ml_summary.json')

            return df_results

def main():
    """Main function to train models and evaluate positions"""
    scorer = TradingConfidenceScorer()

    # Train models
    if scorer.train_models():
        # Evaluate active positions
        scorer.evaluate_active_positions()
    else:
        print("\nFailed to train models. Please ensure you have enough historical data with ML features.")
        print("Run the enhanced screener (screener.py) to capture ML features for new trades.")

if __name__ == "__main__":
    main()
