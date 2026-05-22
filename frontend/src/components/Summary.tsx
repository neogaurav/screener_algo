import { useState } from 'react';
import { ScreenerData, ClosedData } from '../types';
import { formatDateTime, getStoredToken, triggerScannerWorkflow } from '../utils/api';

interface SummaryProps {
  screener: ScreenerData | null;
  closed: ClosedData | null;
  lastRefresh: Date | null;
  onRefresh: () => void;
  loading: boolean;
}

export function Summary({ screener, closed, lastRefresh, onRefresh, loading }: SummaryProps) {
  const [triggerStatus, setTriggerStatus] = useState<{ type: 'success' | 'error' | null; message: string }>({ type: null, message: '' });
  const [isTriggering, setIsTriggering] = useState(false);

  const summary = screener?.summary;
  const perf = closed?.performance;

  const handleTriggerScan = async () => {
    let token = getStoredToken();

    if (!token) {
      token = window.prompt(
        'Enter your GitHub Personal Access Token (PAT):\n\n' +
        'To create one:\n' +
        '1. Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)\n' +
        '2. Generate new token with "repo" and "workflow" scopes\n\n' +
        'Token will be stored in session only (cleared on tab close):'
      );

      if (!token) {
        return;
      }
    }

    setIsTriggering(true);
    setTriggerStatus({ type: null, message: '' });

    const result = await triggerScannerWorkflow(token);

    setTriggerStatus({
      type: result.success ? 'success' : 'error',
      message: result.message,
    });
    setIsTriggering(false);

    if (result.success) {
      setTimeout(() => {
        setTriggerStatus({ type: null, message: '' });
      }, 5000);
    }
  };

  const vixLevel = summary?.vix;
  const vixLabel = vixLevel != null
    ? vixLevel < 15 ? 'Low' : vixLevel < 25 ? 'Moderate' : 'High'
    : null;

  return (
    <div className="summary">
      <div className="summary-header">
        <h1>8/21 EMA Screener Dashboard</h1>
        <div className="summary-meta">
          <span className="last-update">
            Last scan: {formatDateTime(screener?.last_updated || null)}
          </span>
          <span className={`spy-status ${summary?.spy_above_200sma ? 'bullish' : 'bearish'}`}>
            SPY: ${summary?.spy_price?.toFixed(2) || '--'}
            {' '}({summary?.spy_above_200sma ? 'Bullish' : 'Bearish'})
          </span>
          {vixLevel != null && (
            <span className="vix-badge">
              VIX: {vixLevel.toFixed(1)} ({vixLabel})
            </span>
          )}
          <div className="button-group">
            <button onClick={onRefresh} disabled={loading} className="refresh-btn">
              {loading ? 'Loading...' : 'Refresh'}
            </button>
            <button
              onClick={handleTriggerScan}
              disabled={isTriggering}
              className="trigger-btn"
              title="Manually trigger the scanner workflow"
            >
              {isTriggering ? 'Triggering...' : 'Run Scanner'}
            </button>
          </div>
        </div>
      </div>

      {triggerStatus.type && (
        <div className={`trigger-status ${triggerStatus.type}`}>
          {triggerStatus.message}
          {triggerStatus.type === 'error' && (
            <button
              className="retry-token-btn"
              onClick={() => {
                sessionStorage.removeItem('github_pat_token');
                setTriggerStatus({ type: null, message: '' });
                handleTriggerScan();
              }}
            >
              Try different token
            </button>
          )}
        </div>
      )}

      <div className="summary-cards">
        <div className="summary-card">
          <h3>Active Setups</h3>
          <div className="card-section">
            <h4>Today's Count</h4>
            <div className="stat-row">
              <span>New Setups:</span>
              <span className="value">{summary?.new_count ?? 0}</span>
            </div>
            <div className="stat-row">
              <span>Existing Positions:</span>
              <span className="value">{summary?.existing_count ?? 0}</span>
            </div>
            <div className="stat-row">
              <span>Total Active:</span>
              <span className="value">{summary?.total_active ?? 0}</span>
            </div>
          </div>
          <div className="card-section">
            <h4>Market Context</h4>
            <div className="stat-row">
              <span>SPY Price:</span>
              <span className="value">${summary?.spy_price?.toFixed(2) ?? '--'}</span>
            </div>
            <div className="stat-row">
              <span>SPY RSI:</span>
              <span className="value">{summary?.spy_rsi?.toFixed(1) ?? '--'}</span>
            </div>
            <div className="stat-row">
              <span>VIX:</span>
              <span className="value">{summary?.vix?.toFixed(1) ?? '--'}</span>
            </div>
          </div>
        </div>

        <div className="summary-card">
          <h3>Closed Performance</h3>
          <div className="card-section">
            <h4>Trade Results</h4>
            <div className="stat-row">
              <span>Total Trades:</span>
              <span className="value">{perf?.total_trades ?? 0}</span>
            </div>
            <div className="stat-row">
              <span>Winners / Losers:</span>
              <span className="value">{perf?.winners ?? 0} / {perf?.losers ?? 0}</span>
            </div>
            <div className="stat-row">
              <span>Win Rate:</span>
              <span className="value">{(perf?.win_rate ?? 0).toFixed(1)}%</span>
            </div>
          </div>
          <div className="card-section">
            <h4>P&amp;L Stats</h4>
            <div className="stat-row">
              <span>Total P&amp;L:</span>
              <span className={`value ${(perf?.total_pnl_dollars ?? 0) >= 0 ? 'positive' : 'negative'}`}>
                {(perf?.total_pnl_dollars ?? 0) >= 0 ? '+' : '-'}${Math.abs(perf?.total_pnl_dollars ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            </div>
            <div className="stat-row">
              <span>Avg Win / Loss:</span>
              <span className="value">
                <span className="positive">+{(perf?.avg_win_pct ?? 0).toFixed(1)}%</span>
                {' / '}
                <span className="negative">{(perf?.avg_loss_pct ?? 0).toFixed(1)}%</span>
              </span>
            </div>
            <div className="stat-row">
              <span>Profit Factor:</span>
              <span className="value">{(perf?.profit_factor ?? 0).toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="client-refresh">
        Client refresh: {lastRefresh?.toLocaleTimeString() || '--'}
      </div>
    </div>
  );
}
