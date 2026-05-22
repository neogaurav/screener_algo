export interface Setup {
  ticker: string;
  entry_date: string;
  entry_price: number;
  current_price: number;
  stop_loss: number;
  fib_target: number;
  risk_reward: number;
  market_cap: string;
  sector: string;
  rsi: number;
  adx: number;
  rs_vs_spy: number;
  ml_confidence: string;
  ml_win_prob: number;
  ml_expected_pnl: number;
  insider_net: number;
  hold_days: number;
}

export interface ClosedPosition {
  ticker: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  pnl_pct: number;
  pnl_dollars: number;
  hold_days: number;
  exit_reason: string;
  market_cap: string;
  sector: string;
  ml_confidence: string;
  risk_reward: number;
}

export interface MLScore {
  ticker: string;
  entry_date: string;
  current_price: number;
  entry_price: number;
  ml_confidence: string;
  ml_win_prob: number;
  ml_expected_pnl: number;
  rsi: number;
  adx: number;
  rs_vs_spy: number;
  market_cap: string;
  sector: string;
}

export interface ConfidenceSummary {
  count: number;
  avg_win_rate: number;
  avg_pnl_pct: number;
}

export interface Performance {
  total_trades: number;
  winners: number;
  losers: number;
  win_rate: number;
  total_pnl_dollars: number;
  avg_pnl_pct: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  profit_factor: number;
}

export interface ScreenerSummary {
  new_count: number;
  existing_count: number;
  total_active: number;
  spy_price: number;
  spy_above_200sma: boolean;
  spy_rsi: number;
  vix: number;
}

export interface ScreenerData {
  new_setups: Setup[];
  existing_setups: Setup[];
  summary: ScreenerSummary;
  last_updated: string;
}

export interface ClosedData {
  closed_positions: ClosedPosition[];
  performance: Performance;
}

export interface MLSummaryData {
  active_with_scores: MLScore[];
  summary_by_confidence: {
    HIGH: ConfidenceSummary;
    MEDIUM: ConfidenceSummary;
    LOW: ConfidenceSummary;
  };
  model_info: {
    last_trained: string;
    is_trained: boolean;
  };
  last_updated: string;
}

export type TabType = 'new-setups' | 'existing-setups' | 'closed' | 'ml-summary';
