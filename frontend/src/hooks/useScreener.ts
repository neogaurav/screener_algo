import { useState, useEffect, useCallback } from 'react';
import { ScreenerData, ClosedData, MLSummaryData } from '../types';
import { fetchScreener, fetchClosed, fetchMLSummary } from '../utils/api';

interface UseScreenerResult {
  screener: ScreenerData | null;
  closed: ClosedData | null;
  mlSummary: MLSummaryData | null;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  lastRefresh: Date | null;
}

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 minutes

export function useScreener(): UseScreenerResult {
  const [screener, setScreener] = useState<ScreenerData | null>(null);
  const [closed, setClosed] = useState<ClosedData | null>(null);
  const [mlSummary, setMLSummary] = useState<MLSummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<Date | null>(null);

  const loadData = useCallback(async () => {
    try {
      setError(null);

      const [screenerData, closedData, mlData] = await Promise.all([
        fetchScreener(),
        fetchClosed(),
        fetchMLSummary(),
      ]);

      setScreener(screenerData);
      setClosed(closedData);
      setMLSummary(mlData);
      setLastRefresh(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadData();
  }, [loadData]);

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const interval = setInterval(loadData, REFRESH_INTERVAL);
    return () => clearInterval(interval);
  }, [loadData]);

  const refresh = useCallback(async () => {
    setLoading(true);
    await loadData();
  }, [loadData]);

  return {
    screener,
    closed,
    mlSummary,
    loading,
    error,
    refresh,
    lastRefresh,
  };
}
