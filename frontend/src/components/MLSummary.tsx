import { useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, ValueFormatterParams } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import { MLSummaryData, MLScore, ConfidenceSummary } from '../types';
import { formatDate } from '../utils/api';

interface MLSummaryProps {
  mlSummary: MLSummaryData | null;
}

function nullSafe(fn: (v: number) => string) {
  return (params: ValueFormatterParams) => {
    if (params.value == null || (typeof params.value === 'number' && isNaN(params.value))) {
      return '--';
    }
    return fn(params.value as number);
  };
}

function ConfidenceBadge({ value }: { value: string }) {
  if (!value) return <span className="badge-na">N/A</span>;
  const conf = value.toUpperCase();
  if (conf === 'HIGH') return <span className="badge-high">HIGH</span>;
  if (conf === 'MEDIUM') return <span className="badge-medium">MEDIUM</span>;
  if (conf === 'LOW') return <span className="badge-low">LOW</span>;
  return <span className="badge-na">{value}</span>;
}

interface ConfidenceCardProps {
  label: string;
  data: ConfidenceSummary | undefined;
  badgeClass: string;
}

function ConfidenceCard({ label, data, badgeClass }: ConfidenceCardProps) {
  return (
    <div className="ml-card">
      <div style={{ marginBottom: '12px' }}>
        <span className={badgeClass}>{label}</span>
      </div>
      <div className="stat-row">
        <span>Active Setups:</span>
        <span className="value">{data?.count ?? '--'}</span>
      </div>
      <div className="stat-row">
        <span>Historical Win Rate:</span>
        <span className="value">{data != null ? `${data.avg_win_rate.toFixed(1)}%` : '--'}</span>
      </div>
      <div className="stat-row">
        <span>Avg P&amp;L %:</span>
        <span className={`value ${(data?.avg_pnl_pct ?? 0) >= 0 ? 'positive' : 'negative'}`}>
          {data != null ? `${data.avg_pnl_pct >= 0 ? '+' : ''}${data.avg_pnl_pct.toFixed(1)}%` : '--'}
        </span>
      </div>
    </div>
  );
}

export function MLSummaryTab({ mlSummary }: MLSummaryProps) {
  const columnDefs = useMemo<ColDef<MLScore>[]>(() => [
    {
      headerName: '#',
      valueGetter: (params) => params.node?.rowIndex != null ? params.node.rowIndex + 1 : '',
      width: 50,
      pinned: 'left',
      sortable: false,
    },
    {
      headerName: 'Ticker',
      field: 'ticker',
      width: 90,
      pinned: 'left',
      cellRenderer: (params: { value: string }) => (
        <a
          href={`https://finance.yahoo.com/quote/${params.value}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: '#60a5fa', textDecoration: 'none', fontWeight: 'bold' }}
        >
          {params.value}
        </a>
      ),
    },
    {
      headerName: 'Entry Date',
      field: 'entry_date',
      width: 110,
      valueFormatter: (params: ValueFormatterParams) => formatDate(params.value as string),
    },
    {
      headerName: 'ML Conf',
      field: 'ml_confidence',
      width: 100,
      cellRenderer: (params: { value: string }) => <ConfidenceBadge value={params.value} />,
      sort: 'asc',
      comparator: (a: string, b: string) => {
        const order: Record<string, number> = { HIGH: 0, MEDIUM: 1, LOW: 2 };
        return (order[a?.toUpperCase()] ?? 3) - (order[b?.toUpperCase()] ?? 3);
      },
    },
    {
      headerName: 'Win Prob %',
      field: 'ml_win_prob',
      width: 100,
      valueFormatter: nullSafe((v) => `${v.toFixed(1)}%`),
      sort: 'desc',
    },
    {
      headerName: 'Exp PnL %',
      field: 'ml_expected_pnl',
      width: 100,
      valueFormatter: nullSafe((v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`),
      cellClass: (params: { value: number | null }) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
    {
      headerName: 'RSI',
      field: 'rsi',
      width: 65,
      valueFormatter: nullSafe((v) => v.toFixed(1)),
    },
    {
      headerName: 'ADX',
      field: 'adx',
      width: 65,
      valueFormatter: nullSafe((v) => v.toFixed(1)),
    },
    {
      headerName: 'RS% vs SPY',
      field: 'rs_vs_spy',
      width: 100,
      valueFormatter: nullSafe((v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`),
      cellClass: (params: { value: number | null }) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
    {
      headerName: 'Market Cap',
      field: 'market_cap',
      width: 100,
    },
    {
      headerName: 'Sector',
      field: 'sector',
      width: 140,
    },
  ], []);

  const defaultColDef = useMemo<ColDef>(() => ({
    sortable: true,
    resizable: true,
  }), []);

  const confSummary = mlSummary?.summary_by_confidence;
  const modelInfo = mlSummary?.model_info;

  return (
    <div>
      {modelInfo && (
        <div className="model-info-bar">
          <span>
            Model Status:{' '}
            <strong style={{ color: modelInfo.is_trained ? 'var(--positive)' : 'var(--negative)' }}>
              {modelInfo.is_trained ? 'Trained' : 'Not Trained'}
            </strong>
          </span>
          {modelInfo.last_trained && (
            <span style={{ marginLeft: '20px' }}>
              Last Trained: <strong>{modelInfo.last_trained}</strong>
            </span>
          )}
        </div>
      )}

      <div className="ml-cards">
        <ConfidenceCard
          label="HIGH"
          data={confSummary?.HIGH}
          badgeClass="badge-high"
        />
        <ConfidenceCard
          label="MEDIUM"
          data={confSummary?.MEDIUM}
          badgeClass="badge-medium"
        />
        <ConfidenceCard
          label="LOW"
          data={confSummary?.LOW}
          badgeClass="badge-low"
        />
      </div>

      <h3 style={{ marginBottom: '12px', color: 'var(--text-secondary)', fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '1px' }}>
        Active Setups with ML Scores
      </h3>

      <div className="ag-theme-alpine grid-container">
        <AgGridReact
          rowData={mlSummary?.active_with_scores ?? []}
          columnDefs={columnDefs}
          defaultColDef={defaultColDef}
          domLayout="autoHeight"
          animateRows={true}
        />
      </div>
    </div>
  );
}
