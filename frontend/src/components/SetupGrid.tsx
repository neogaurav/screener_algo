import { useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, ValueFormatterParams, CellClassParams } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import { Setup } from '../types';
import { formatDate } from '../utils/api';

interface SetupGridProps {
  setups: Setup[];
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

export function SetupGrid({ setups }: SetupGridProps) {
  const columnDefs = useMemo<ColDef<Setup>[]>(() => [
    {
      headerName: '#',
      valueGetter: (params) => params.node?.rowIndex != null ? params.node.rowIndex + 1 : '',
      width: 50,
      pinned: 'left',
      sortable: false,
      filter: false,
      floatingFilter: false,
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
      headerName: 'AO',
      field: 'has_ao_entry',
      width: 60,
      pinned: 'left',
      filter: false,
      floatingFilter: false,
      cellRenderer: (params: { value: boolean | null }) =>
        params.value ? <span style={{ color: '#4ade80', fontWeight: 'bold', fontSize: '16px' }}>✓</span> : <span style={{ color: '#374151' }}>—</span>,
      headerTooltip: 'Also has an active AO Saucer entry',
    },
    {
      headerName: 'Entry Date',
      field: 'entry_date',
      width: 110,
      valueFormatter: (params: ValueFormatterParams) => formatDate(params.value as string),
    },
    {
      headerName: 'Entry $',
      field: 'entry_price',
      width: 90,
      valueFormatter: nullSafe((v) => `$${v.toFixed(2)}`),
    },
    {
      headerName: 'Current $',
      field: 'current_price',
      width: 90,
      valueFormatter: nullSafe((v) => `$${v.toFixed(2)}`),
      cellClass: (params: CellClassParams<Setup>) => {
        if (params.data == null) return '';
        const diff = (params.data.current_price ?? 0) - (params.data.entry_price ?? 0);
        return diff >= 0 ? 'cell-positive' : 'cell-negative';
      },
    },
    {
      headerName: 'Stop $',
      field: 'stop_loss',
      width: 85,
      valueFormatter: nullSafe((v) => `$${v.toFixed(2)}`),
    },
    {
      headerName: 'Target $',
      field: 'fib_target',
      width: 90,
      valueFormatter: nullSafe((v) => `$${v.toFixed(2)}`),
    },
    {
      headerName: 'Dist to Target',
      width: 115,
      valueGetter: (params) => {
        const cur = params.data?.current_price;
        const tgt = params.data?.fib_target;
        if (cur == null || tgt == null || cur === 0) return null;
        return ((tgt - cur) / cur) * 100;
      },
      valueFormatter: nullSafe((v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`),
      cellClass: (params: CellClassParams) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
    {
      headerName: 'R/R',
      field: 'risk_reward',
      width: 70,
      valueFormatter: nullSafe((v) => v.toFixed(2)),
    },
    {
      headerName: 'Days',
      field: 'hold_days',
      width: 65,
      valueFormatter: (params: ValueFormatterParams) =>
        params.value == null ? '--' : String(params.value),
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
      cellClass: (params: CellClassParams) =>
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
    {
      headerName: 'ML Conf',
      field: 'ml_confidence',
      width: 100,
      cellRenderer: (params: { value: string }) => <ConfidenceBadge value={params.value} />,
    },
    {
      headerName: 'ML Win%',
      field: 'ml_win_prob',
      width: 90,
      valueFormatter: nullSafe((v) => `${v.toFixed(1)}%`),
    },
    {
      headerName: 'ML Exp PnL%',
      field: 'ml_expected_pnl',
      width: 105,
      valueFormatter: nullSafe((v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`),
      cellClass: (params: CellClassParams) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
    {
      headerName: 'Insider Net',
      field: 'insider_net',
      width: 105,
      valueFormatter: (params: ValueFormatterParams) => {
        const v = params.value as number | null | undefined;
        if (v == null || isNaN(v as number)) return '--';
        const abs = Math.abs(v).toLocaleString('en-US');
        return v >= 0 ? `+${abs}` : `-${abs}`;
      },
      cellClass: (params: CellClassParams) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
  ], []);

  const defaultColDef = useMemo<ColDef>(() => ({
    sortable: true,
    resizable: true,
    filter: true,
    floatingFilter: true,
  }), []);

  return (
    <div className="ag-theme-alpine grid-container">
      <AgGridReact
        rowData={setups}
        columnDefs={columnDefs}
        defaultColDef={defaultColDef}
        domLayout="autoHeight"
        animateRows={true}
      />
    </div>
  );
}
