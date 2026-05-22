import { useMemo } from 'react';
import { AgGridReact } from 'ag-grid-react';
import { ColDef, ValueFormatterParams, CellClassParams } from 'ag-grid-community';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import { ClosedPosition } from '../types';
import { formatDate } from '../utils/api';

interface ClosedGridProps {
  positions: ClosedPosition[];
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

export function ClosedGrid({ positions }: ClosedGridProps) {
  const columnDefs = useMemo<ColDef<ClosedPosition>[]>(() => [
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
      headerName: 'Exit Date',
      field: 'exit_date',
      width: 110,
      valueFormatter: (params: ValueFormatterParams) => formatDate(params.value as string),
      sort: 'desc',
    },
    {
      headerName: 'Entry $',
      field: 'entry_price',
      width: 85,
      valueFormatter: nullSafe((v) => `$${v.toFixed(2)}`),
    },
    {
      headerName: 'Exit $',
      field: 'exit_price',
      width: 85,
      valueFormatter: nullSafe((v) => `$${v.toFixed(2)}`),
    },
    {
      headerName: 'P&L %',
      field: 'pnl_pct',
      width: 85,
      valueFormatter: nullSafe((v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`),
      cellClass: (params: CellClassParams) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
    {
      headerName: 'P&L $',
      field: 'pnl_dollars',
      width: 90,
      valueFormatter: nullSafe((v) => {
        const prefix = v >= 0 ? '+$' : '-$';
        return `${prefix}${Math.abs(v).toFixed(2)}`;
      }),
      cellClass: (params: CellClassParams) =>
        params.value == null ? '' : params.value >= 0 ? 'cell-positive' : 'cell-negative',
    },
    {
      headerName: 'Days',
      field: 'hold_days',
      width: 65,
      valueFormatter: (params: ValueFormatterParams) =>
        params.value == null ? '--' : String(params.value),
    },
    {
      headerName: 'Exit Reason',
      field: 'exit_reason',
      width: 130,
    },
    {
      headerName: 'R/R',
      field: 'risk_reward',
      width: 70,
      valueFormatter: nullSafe((v) => v.toFixed(2)),
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
  ], []);

  const defaultColDef = useMemo<ColDef>(() => ({
    sortable: true,
    resizable: true,
  }), []);

  return (
    <div className="ag-theme-alpine grid-container">
      <AgGridReact
        rowData={positions}
        columnDefs={columnDefs}
        defaultColDef={defaultColDef}
        domLayout="autoHeight"
        animateRows={true}
      />
    </div>
  );
}
