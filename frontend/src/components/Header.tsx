import { TabType, ScreenerData, ClosedData, MLSummaryData } from '../types';

interface HeaderProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  screener: ScreenerData | null;
  closed: ClosedData | null;
  mlSummary: MLSummaryData | null;
}

export function Header({ activeTab, onTabChange, screener, closed, mlSummary }: HeaderProps) {
  const newCount = screener?.new_setups.length ?? 0;
  const existingCount = screener?.existing_setups.length ?? 0;
  const closedCount = closed?.closed_positions.length ?? 0;
  const mlCount = mlSummary?.active_with_scores.length ?? 0;

  const tabs: { id: TabType; label: string; count: number }[] = [
    { id: 'all-open', label: 'All Open', count: newCount + existingCount },
    { id: 'new-setups', label: 'New Setups', count: newCount },
    { id: 'existing-setups', label: 'Existing Setups', count: existingCount },
    { id: 'closed', label: 'Closed Positions', count: closedCount },
    { id: 'ml-summary', label: 'ML Summary', count: mlCount },
  ];

  return (
    <nav className="tab-header">
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
          <span className="tab-count">[{tab.count}]</span>
        </button>
      ))}
    </nav>
  );
}
