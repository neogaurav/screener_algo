import { useState } from 'react';
import { useScreener } from './hooks/useScreener';
import { Summary } from './components/Summary';
import { Header } from './components/Header';
import { SetupGrid } from './components/SetupGrid';
import { ClosedGrid } from './components/ClosedGrid';
import { MLSummaryTab } from './components/MLSummary';
import { TabType } from './types';

function App() {
  const { screener, closed, mlSummary, loading, error, refresh, lastRefresh } = useScreener();
  const [activeTab, setActiveTab] = useState<TabType>('all-open');

  const renderContent = () => {
    if (error) {
      return (
        <div className="error-message">
          <h3>Error loading data</h3>
          <p>{error}</p>
          <p className="hint">
            Make sure the JSON files exist at the expected paths in the screener_algo GitHub repository.
          </p>
          <button onClick={refresh}>Retry</button>
        </div>
      );
    }

    switch (activeTab) {
      case 'all-open':
        return <SetupGrid setups={[...(screener?.new_setups ?? []), ...(screener?.existing_setups ?? [])]} />;
      case 'new-setups':
        return <SetupGrid setups={screener?.new_setups ?? []} />;
      case 'existing-setups':
        return <SetupGrid setups={screener?.existing_setups ?? []} />;
      case 'closed':
        return <ClosedGrid positions={closed?.closed_positions ?? []} />;
      case 'ml-summary':
        return <MLSummaryTab mlSummary={mlSummary} />;
      default:
        return null;
    }
  };

  return (
    <div className="app">
      <Summary
        screener={screener}
        closed={closed}
        lastRefresh={lastRefresh}
        onRefresh={refresh}
        loading={loading}
      />

      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        screener={screener}
        closed={closed}
        mlSummary={mlSummary}
      />

      <main className="main-content">
        {loading && !screener ? (
          <div className="loading">Loading...</div>
        ) : (
          renderContent()
        )}
      </main>
    </div>
  );
}

export default App;
