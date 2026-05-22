import { ScreenerData, ClosedData, MLSummaryData } from '../types';

// Configure these for your repository
const REPO_OWNER = 'neogaurav';
const REPO_NAME = 'screener_algo';

const BASE_URL = `https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/main/backend/data`;

export async function fetchScreener(): Promise<ScreenerData> {
  const url = `${BASE_URL}/screener.json?t=${Date.now()}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch screener data: ${response.status}`);
  }

  return response.json();
}

export async function fetchClosed(): Promise<ClosedData> {
  const url = `${BASE_URL}/closed.json?t=${Date.now()}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch closed positions: ${response.status}`);
  }

  return response.json();
}

export async function fetchMLSummary(): Promise<MLSummaryData> {
  const url = `${BASE_URL}/ml_summary.json?t=${Date.now()}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Failed to fetch ML summary: ${response.status}`);
  }

  return response.json();
}

export function formatCurrency(value: number): string {
  const prefix = value >= 0 ? '+$' : '-$';
  return `${prefix}${Math.abs(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function formatPercent(value: number): string {
  const prefix = value >= 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}%`;
}

export function formatDate(dateStr: string | null | undefined): string {
  if (!dateStr) return '--';
  const date = new Date(dateStr);
  if (isNaN(date.getTime())) return '--';
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}

export function formatDateTime(isoStr: string | null | undefined): string {
  if (!isoStr) return 'Never';
  const date = new Date(isoStr);
  if (isNaN(date.getTime())) return '--';
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZoneName: 'short'
  });
}

// GitHub API for triggering workflows
const GITHUB_API_URL = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/actions/workflows/screener.yml/dispatches`;
const TOKEN_KEY = 'github_pat_token';

export function getStoredToken(): string | null {
  return sessionStorage.getItem(TOKEN_KEY);
}

export function storeToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

export async function triggerScannerWorkflow(token: string): Promise<{ success: boolean; message: string }> {
  try {
    const response = await fetch(GITHUB_API_URL, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        ref: 'main'
      })
    });

    if (response.status === 204) {
      storeToken(token);
      return { success: true, message: 'Scanner workflow triggered! Check Actions tab for progress.' };
    } else if (response.status === 401) {
      clearToken();
      return { success: false, message: 'Invalid token. Please check your PAT.' };
    } else if (response.status === 404) {
      return { success: false, message: 'Workflow not found. Check repo settings.' };
    } else {
      const error = await response.json().catch(() => ({}));
      return { success: false, message: `Failed: ${(error as { message?: string }).message || response.statusText}` };
    }
  } catch (error) {
    return { success: false, message: `Network error: ${error instanceof Error ? error.message : 'Unknown'}` };
  }
}
