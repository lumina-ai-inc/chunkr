import type { LiveShare, ShareView, InvestorInterest } from './liveShareApi';

// LocalStorage keys
const MOCK_LIVE_SHARES_KEY = 'orin_mock_live_shares';
const MOCK_SHARE_VIEWS_KEY = 'orin_mock_share_views';
const MOCK_INVESTOR_INTEREST_KEY = 'orin_mock_investor_interest';

// Default mock data
const DEFAULT_MOCK_LIVE_SHARES: LiveShare[] = [
  {
    id: 'live-share-001',
    deal_id: 'deal-003-mockdata', // Riverside Townhomes
    short_id: 'riverside2024',
    share_url: typeof window !== 'undefined' 
      ? `${window.location.origin}/share/riverside2024`
      : '/share/riverside2024',
    expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days from now
    view_count: 12,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
    is_expired: false,
  },
];

// Default mock share views
const DEFAULT_MOCK_SHARE_VIEWS: Record<string, ShareView[]> = {
  'live-share-001': [
    {
      id: 'view-001',
      live_share_id: 'live-share-001',
      viewed_at: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(), // 1 hour ago
      ip_address: '192.168.1.100',
      user_agent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    },
    {
      id: 'view-002',
      live_share_id: 'live-share-001',
      viewed_at: new Date(Date.now() - 3 * 60 * 60 * 1000).toISOString(), // 3 hours ago
      ip_address: '192.168.1.101',
      user_agent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    },
    {
      id: 'view-003',
      live_share_id: 'live-share-001',
      viewed_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
      ip_address: '192.168.1.102',
      user_agent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
    },
  ],
};

// Default mock investor interest
const DEFAULT_MOCK_INVESTOR_INTEREST: Record<string, InvestorInterest[]> = {
  'live-share-001': [
    {
      id: 'interest-001',
      live_share_id: 'live-share-001',
      name: 'Acme Capital Partners',
      amount: 500000,
      status: 'Interested',
      notes: 'Very interested in multi-family assets. Looking to close within 30 days.',
      created_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(), // 1 day ago
    },
    {
      id: 'interest-002',
      live_share_id: 'live-share-001',
      name: 'Smith Family Office',
      amount: 250000,
      status: 'Maybe',
      notes: 'Reviewing with investment committee. Will decide by end of week.',
      created_at: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(), // 12 hours ago
    },
    {
      id: 'interest-003',
      live_share_id: 'live-share-001',
      name: 'Johnson Real Estate Fund',
      amount: undefined,
      status: 'Passed',
      notes: 'Not the right fit for our portfolio strategy.',
      created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(), // 2 days ago
    },
  ],
};

// Load from localStorage with fallback to defaults
const loadFromLocalStorage = <T>(key: string, defaultValue: T): T => {
  try {
    if (typeof window === 'undefined') return defaultValue;
    const stored = localStorage.getItem(key);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (e) {
    console.error(`Failed to load ${key} from localStorage:`, e);
  }
  return defaultValue;
};

// Initialize from localStorage
export const MOCK_LIVE_SHARES: LiveShare[] = loadFromLocalStorage(
  MOCK_LIVE_SHARES_KEY,
  DEFAULT_MOCK_LIVE_SHARES
);
export const MOCK_SHARE_VIEWS: Record<string, ShareView[]> = loadFromLocalStorage(
  MOCK_SHARE_VIEWS_KEY,
  DEFAULT_MOCK_SHARE_VIEWS
);
export const MOCK_INVESTOR_INTEREST: Record<string, InvestorInterest[]> = loadFromLocalStorage(
  MOCK_INVESTOR_INTEREST_KEY,
  DEFAULT_MOCK_INVESTOR_INTEREST
);

// Save to localStorage
export const saveMockLiveShareData = () => {
  try {
    if (typeof window === 'undefined') return;
    localStorage.setItem(MOCK_LIVE_SHARES_KEY, JSON.stringify(MOCK_LIVE_SHARES));
    localStorage.setItem(MOCK_SHARE_VIEWS_KEY, JSON.stringify(MOCK_SHARE_VIEWS));
    localStorage.setItem(MOCK_INVESTOR_INTEREST_KEY, JSON.stringify(MOCK_INVESTOR_INTEREST));
  } catch (e) {
    console.error('Failed to save mock live share data to localStorage:', e);
  }
};

// Helper function to get mock deal data for public share page
export const getMockDealForShare = (dealId: string) => {
  if (dealId === 'deal-003-mockdata') {
    return {
      deal_id: 'deal-003-mockdata',
      deal_name: 'Riverside Townhomes',
      property_name: 'Riverside Townhomes',
      address: '123 Riverside Drive, Austin, TX 78701',
      purchase_price: 1200000,
      noi: 70000,
      summary: `Riverside Townhomes presents an attractive investment opportunity in a well-established residential community. The property consists of 10 townhome units with strong occupancy (90%) and healthy cash flow. With a DSCR of 1.75x and NOI of $70,000, the property demonstrates solid fundamentals suitable for institutional and accredited investors seeking stable income with moderate growth potential.

The property is located in the growing Austin metro area with a strong job market and increasing demand for quality rental housing. Recent construction (2018) minimizes near-term capital expenditure needs, making this an ideal investment for those seeking immediate cash flow.`,
      documents: [
        {
          id: 'doc-001',
          file_name: 'Rent Roll - Q4 2024.pdf',
          page_count: 5,
        },
        {
          id: 'doc-002',
          file_name: 'Financial Statements - 2024.xlsx',
          page_count: 12,
        },
        {
          id: 'doc-003',
          file_name: 'Property Inspection Report.pdf',
          page_count: 28,
        },
      ],
    };
  }
  return null;
};
