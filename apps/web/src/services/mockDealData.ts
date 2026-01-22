// Mock data for testing deal flow UI without backend
// TODO: Remove this file once backend is fully operational

import { DealResponse, DocumentResponse, FactResponse } from "./dealApi";

// LocalStorage keys
const MOCK_DEALS_KEY = 'orin_mock_deals';
const MOCK_DOCUMENTS_KEY = 'orin_mock_documents';

// Default mock deals
const DEFAULT_MOCK_DEALS: DealResponse[] = [
  {
    deal_id: "deal-002-mockdata",
    user_id: "mock-user-001",
    deal_name: "Downtown Commercial Property",
    status: "processing_documents",
    deal_type: "rental_income",
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    metadata: {},
    document_count: 2,
    fact_count: 7,
  },
  {
    deal_id: "deal-003-mockdata",
    user_id: "mock-user-001",
    deal_name: "Riverside Townhomes",
    status: "ready_for_underwriting",
    deal_type: "rental_income",
    created_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    metadata: {},
    document_count: 4,
    fact_count: 7,
  },
];

// Default mock documents
const DEFAULT_MOCK_DOCUMENTS: DocumentResponse[] = [
  // Documents for Downtown Commercial Property (deal-002-mockdata)
  {
    document_id: "doc-downtown-001",
    deal_id: "deal-002-mockdata",
    file_name: "Commercial_Lease_Summary.pdf",
    document_type: "rent_roll",
    status: "processed",
    storage_location: "/mock-documents/Commercial_Lease_Summary.pdf",
    page_count: 4,
    created_at: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    extracted_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    document_id: "doc-downtown-002",
    deal_id: "deal-002-mockdata",
    file_name: "Commercial_PL_2024.pdf",
    document_type: "profit_and_loss",
    status: "processed",
    storage_location: "/mock-documents/Commercial_PL_2024.pdf",
    page_count: 6,
    created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
    extracted_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
  },
  // Documents for Riverside Townhomes (deal-003-mockdata)
  {
    document_id: "doc-001-mockdata",
    deal_id: "deal-003-mockdata",
    file_name: "Rent_Roll_2024.pdf",
    document_type: "rent_roll",
    status: "processed",
    storage_location: "/mock-documents/Rent_Roll_2024.pdf",
    page_count: 3,
    created_at: new Date(Date.now() - 10 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    extracted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    document_id: "doc-002-mockdata",
    deal_id: "deal-003-mockdata",
    file_name: "T12_Profit_Loss.pdf",
    document_type: "profit_and_loss",
    status: "processed",
    storage_location: "/mock-documents/T12_Profit_Loss.pdf",
    page_count: 5,
    created_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    extracted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    document_id: "doc-003-mockdata",
    deal_id: "deal-003-mockdata",
    file_name: "Mortgage_Statement_Jan2024.pdf",
    document_type: "mortgage_statement",
    status: "processed",
    storage_location: "/mock-documents/Mortgage_Statement_Jan2024.pdf",
    page_count: 2,
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    extracted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    document_id: "doc-004-mockdata",
    deal_id: "deal-003-mockdata",
    file_name: "Property_Tax_2023.pdf",
    document_type: "tax_document",
    status: "processed",
    storage_location: "/mock-documents/Property_Tax_2023.pdf",
    page_count: 1,
    created_at: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
    updated_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    extracted_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

export const MOCK_FACTS: FactResponse[] = [
  // Facts for Downtown Commercial Property (deal-002-mockdata)
  {
    fact_id: "fact-downtown-001",
    document_id: "doc-downtown-001",
    deal_id: "deal-002-mockdata",
    fact_type: "financial",
    label: "Unit Count",
    value: "12",
    unit: undefined,
    source_citation: {
      document: "Commercial_Lease_Summary.pdf",
      page: 1,
      line: "Total Commercial Units: 12",
    },
    status: "pending_approval",
    confidence_score: 0.93,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-downtown-002",
    document_id: "doc-downtown-001",
    deal_id: "deal-002-mockdata",
    fact_type: "financial",
    label: "Occupancy %",
    value: "91.7",
    unit: "%",
    source_citation: {
      document: "Commercial_Lease_Summary.pdf",
      page: 1,
      line: "Current Occupancy: 91.7%",
    },
    status: "pending_approval",
    confidence_score: 0.90,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-downtown-003",
    document_id: "doc-downtown-002",
    deal_id: "deal-002-mockdata",
    fact_type: "financial",
    label: "Gross Scheduled Rent",
    value: "480000",
    unit: "USD/year",
    source_citation: {
      document: "Commercial_PL_2024.pdf",
      page: 2,
      line: "Annual Gross Rent: $480,000",
    },
    status: "pending_approval",
    confidence_score: 0.97,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-downtown-004",
    document_id: "doc-downtown-002",
    deal_id: "deal-002-mockdata",
    fact_type: "financial",
    label: "Operating Expenses",
    value: "160000",
    unit: "USD/year",
    source_citation: {
      document: "Commercial_PL_2024.pdf",
      page: 3,
      line: "Total Operating Expenses: $160,000",
    },
    status: "pending_approval",
    confidence_score: 0.85,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-downtown-005",
    document_id: "doc-downtown-002",
    deal_id: "deal-002-mockdata",
    fact_type: "financial",
    label: "Collected Rent",
    value: "440000",
    unit: "USD/year",
    source_citation: {
      document: "Commercial_PL_2024.pdf",
      page: 1,
      line: "Actual Collections (T-12): $440,000",
    },
    status: "pending_approval",
    confidence_score: 0.92,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-downtown-006",
    document_id: "doc-downtown-002",
    deal_id: "deal-002-mockdata",
    fact_type: "debt",
    label: "Debt Service",
    value: "",
    unit: "USD/year",
    source_citation: {
      document: "Commercial_PL_2024.pdf",
      page: 0,
      line: "",
    },
    status: "missing",
    confidence_score: undefined,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-downtown-007",
    document_id: "doc-downtown-001",
    deal_id: "deal-002-mockdata",
    fact_type: "financial",
    label: "Net Operating Income",
    value: "320000",
    unit: "USD/year",
    source_citation: {
      document: "Commercial_PL_2024.pdf",
      page: 4,
      line: "NOI: $320,000",
    },
    status: "pending_approval",
    confidence_score: 0.75,
    approved_at: undefined,
    approved_by: undefined,
    locked: false,
    created_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
  },
  // Facts for Riverside Townhomes (deal-003-mockdata)
  {
    fact_id: "fact-001-mockdata",
    document_id: "doc-001-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "financial",
    label: "Unit Count",
    value: "24",
    unit: undefined,
    source_citation: {
      document: "Rent_Roll_2024.pdf",
      page: 1,
      line: "Total Units: 24",
    },
    status: "approved",
    confidence_score: 0.95,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-002-mockdata",
    document_id: "doc-001-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "financial",
    label: "Occupancy %",
    value: "87.5",
    unit: "%",
    source_citation: {
      document: "Rent_Roll_2024.pdf",
      page: 1,
      line: "Current Occupancy: 87.5%",
    },
    status: "approved",
    confidence_score: 0.92,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-003-mockdata",
    document_id: "doc-001-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "financial",
    label: "Gross Scheduled Rent",
    value: "120000",
    unit: "USD/year",
    source_citation: {
      document: "Rent_Roll_2024.pdf",
      page: 2,
      line: "Annual Gross Rent: $120,000",
    },
    status: "approved",
    confidence_score: 0.98,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 8 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-004-mockdata",
    document_id: "doc-002-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "financial",
    label: "Operating Expenses",
    value: "50000",
    unit: "USD/year",
    source_citation: {
      document: "T12_Profit_Loss.pdf",
      page: 3,
      line: "Total Operating Expenses: $50,000",
    },
    status: "approved",
    confidence_score: 0.88,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-005-mockdata",
    document_id: "doc-002-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "financial",
    label: "Collected Rent",
    value: "110000",
    unit: "USD/year",
    source_citation: {
      document: "T12_Profit_Loss.pdf",
      page: 1,
      line: "Actual Collections (T-12): $110,000",
    },
    status: "approved",
    confidence_score: 0.96,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-006-mockdata",
    document_id: "doc-003-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "debt",
    label: "Debt Service",
    value: "40000",
    unit: "USD/year",
    source_citation: {
      document: "Mortgage_Statement_Jan2024.pdf",
      page: 1,
      line: "Annual Debt Service: $40,000",
    },
    status: "approved",
    confidence_score: 0.94,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 6 * 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    fact_id: "fact-007-mockdata",
    document_id: "doc-004-mockdata",
    deal_id: "deal-003-mockdata",
    fact_type: "tax",
    label: "Property Value",
    value: "1200000",
    unit: "USD",
    source_citation: {
      document: "Property_Tax_2023.pdf",
      page: 1,
      line: "Assessed Value: $1,200,000",
    },
    status: "approved",
    confidence_score: 0.85,
    approved_at: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
    approved_by: "mock-user-001",
    locked: true,
    created_at: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
  },
];

// These two pre-existing deals have pre-loaded documents for demo
const PREEXISTING_MOCK_DEALS = ['deal-002-mockdata', 'deal-003-mockdata'];

export const isMockDeal = (dealId?: string): boolean => {
  // ANY deal with "mockdata" suffix uses mock mode for deal management
  // (but file uploads still go through REAL OCR processing)
  return dealId ? dealId.includes("mockdata") : false;
};

export const isPreexistingMockDeal = (dealId?: string): boolean => {
  // Check if it's one of the two pre-existing deals with pre-loaded documents
  return dealId ? PREEXISTING_MOCK_DEALS.includes(dealId) : false;
};

export const generateMockDealId = (): string => {
  // NEW deals get "mockdata" suffix so they use mock mode for deal management
  // but file uploads still use REAL backend for OCR processing
  return `deal-${Date.now()}-mockdata`;
};

// Load from localStorage with fallback to defaults
const loadFromLocalStorage = <T>(key: string, defaultValue: T): T => {
  try {
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
export const MOCK_DEALS: DealResponse[] = loadFromLocalStorage(MOCK_DEALS_KEY, DEFAULT_MOCK_DEALS);
export const MOCK_DOCUMENTS: DocumentResponse[] = loadFromLocalStorage(MOCK_DOCUMENTS_KEY, DEFAULT_MOCK_DOCUMENTS);

// Save to localStorage
export const saveMockData = () => {
  try {
    localStorage.setItem(MOCK_DEALS_KEY, JSON.stringify(MOCK_DEALS));
    localStorage.setItem(MOCK_DOCUMENTS_KEY, JSON.stringify(MOCK_DOCUMENTS));
  } catch (e) {
    console.error('Failed to save mock data to localStorage:', e);
  }
};

export const createMockDeal = (dealName: string, dealType: string = 'rental_income'): DealResponse => {
  const newDeal: DealResponse = {
    deal_id: generateMockDealId(),
    user_id: "mock-user-001",
    deal_name: dealName,
    status: "draft",
    deal_type: dealType as 'rental_income' | 'value_add',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    metadata: {},
    document_count: 0,
    fact_count: 0,
  };
  MOCK_DEALS.push(newDeal);
  saveMockData();  // Auto-save
  return newDeal;
};
