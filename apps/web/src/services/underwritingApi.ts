import axiosInstance from "./axios.config";
import { getDealFacts, FactResponse } from "./dealApi";

export interface UnderwritingInput {
  unit_count?: number;
  occupancy_rate?: number;
  gross_scheduled_rent?: number;
  collected_rent: number;
  operating_expenses: number;
  debt_service?: number;
  property_value?: number;
  mortgage_balance?: number;
  interest_rate?: number;
}

export interface CalculationStep {
  metric: string;
  formula: string;
  inputs: [string, number][];
  result: number;
  sources: string[];
}

export interface UnderwritingResult {
  noi: number;
  dscr?: number;
  cash_flow_after_debt?: number;
  cap_rate?: number;
  ltv?: number;
  gross_rent_multiplier?: number;
  audit_trail: CalculationStep[];
  warnings: string[];
}

export interface StressTestInput {
  base_result: UnderwritingResult;
  occupancy_adjustment?: number;
  rent_adjustment?: number;
  expense_adjustment?: number;
  interest_rate_adjustment?: number;
}

export interface StressTestComparison {
  noi_change: number;
  noi_change_pct: number;
  dscr_change?: number;
  cash_flow_change?: number;
}

export interface StressTestResult {
  stressed_noi: number;
  stressed_dscr?: number;
  stressed_cash_flow?: number;
  comparison: StressTestComparison;
}

const USE_MOCK_DATA = true; // Set to false to use real API

// Mock underwriting data
const MOCK_UNDERWRITING_DATA: Record<string, UnderwritingResult> = {
  "deal-002-mockdata": {
    noi: 320000, // Gross Scheduled Rent (480,000) - Operating Expenses (160,000)
    dscr: 1.60, // NOI (320,000) / Debt Service (200,000)
    cash_flow_after_debt: 120000, // NOI (320,000) - Debt Service (200,000)
    cap_rate: undefined,
    ltv: undefined,
    gross_rent_multiplier: undefined,
    audit_trail: [
      {
        metric: "Net Operating Income (NOI)",
        formula: "Gross Scheduled Rent - Operating Expenses",
        inputs: [
          ["Gross Scheduled Rent", 480000],
          ["Operating Expenses", 160000],
        ],
        result: 320000,
        sources: ["fact-downtown-003"],
      },
      {
        metric: "Debt Service Coverage Ratio (DSCR)",
        formula: "Net Operating Income (NOI) / Annual Debt Service",
        inputs: [
          ["Net Operating Income (NOI)", 320000],
          ["Annual Debt Service", 200000],
        ],
        result: 1.60,
        sources: [],
      },
      {
        metric: "Cash Flow After Debt Service",
        formula: "Net Operating Income (NOI) - Annual Debt Service",
        inputs: [
          ["Net Operating Income (NOI)", 320000],
          ["Annual Debt Service", 200000],
        ],
        result: 120000,
        sources: [],
      },
    ],
    warnings: [
      "Commercial property occupancy at 91.7% - monitor tenant retention",
    ],
  },
  "deal-003-mockdata": {
    noi: 70000, // Gross Scheduled Rent (120,000) - Operating Expenses (50,000)
    dscr: 1.75, // NOI (70,000) / Debt Service (40,000)
    cash_flow_after_debt: 30000, // NOI (70,000) - Debt Service (40,000)
    cap_rate: undefined,
    ltv: undefined,
    gross_rent_multiplier: undefined,
    audit_trail: [
      {
        metric: "Net Operating Income (NOI)",
        formula: "Gross Scheduled Rent - Operating Expenses",
        inputs: [
          ["Gross Scheduled Rent", 120000],
          ["Operating Expenses", 50000],
        ],
        result: 70000,
        sources: ["fact-003-mockdata", "fact-004-mockdata"],
      },
      {
        metric: "Debt Service Coverage Ratio (DSCR)",
        formula: "NOI / Annual Debt Service",
        inputs: [
          ["NOI", 70000],
          ["Annual Debt Service", 40000],
        ],
        result: 1.75,
        sources: ["fact-006-mockdata"],
      },
      {
        metric: "Cash Flow After Debt Service",
        formula: "NOI - Annual Debt Service",
        inputs: [
          ["NOI", 70000],
          ["Annual Debt Service", 40000],
        ],
        result: 30000,
        sources: ["fact-006-mockdata"],
      },
    ],
    warnings: [
      "Management fee of 5.8% is slightly above typical range (3-5%)",
    ],
  },
};

// Helper function to extract numeric value from fact
const extractNumericValue = (fact: FactResponse): number | null => {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:extractNumericValue',message:'Extracting numeric value from fact',data:{factId:fact.fact_id,label:fact.label,value:fact.value,unit:fact.unit},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
  // #endregion
  
  // Remove currency symbols, commas, and whitespace
  const cleaned = fact.value.replace(/[$,\s]/g, '');
  const numValue = parseFloat(cleaned);
  
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:extractNumericValue',message:'Numeric value extracted',data:{factId:fact.fact_id,originalValue:fact.value,cleanedValue:cleaned,parsedValue:numValue,isNaN:isNaN(numValue)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
  // #endregion
  
  return isNaN(numValue) ? null : numValue;
};

// Calculate underwriting from facts dynamically
const calculateFromFacts = (facts: FactResponse[]): UnderwritingResult | null => {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Starting calculation from facts',data:{factsCount:facts.length,factLabels:facts.map(f=>f.label)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
  // #endregion
  
  // Find facts by label (case-insensitive, flexible matching)
  const findFact = (labelPatterns: string[]): FactResponse | null => {
    const lowerFacts = facts.map(f => ({ ...f, labelLower: f.label.toLowerCase() }));
    for (const pattern of labelPatterns) {
      const fact = lowerFacts.find(f => f.labelLower.includes(pattern.toLowerCase()));
      if (fact) return fact;
    }
    return null;
  };

  // Extract key financial values
  const grossRentFact = findFact(["Gross Rent", "Gross Scheduled Rent", "Collected Rent", "Rent"]);
  const operatingExpensesFact = findFact(["Operating Expenses", "Operating Expense", "Expenses"]);
  const loanAmountFact = findFact(["Loan Amount", "Loan", "Mortgage Balance"]);
  const interestRateFact = findFact(["Interest Rate", "Interest"]);
  const loanTermFact = findFact(["Loan Term", "Term"]);

  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Facts found',data:{hasGrossRent:!!grossRentFact,hasOperatingExpenses:!!operatingExpensesFact,hasLoanAmount:!!loanAmountFact,hasInterestRate:!!interestRateFact,hasLoanTerm:!!loanTermFact},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
  // #endregion

  const grossRent = grossRentFact ? extractNumericValue(grossRentFact) : null;
  const operatingExpenses = operatingExpensesFact ? extractNumericValue(operatingExpensesFact) : null;

  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Extracted values',data:{grossRent,operatingExpenses},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
  // #endregion

  // Need at least Gross Rent and Operating Expenses to calculate NOI
  if (grossRent === null || operatingExpenses === null) {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Missing required facts',data:{grossRent,operatingExpenses},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
    // #endregion
    return null;
  }

  // Calculate NOI
  const noi = grossRent - operatingExpenses;

  // Calculate debt service if we have loan details
  let debtService: number | undefined;
  let dscr: number | undefined;
  let cashFlowAfterDebt: number | undefined;

  const loanAmount = loanAmountFact ? extractNumericValue(loanAmountFact) : null;
  const interestRate = interestRateFact ? extractNumericValue(interestRateFact) : null;
  const loanTerm = loanTermFact ? extractNumericValue(loanTermFact) : null;

  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Loan details extracted',data:{loanAmount,interestRate,loanTerm},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
  // #endregion

  if (loanAmount !== null && interestRate !== null && loanTerm !== null) {
    // Calculate annual debt service using amortization formula
    // Monthly payment = P * [r(1+r)^n] / [(1+r)^n - 1]
    // Where P = principal, r = monthly interest rate, n = number of payments
    const monthlyRate = (interestRate / 100) / 12;
    const numPayments = loanTerm * 12;
    const monthlyPayment = loanAmount * (monthlyRate * Math.pow(1 + monthlyRate, numPayments)) / 
                           (Math.pow(1 + monthlyRate, numPayments) - 1);
    debtService = monthlyPayment * 12; // Annual debt service

    dscr = debtService > 0 ? noi / debtService : undefined;
    cashFlowAfterDebt = noi - debtService;

    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Debt service calculated',data:{monthlyPayment,debtService,dscr,cashFlowAfterDebt},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
    // #endregion
  }

  // Build audit trail
  const auditTrail: CalculationStep[] = [
    {
      metric: "Net Operating Income (NOI)",
      formula: "Gross Rent - Operating Expenses",
      inputs: [
        ["Gross Rent", grossRent],
        ["Operating Expenses", operatingExpenses],
      ],
      result: noi,
      sources: [
        grossRentFact?.fact_id || "",
        operatingExpensesFact?.fact_id || "",
      ].filter(Boolean),
    },
  ];

  if (debtService !== undefined && dscr !== undefined) {
    auditTrail.push({
      metric: "Debt Service Coverage Ratio (DSCR)",
      formula: "NOI / Annual Debt Service",
      inputs: [
        ["NOI", noi],
        ["Annual Debt Service", debtService],
      ],
      result: dscr,
      sources: [
        loanAmountFact?.fact_id || "",
        interestRateFact?.fact_id || "",
        loanTermFact?.fact_id || "",
      ].filter(Boolean),
    });

    if (cashFlowAfterDebt !== undefined) {
      auditTrail.push({
        metric: "Cash Flow After Debt Service",
        formula: "NOI - Annual Debt Service",
        inputs: [
          ["NOI", noi],
          ["Annual Debt Service", debtService],
        ],
        result: cashFlowAfterDebt,
        sources: [],
      });
    }
  }

  const warnings: string[] = [];
  if (dscr !== undefined && dscr < 1.2) {
    warnings.push("DSCR below 1.2 - may indicate high risk");
  }
  if (dscr !== undefined && dscr > 2.0) {
    warnings.push("DSCR above 2.0 - strong cash flow coverage");
  }

  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateFromFacts',message:'Calculation complete',data:{noi,dscr,cashFlowAfterDebt,auditTrailLength:auditTrail.length,warningsCount:warnings.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'B'})}).catch(()=>{});
  // #endregion

  return {
    noi,
    dscr,
    cash_flow_after_debt: cashFlowAfterDebt,
    cap_rate: undefined,
    ltv: undefined,
    gross_rent_multiplier: undefined,
    audit_trail: auditTrail,
    warnings,
  };
};

// Calculate underwriting metrics for a deal
export const calculateUnderwriting = async (
  dealId: string
): Promise<UnderwritingResult> => {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'calculateUnderwriting called',data:{dealId,USE_MOCK_DATA},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
  // #endregion

  if (USE_MOCK_DATA) {
    console.log(`[Mock] Calculating underwriting for deal: ${dealId}`);
    
    // First check if we have pre-defined mock data
    const mockData = MOCK_UNDERWRITING_DATA[dealId];
    if (mockData) {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'Using pre-defined mock data',data:{dealId},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
      return new Promise((resolve) => setTimeout(() => resolve(mockData), 500));
    }

    // Try to calculate from facts
    try {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'Fetching facts for dynamic calculation',data:{dealId},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
      
      const facts = await getDealFacts(dealId);
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'Facts retrieved',data:{dealId,factsCount:facts.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
      
      const calculated = calculateFromFacts(facts);
      
      if (calculated) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'Dynamic calculation successful',data:{dealId,noi:calculated.noi,dscr:calculated.dscr},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
        // #endregion
        return new Promise((resolve) => setTimeout(() => resolve(calculated), 500));
      }
      
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'Dynamic calculation failed - insufficient facts',data:{dealId,factsCount:facts.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
    } catch (error) {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'underwritingApi.ts:calculateUnderwriting',message:'Error fetching facts',data:{dealId,error:error instanceof Error?error.message:String(error)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'C'})}).catch(()=>{});
      // #endregion
    }

    throw new Error("Mock underwriting data not found for this deal and insufficient facts to calculate");
  }

  const response = await axiosInstance.post(
    `/api/v1/deals/${dealId}/underwrite`
  );
  return response.data;
};

// Apply stress test scenarios (frontend calculation)
export const applyStressTest = (
  input: StressTestInput
): StressTestResult => {
  const base = input.base_result;

  // Extract base values from audit trail
  let gross_rent = 0;
  let operating_expenses = 0;
  let debt_service = 0;

  for (const step of base.audit_trail) {
    for (const [name, value] of step.inputs) {
      if (name === "Gross Scheduled Rent") gross_rent = value;
      if (name === "Collected Rent") gross_rent = value;
      if (name === "Operating Expenses") operating_expenses = value;
      if (name === "Annual Debt Service") debt_service = value;
      if (name === "Debt Service") debt_service = value;
    }
  }

  // Apply adjustments
  if (input.rent_adjustment !== undefined) {
    gross_rent *= 1 + input.rent_adjustment / 100;
  }

  if (input.expense_adjustment !== undefined) {
    operating_expenses *= 1 + input.expense_adjustment / 100;
  }

  if (input.interest_rate_adjustment !== undefined) {
    // Simplified: adjust debt service proportionally
    debt_service *= 1 + input.interest_rate_adjustment / 10000;
  }

  // Calculate stressed metrics
  const stressed_noi = gross_rent - operating_expenses;
  const stressed_dscr = debt_service > 0 ? stressed_noi / debt_service : undefined;
  const stressed_cash_flow = stressed_noi - debt_service;

  // Calculate changes
  const noi_change = stressed_noi - base.noi;
  const noi_change_pct = base.noi !== 0 ? (noi_change / base.noi) * 100 : 0;

  const dscr_change =
    stressed_dscr !== undefined && base.dscr !== undefined
      ? stressed_dscr - base.dscr
      : undefined;

  const cash_flow_change =
    base.cash_flow_after_debt !== undefined
      ? stressed_cash_flow - base.cash_flow_after_debt
      : undefined;

  return {
    stressed_noi,
    stressed_dscr,
    stressed_cash_flow,
    comparison: {
      noi_change,
      noi_change_pct,
      dscr_change,
      cash_flow_change,
    },
  };
};

