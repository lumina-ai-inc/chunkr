import { Flex, Card, Text, Button, TextArea, TextField } from "@radix-ui/themes";
import { useState, useEffect } from "react";
import { useQuery } from "react-query";
import { getDeals, DealResponse, getDeal } from "../../services/dealApi";
import { CreateLiveShareModal } from "../LiveShare/CreateLiveShareModal";
import { DealTypeBadge } from "../Dashboard/DealTypeBadge";
import "./InvestorPackage.css";

interface MemoBlock {
  id: string;
  title: string;
  content: string;
  type: "text" | "list";
  listItems?: string[];
}

interface InvestorMemo {
  dealId: string;
  dealName: string;
  executiveSummary: string;
  propertyDetails: {
    address: string;
    propertyType: string;
    units: number;
    yearBuilt: number;
  };
  financialHighlights: {
    noi: number;
    dscr: number;
    capRate: number;
    cashFlow: number;
  };
  investmentThesis: string[];
  riskFactors: string[];
  useOfProceeds: string;
  timeline: string;
}

// Mock Investor Memos for different deals
const MOCK_INVESTOR_MEMOS: Record<string, InvestorMemo> = {
  "deal-002-mockdata": {
    dealId: "deal-002-mockdata",
    dealName: "Downtown Commercial Property",
    executiveSummary: "Downtown Commercial Property offers a prime investment opportunity in a central business district. The property features modern office and retail spaces with strong tenant retention and consistent rental income. With a DSCR of 2.1x and NOI of $120,000, this property represents a stable investment for institutional investors seeking commercial real estate exposure.",
    propertyDetails: {
      address: "352 E 25th Street, Baltimore, MD 21218",
      propertyType: "Commercial Mixed-Use",
      units: 8,
      yearBuilt: 2015,
    },
    financialHighlights: {
      noi: 120000,
      dscr: 2.1,
      capRate: 6.2,
      cashFlow: 65000,
    },
    investmentThesis: [
      "Excellent DSCR of 2.1x provides strong debt service coverage",
      "Prime downtown location with high foot traffic and visibility",
      "Mixed-use property diversifies income streams (office + retail)",
      "Modern construction (2015) reduces maintenance requirements",
      "Strong cash flow of $65,000 annually with growth potential",
    ],
    riskFactors: [
      "Commercial real estate market sensitivity to economic cycles",
      "Tenant concentration risk if major tenant vacates",
      "Property tax assessments may increase with area development",
      "Competition from newer developments in the area",
    ],
    useOfProceeds: "Proceeds will be allocated to property renovations, tenant improvements, and debt refinancing. Approximately 50% for capital improvements, 40% for debt reduction, and 10% for operational reserves.",
    timeline: "Closing expected within 45-60 days. Renovations to commence immediately post-closing with completion within 120 days. First investor distribution scheduled for Q3 2024.",
  },
  "deal-003-mockdata": {
  dealId: "deal-003-mockdata",
  dealName: "Riverside Townhomes",
  executiveSummary: "Riverside Townhomes presents an attractive investment opportunity in a well-established residential community. The property consists of 10 townhome units with strong occupancy (90%) and healthy cash flow. With a DSCR of 1.75x and NOI of $70,000, the property demonstrates solid fundamentals suitable for institutional and accredited investors seeking stable income with moderate growth potential.",
  propertyDetails: {
    address: "123 Riverside Drive, Austin, TX 78701",
    propertyType: "Multi-Family Townhomes",
    units: 10,
    yearBuilt: 2018,
  },
  financialHighlights: {
    noi: 70000,
    dscr: 1.75,
    capRate: 5.8,
    cashFlow: 30000,
  },
  investmentThesis: [
    "Strong DSCR of 1.75x provides comfortable debt service coverage",
    "90% occupancy rate indicates stable tenant base and market demand",
    "Property located in growing Austin metro area with strong job market",
    "Recent construction (2018) minimizes near-term capital expenditure needs",
    "Positive cash flow of $30,000 annually provides attractive yield",
  ],
  riskFactors: [
    "Management fee of 5.8% is slightly above market average (3-5%)",
    "Single property concentration risk - all units in one location",
    "Market rent growth assumptions may be impacted by economic conditions",
    "Property tax increases could impact cash flow margins",
  ],
  useOfProceeds: "Proceeds from this investment will be used to refinance existing debt, fund property improvements, and provide working capital for operations. Approximately 60% will go toward debt reduction, 30% toward capital improvements, and 10% toward operational reserves.",
  timeline: "Investment closing expected within 30-45 days of commitment. Property improvements to be completed within 90 days post-closing. First distribution to investors scheduled for Q2 2024.",
  },
};

// Default memo for deals without specific mock data
const getDefaultMemo = (dealId: string, dealName: string): InvestorMemo => ({
  dealId,
  dealName,
  executiveSummary: `${dealName} presents an attractive investment opportunity. Complete underwriting analysis to generate detailed investment memorandum.`,
  propertyDetails: {
    address: "Address to be determined",
    propertyType: "Multi-Family",
    units: 0,
    yearBuilt: 0,
  },
  financialHighlights: {
    noi: 0,
    dscr: 0,
    capRate: 0,
    cashFlow: 0,
  },
  investmentThesis: [],
  riskFactors: [],
  useOfProceeds: "To be determined based on investment structure.",
  timeline: "Timeline to be established upon commitment.",
});

const InvestorPackage = () => {
  const [showCreateLiveShareModal, setShowCreateLiveShareModal] = useState(false);
  const [editingField, setEditingField] = useState<string | null>(null);
  const [editingValue, setEditingValue] = useState<string>("");
  const [customBlocks, setCustomBlocks] = useState<MemoBlock[]>([]);
  
  const { data: deals, isLoading } = useQuery<DealResponse[]>({
    queryKey: ["deals"],
    queryFn: getDeals,
  });

  // Filter deals that have completed underwriting
  const dealsWithMemos = deals?.filter(
    (deal) => deal.status === "ready_for_underwriting"
  ) || [];

  // Initialize memo based on the first deal with memo, or use default
  const initialDeal = dealsWithMemos[0];
  const initialMemo = initialDeal
    ? MOCK_INVESTOR_MEMOS[initialDeal.deal_id] || getDefaultMemo(initialDeal.deal_id, initialDeal.deal_name)
    : getDefaultMemo("", "");

  const [memo, setMemo] = useState<InvestorMemo>(initialMemo);

  // Get deal data for the current memo to access deal_type
  const { data: currentDeal } = useQuery<DealResponse>(
    ["deal", memo.dealId],
    () => getDeal(memo.dealId),
    { enabled: !!memo.dealId }
  );

  // Update memo when deals change - initialize with correct deal-specific data
  useEffect(() => {
    if (dealsWithMemos.length > 0 && deals) {
      // Get the first deal (or the one that matches current memo if it exists)
      const currentDeal = dealsWithMemos.find((d) => d.deal_id === memo.dealId) || dealsWithMemos[0];
      
      // Always start with mock data to ensure correct addresses
      const mockMemo = MOCK_INVESTOR_MEMOS[currentDeal.deal_id] || getDefaultMemo(currentDeal.deal_id, currentDeal.deal_name);
      
      // Check localStorage for user edits
      const savedMemoKey = `memo_${currentDeal.deal_id}`;
      const savedMemo = localStorage.getItem(savedMemoKey);
      
      if (savedMemo) {
        try {
          const parsed = JSON.parse(savedMemo);
          // Only use saved memo if it's for the correct deal
          if (parsed.dealId === currentDeal.deal_id) {
            // Merge: mock data as base, but preserve user edits
            // CRITICAL: Always use mock address (don't use old saved address)
            const mergedMemo = {
              ...mockMemo,
              ...parsed,
              propertyDetails: {
                ...mockMemo.propertyDetails, // Start with mock property details
                ...parsed.propertyDetails,   // Apply user edits
                address: mockMemo.propertyDetails.address, // Force correct address from mock
              },
            };
            setMemo(mergedMemo);
            // Update localStorage with correct address
            localStorage.setItem(savedMemoKey, JSON.stringify(mergedMemo));
          } else {
            // Saved memo is for wrong deal, use fresh mock data
            setMemo(mockMemo);
          }
        } catch (e) {
          console.error("Failed to load saved memo:", e);
          setMemo(mockMemo);
        }
      } else {
        // No saved memo, use fresh mock data
        setMemo(mockMemo);
      }
    }
  }, [deals]);

  // Load custom blocks from localStorage
  useEffect(() => {
    if (memo.dealId) {
      const savedBlocks = localStorage.getItem(`memo_blocks_${memo.dealId}`);
      if (savedBlocks) {
        try {
          setCustomBlocks(JSON.parse(savedBlocks));
        } catch (e) {
          console.error("Failed to load saved blocks:", e);
        }
      }
    }
  }, [memo.dealId]);

  // Save memo to localStorage when it changes (but preserve mock address)
  useEffect(() => {
    if (memo.dealId) {
      // Get the mock memo to ensure we always save with correct address
      const mockMemo = MOCK_INVESTOR_MEMOS[memo.dealId];
      if (mockMemo) {
        // When saving, ensure address comes from mock data
        const memoToSave = {
          ...memo,
          propertyDetails: {
            ...memo.propertyDetails,
            address: mockMemo.propertyDetails.address, // Always use mock address
          },
        };
        localStorage.setItem(`memo_${memo.dealId}`, JSON.stringify(memoToSave));
      } else {
        localStorage.setItem(`memo_${memo.dealId}`, JSON.stringify(memo));
      }
    }
  }, [memo]);

  // Save custom blocks to localStorage when they change
  useEffect(() => {
    localStorage.setItem(`memo_blocks_${memo.dealId}`, JSON.stringify(customBlocks));
  }, [customBlocks, memo.dealId]);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const handleStartEdit = (field: string, currentValue: string) => {
    setEditingField(field);
    setEditingValue(currentValue);
  };

  const handleSaveEdit = (field: string) => {
    if (field === "dealName") {
      setMemo({ ...memo, dealName: editingValue.trim() });
    } else if (field.startsWith("executiveSummary")) {
      setMemo({ ...memo, executiveSummary: editingValue });
    } else if (field.startsWith("useOfProceeds")) {
      setMemo({ ...memo, useOfProceeds: editingValue });
    } else if (field.startsWith("timeline")) {
      setMemo({ ...memo, timeline: editingValue });
    } else if (field.startsWith("propertyDetails.")) {
      const prop = field.split(".")[1];
      setMemo({
        ...memo,
        propertyDetails: { ...memo.propertyDetails, [prop]: editingValue },
      });
    } else if (field.startsWith("financialHighlights.")) {
      const prop = field.split(".")[1];
      setMemo({
        ...memo,
        financialHighlights: {
          ...memo.financialHighlights,
          [prop]: parseFloat(editingValue) || 0,
        },
      });
    } else if (field.startsWith("investmentThesis.")) {
      const index = parseInt(field.split(".")[1]);
      const newThesis = [...memo.investmentThesis];
      newThesis[index] = editingValue;
      setMemo({ ...memo, investmentThesis: newThesis });
    } else if (field.startsWith("riskFactors.")) {
      const index = parseInt(field.split(".")[1]);
      const newRisks = [...memo.riskFactors];
      newRisks[index] = editingValue;
      setMemo({ ...memo, riskFactors: newRisks });
    } else if (field.startsWith("customBlock.")) {
      const blockId = field.split(".")[1];
      setCustomBlocks(
        customBlocks.map((block) =>
          block.id === blockId ? { ...block, content: editingValue } : block
        )
      );
    } else if (field.startsWith("customBlockList.")) {
      const [blockId, itemIndex] = field.split(".").slice(1);
      const blockIndex = customBlocks.findIndex((b) => b.id === blockId);
      if (blockIndex !== -1) {
        const newBlocks = [...customBlocks];
        const newItems = [...(newBlocks[blockIndex].listItems || [])];
        newItems[parseInt(itemIndex)] = editingValue;
        newBlocks[blockIndex] = { ...newBlocks[blockIndex], listItems: newItems };
        setCustomBlocks(newBlocks);
      }
    }
    setEditingField(null);
    setEditingValue("");
  };

  const handleCancelEdit = () => {
    setEditingField(null);
    setEditingValue("");
  };

  const handleAddListItem = (listType: "investmentThesis" | "riskFactors") => {
    if (listType === "investmentThesis") {
      setMemo({
        ...memo,
        investmentThesis: [...memo.investmentThesis, "New point..."],
      });
    } else {
      setMemo({
        ...memo,
        riskFactors: [...memo.riskFactors, "New risk..."],
      });
    }
  };

  const handleDeleteListItem = (listType: "investmentThesis" | "riskFactors", index: number) => {
    if (listType === "investmentThesis") {
      setMemo({
        ...memo,
        investmentThesis: memo.investmentThesis.filter((_, i) => i !== index),
      });
    } else {
      setMemo({
        ...memo,
        riskFactors: memo.riskFactors.filter((_, i) => i !== index),
      });
    }
  };

  const handleAddCustomBlock = () => {
    const newBlock: MemoBlock = {
      id: `block-${Date.now()}`,
      title: "New Section",
      content: "Click to edit content...",
      type: "text",
    };
    setCustomBlocks([...customBlocks, newBlock]);
  };

  const handleDeleteCustomBlock = (blockId: string) => {
    setCustomBlocks(customBlocks.filter((b) => b.id !== blockId));
  };

  if (isLoading) {
    return (
      <Flex justify="center" align="center" p="8">
        <Text>Loading investor packages...</Text>
      </Flex>
    );
  }

  if (dealsWithMemos.length === 0) {
    return (
      <Flex
        direction="column"
        align="center"
        justify="center"
        p="8"
        gap="4"
        style={{ height: "100%" }}
      >
        <Text size="6" weight="bold" color="gray">
          No Investor Packages Available
        </Text>
        <Text size="3" color="gray">
          Complete underwriting analysis to generate investor-ready packages
        </Text>
      </Flex>
    );
  }

  return (
    <Flex
      direction="column"
      gap="4"
      p="24px"
      style={{ overflowY: "auto", height: "100%", minHeight: 0 }}
      className="investor-package-container"
    >

      <Flex gap="4" wrap="wrap">
        {dealsWithMemos.map((deal) => {
          if (deal.deal_id !== memo.dealId) return null;

          return (
            <Card
              key={deal.deal_id}
              style={{
                width: "100%",
                maxWidth: "900px",
                padding: "32px",
                border: "1px solid #e0e0e0",
              }}
            >
              <Flex direction="column" gap="4">
                {/* Header */}
                <Flex direction="column" gap="2" style={{ position: "relative" }}>
                  <Flex justify="end" style={{ position: "absolute", top: 0, right: 0, zIndex: 10 }}>
                    <DealTypeBadge dealType={currentDeal?.deal_type} />
                  </Flex>
                  <Flex direction="column" gap="2" style={{ flex: 1 }}>
                    {editingField === "dealName" ? (
                  <Flex direction="column" gap="2">
                        <TextField.Root
                          value={editingValue}
                          onChange={(e) => setEditingValue(e.target.value)}
                          placeholder="Deal Name"
                          autoFocus
                          style={{ width: "100%" }}
                        />
                        <Flex gap="2">
                          <Button size="2" onClick={() => handleSaveEdit("dealName")} disabled={!editingValue.trim()}>
                            Save
                          </Button>
                          <Button size="2" variant="soft" onClick={handleCancelEdit}>
                            Cancel
                          </Button>
                        </Flex>
                      </Flex>
                    ) : (
                      <Text
                        size="7"
                        weight="bold"
                        onClick={() => !editingField && handleStartEdit("dealName", memo.dealName)}
                        style={{
                          cursor: "pointer",
                          padding: "4px 8px",
                          borderRadius: "4px",
                          transition: "background-color 0.2s",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = "#f0f0f0";
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = "transparent";
                        }}
                      >
                      {memo.dealName}
                    </Text>
                    )}
                    <Text size="3" color="gray">
                      Investment Memorandum
                    </Text>
                  </Flex>
                </Flex>

                {/* Executive Summary */}
                <Card
                  style={{
                    background: "#f8f9fa",
                    padding: "20px",
                    cursor: "pointer",
                    border: editingField === "executiveSummary" ? "2px solid #1976D2" : "1px solid #e0e0e0",
                  }}
                  onClick={() => !editingField && handleStartEdit("executiveSummary", memo.executiveSummary)}
                >
                  <Text size="4" weight="bold" mb="2">
                    Executive Summary
                  </Text>
                  <br />
                  {editingField === "executiveSummary" ? (
                    <Flex direction="column" gap="2">
                      <TextArea
                        value={editingValue}
                        onChange={(e) => setEditingValue(e.target.value)}
                        rows={6}
                        autoFocus
                        style={{ width: "100%" }}
                      />
                      <Flex gap="2">
                        <Button size="2" onClick={() => handleSaveEdit("executiveSummary")}>
                          Save
                        </Button>
                        <Button size="2" variant="soft" onClick={handleCancelEdit}>
                          Cancel
                        </Button>
                      </Flex>
                    </Flex>
                  ) : (
                  <Text size="3" style={{ lineHeight: "1.6" }}>
                    {memo.executiveSummary}
                  </Text>
                  )}
                </Card>

                {/* Property Details */}
                <Flex gap="4" wrap="wrap">
                  {[
                    { key: "propertyType", label: "Property Type", value: memo.propertyDetails.propertyType },
                    { key: "units", label: "Units", value: String(memo.propertyDetails.units) },
                    { key: "yearBuilt", label: "Year Built", value: String(memo.propertyDetails.yearBuilt) },
                    { key: "address", label: "Location", value: memo.propertyDetails.address },
                  ].map(({ key, label, value }) => (
                    <Card
                      key={key}
                      style={{
                        flex: 1,
                        minWidth: "200px",
                        padding: "16px",
                        cursor: "pointer",
                        border: editingField === `propertyDetails.${key}` ? "2px solid #1976D2" : "1px solid #e0e0e0",
                      }}
                      onClick={() => !editingField && handleStartEdit(`propertyDetails.${key}`, value)}
                    >
                    <Text size="2" color="gray" mb="1">
                        {label}:
                    </Text>
                    <br />
                      {editingField === `propertyDetails.${key}` ? (
                        <Flex direction="column" gap="2" mt="2">
                          <TextField.Root
                            value={editingValue}
                            onChange={(e) => setEditingValue(e.target.value)}
                            autoFocus
                          />
                          <Flex gap="2">
                            <Button size="1" onClick={() => handleSaveEdit(`propertyDetails.${key}`)}>
                              Save
                            </Button>
                            <Button size="1" variant="soft" onClick={handleCancelEdit}>
                              Cancel
                            </Button>
                          </Flex>
                        </Flex>
                      ) : (
                        <Text size={key === "address" ? "2" : "3"} weight="medium">
                          {value}
                    </Text>
                      )}
                  </Card>
                  ))}
                </Flex>

                {/* Financial Highlights */}
                <Card style={{ background: "#f0f9ff", padding: "20px", border: "2px solid #0ea5e9" }}>
                  <Text size="4" weight="bold" mb="3">
                    Financial Highlights
                  </Text>
                  <Flex gap="4" wrap="wrap">
                    {[
                      { key: "noi", label: "Net Operating Income", value: formatCurrency(memo.financialHighlights.noi), rawValue: memo.financialHighlights.noi },
                      { key: "dscr", label: "DSCR", value: `${memo.financialHighlights.dscr.toFixed(2)}x`, rawValue: memo.financialHighlights.dscr },
                      { key: "capRate", label: "Cap Rate", value: formatPercent(memo.financialHighlights.capRate), rawValue: memo.financialHighlights.capRate },
                      { key: "cashFlow", label: "Annual Cash Flow", value: formatCurrency(memo.financialHighlights.cashFlow), rawValue: memo.financialHighlights.cashFlow },
                    ].map(({ key, label, value, rawValue }) => (
                      <Flex
                        key={key}
                        direction="column"
                        gap="1"
                        style={{
                          flex: 1,
                          minWidth: "150px",
                          padding: "8px",
                          borderRadius: "4px",
                          cursor: editingField !== `financialHighlights.${key}` ? "pointer" : "default",
                          backgroundColor: editingField === `financialHighlights.${key}` ? "#e0f2fe" : "transparent",
                        }}
                        onClick={() => !editingField && handleStartEdit(`financialHighlights.${key}`, String(rawValue))}
                      >
                      <Text size="2" color="gray">
                          {label}
                      </Text>
                        {editingField === `financialHighlights.${key}` ? (
                          <Flex direction="column" gap="2">
                            <TextField.Root
                              type="number"
                              value={editingValue}
                              onChange={(e) => setEditingValue(e.target.value)}
                              autoFocus
                            />
                            <Flex gap="2">
                              <Button size="1" onClick={() => handleSaveEdit(`financialHighlights.${key}`)}>
                                Save
                              </Button>
                              <Button size="1" variant="soft" onClick={handleCancelEdit}>
                                Cancel
                              </Button>
                    </Flex>
                    </Flex>
                        ) : (
                      <Text size="5" weight="bold">
                            {value}
                      </Text>
                        )}
                    </Flex>
                    ))}
                  </Flex>
                </Card>

                {/* Investment Thesis */}
                <Card style={{ padding: "20px" }}>
                  <Flex justify="between" align="center" mb="3">
                    <Text size="4" weight="bold">
                    Investment Thesis
                  </Text>
                    <Button
                      size="1"
                      variant="soft"
                      onClick={() => handleAddListItem("investmentThesis")}
                      style={{ cursor: "pointer" }}
                    >
                      + Add
                    </Button>
                  </Flex>
                  <Flex direction="column" gap="2">
                    {memo.investmentThesis.map((point, idx) => (
                      <Flex
                        key={idx}
                        align="start"
                        gap="2"
                        style={{
                          padding: "8px",
                          borderRadius: "4px",
                          cursor: editingField !== `investmentThesis.${idx}` ? "pointer" : "default",
                          backgroundColor: editingField === `investmentThesis.${idx}` ? "#f0f0f0" : "transparent",
                        }}
                        onClick={() =>
                          !editingField && handleStartEdit(`investmentThesis.${idx}`, point)
                        }
                      >
                        <Text size="3" style={{ color: "#16a34a" }}>
                          ✓
                        </Text>
                        {editingField === `investmentThesis.${idx}` ? (
                          <Flex direction="column" gap="2" style={{ flex: 1 }}>
                            <TextField.Root
                              value={editingValue}
                              onChange={(e) => setEditingValue(e.target.value)}
                              autoFocus
                            />
                            <Flex gap="2">
                              <Button size="1" onClick={() => handleSaveEdit(`investmentThesis.${idx}`)}>
                                Save
                              </Button>
                              <Button size="1" variant="soft" onClick={handleCancelEdit}>
                                Cancel
                              </Button>
                              <Button
                                size="1"
                                variant="soft"
                                color="red"
                                onClick={() => {
                                  handleDeleteListItem("investmentThesis", idx);
                                  handleCancelEdit();
                                }}
                              >
                                Delete
                              </Button>
                            </Flex>
                          </Flex>
                        ) : (
                        <Text size="3" style={{ flex: 1 }}>
                          {point}
                        </Text>
                        )}
                      </Flex>
                    ))}
                  </Flex>
                </Card>

                {/* Risk Factors */}
                <Card style={{ background: "#fff7ed", padding: "20px", border: "1px solid #fb923c" }}>
                  <Flex justify="between" align="center" mb="3">
                    <Text size="4" weight="bold">
                    Risk Factors
                  </Text>
                    <Button
                      size="1"
                      variant="soft"
                      onClick={() => handleAddListItem("riskFactors")}
                      style={{ cursor: "pointer" }}
                    >
                      + Add
                    </Button>
                  </Flex>
                  <Flex direction="column" gap="2">
                    {memo.riskFactors.map((risk, idx) => (
                      <Flex
                        key={idx}
                        align="start"
                        gap="2"
                        style={{
                          padding: "8px",
                          borderRadius: "4px",
                          cursor: editingField !== `riskFactors.${idx}` ? "pointer" : "default",
                          backgroundColor: editingField === `riskFactors.${idx}` ? "#f0f0f0" : "transparent",
                        }}
                        onClick={() => !editingField && handleStartEdit(`riskFactors.${idx}`, risk)}
                      >
                        <Text size="3" style={{ color: "#ea580c" }}>
                          ⚠
                        </Text>
                        {editingField === `riskFactors.${idx}` ? (
                          <Flex direction="column" gap="2" style={{ flex: 1 }}>
                            <TextField.Root
                              value={editingValue}
                              onChange={(e) => setEditingValue(e.target.value)}
                              autoFocus
                            />
                            <Flex gap="2">
                              <Button size="1" onClick={() => handleSaveEdit(`riskFactors.${idx}`)}>
                                Save
                              </Button>
                              <Button size="1" variant="soft" onClick={handleCancelEdit}>
                                Cancel
                              </Button>
                              <Button
                                size="1"
                                variant="soft"
                                color="red"
                                onClick={() => {
                                  handleDeleteListItem("riskFactors", idx);
                                  handleCancelEdit();
                                }}
                              >
                                Delete
                              </Button>
                            </Flex>
                          </Flex>
                        ) : (
                        <Text size="3" style={{ flex: 1 }}>
                          {risk}
                        </Text>
                        )}
                      </Flex>
                    ))}
                  </Flex>
                </Card>

                {/* Use of Proceeds */}
                <Card
                  style={{
                    padding: "20px",
                    cursor: "pointer",
                    border: editingField === "useOfProceeds" ? "2px solid #1976D2" : "1px solid #e0e0e0",
                  }}
                  onClick={() => !editingField && handleStartEdit("useOfProceeds", memo.useOfProceeds)}
                >
                  <Text size="4" weight="bold" mb="2">
                    Use of Proceeds
                  </Text>
                  <br />
                  {editingField === "useOfProceeds" ? (
                    <Flex direction="column" gap="2">
                      <TextArea
                        value={editingValue}
                        onChange={(e) => setEditingValue(e.target.value)}
                        rows={4}
                        autoFocus
                        style={{ width: "100%" }}
                      />
                      <Flex gap="2">
                        <Button size="2" onClick={() => handleSaveEdit("useOfProceeds")}>
                          Save
                        </Button>
                        <Button size="2" variant="soft" onClick={handleCancelEdit}>
                          Cancel
                        </Button>
                      </Flex>
                    </Flex>
                  ) : (
                  <Text size="3" style={{ lineHeight: "1.6" }}>
                    {memo.useOfProceeds}
                  </Text>
                  )}
                </Card>

                {/* Timeline */}
                <Card
                  style={{
                    padding: "20px",
                    cursor: "pointer",
                    border: editingField === "timeline" ? "2px solid #1976D2" : "1px solid #e0e0e0",
                  }}
                  onClick={() => !editingField && handleStartEdit("timeline", memo.timeline)}
                >
                  <Text size="4" weight="bold" mb="2">
                    Investment Timeline
                  </Text>
                  <br />
                  {editingField === "timeline" ? (
                    <Flex direction="column" gap="2">
                      <TextArea
                        value={editingValue}
                        onChange={(e) => setEditingValue(e.target.value)}
                        rows={3}
                        autoFocus
                        style={{ width: "100%" }}
                      />
                      <Flex gap="2">
                        <Button size="2" onClick={() => handleSaveEdit("timeline")}>
                          Save
                        </Button>
                        <Button size="2" variant="soft" onClick={handleCancelEdit}>
                          Cancel
                        </Button>
                      </Flex>
                    </Flex>
                  ) : (
                  <Text size="3" style={{ lineHeight: "1.6" }}>
                    {memo.timeline}
                  </Text>
                  )}
                </Card>

                {/* Custom Blocks */}
                {customBlocks.map((block) => (
                  <Card
                    key={block.id}
                    style={{
                      padding: "20px",
                      cursor: editingField === `customBlock.${block.id}` ? "default" : "pointer",
                      border: editingField === `customBlock.${block.id}` ? "2px solid #1976D2" : "1px solid #e0e0e0",
                    }}
                    onClick={() => !editingField && handleStartEdit(`customBlock.${block.id}`, block.content)}
                  >
                    <Flex justify="between" align="center" mb="2">
                      {editingField === `customBlockTitle.${block.id}` ? (
                        <Flex direction="column" gap="2" style={{ flex: 1 }}>
                          <TextField.Root
                            value={editingValue}
                            onChange={(e) => setEditingValue(e.target.value)}
                            placeholder="Section Title"
                            autoFocus
                          />
                          <Flex gap="2">
                            <Button
                              size="1"
                              onClick={(e) => {
                                e.stopPropagation();
                                setCustomBlocks(
                                  customBlocks.map((b) =>
                                    b.id === block.id ? { ...b, title: editingValue } : b
                                  )
                                );
                                setEditingField(null);
                                setEditingValue("");
                              }}
                            >
                              Save
                            </Button>
                            <Button size="1" variant="soft" onClick={(e) => {
                              e.stopPropagation();
                              handleCancelEdit();
                            }}>
                              Cancel
                            </Button>
                          </Flex>
                        </Flex>
                      ) : (
                        <Text
                          size="4"
                          weight="bold"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleStartEdit(`customBlockTitle.${block.id}`, block.title);
                          }}
                          style={{ cursor: "pointer" }}
                        >
                          {block.title}
                        </Text>
                      )}
                      <Button
                        size="1"
                        variant="soft"
                        color="red"
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCustomBlock(block.id);
                        }}
                      >
                        Delete
                      </Button>
                    </Flex>
                    {editingField === `customBlock.${block.id}` ? (
                      <Flex direction="column" gap="2" onClick={(e) => e.stopPropagation()}>
                        <TextArea
                          value={editingValue}
                          onChange={(e) => setEditingValue(e.target.value)}
                          rows={4}
                          autoFocus
                          style={{ width: "100%" }}
                        />
                        <Flex gap="2">
                          <Button size="2" onClick={(e) => {
                            e.stopPropagation();
                            handleSaveEdit(`customBlock.${block.id}`);
                          }}>
                            Save
                          </Button>
                          <Button size="2" variant="soft" onClick={(e) => {
                            e.stopPropagation();
                            handleCancelEdit();
                          }}>
                            Cancel
                          </Button>
                        </Flex>
                      </Flex>
                    ) : (
                      <Text
                        size="3"
                        style={{ lineHeight: "1.6" }}
                        onClick={(e) => {
                          e.stopPropagation();
                          handleStartEdit(`customBlock.${block.id}`, block.content);
                        }}
                      >
                        {block.content}
                      </Text>
                    )}
                  </Card>
                ))}

                {/* Add Custom Block Button */}
                <Flex justify="center">
                  <Button
                    size="2"
                    variant="soft"
                    onClick={handleAddCustomBlock}
                    style={{ cursor: "pointer" }}
                  >
                    + Add Block
                  </Button>
                </Flex>

                {/* Actions */}
                <Flex gap="3" justify="end" mt="2">
                  <Button variant="soft">Download PDF</Button>
                  <Button
                    onClick={() => {
                      setShowCreateLiveShareModal(true);
                    }}
                  >
                    Create Live Share
                  </Button>
                </Flex>
              </Flex>
            </Card>
          );
        })}
      </Flex>

      {/* Create Live Share Modal */}
      {memo && dealsWithMemos[0] && (
        <CreateLiveShareModal
          dealId={dealsWithMemos[0].deal_id}
          dealName={memo.dealName}
          isOpen={showCreateLiveShareModal}
          onClose={() => {
            setShowCreateLiveShareModal(false);
          }}
        />
      )}
    </Flex>
  );
};

export default InvestorPackage;

