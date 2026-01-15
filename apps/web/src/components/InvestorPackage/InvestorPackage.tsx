import { Flex, Card, Text, Button, Badge } from "@radix-ui/themes";
import { useState } from "react";
import { useQuery } from "react-query";
import { getDeals, DealResponse } from "../../services/dealApi";
import ShareWithContactsModal from "../Contacts/ShareWithContactsModal";
import "./InvestorPackage.css";

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

// Mock Investor Memo for Riverside Townhomes
const MOCK_INVESTOR_MEMO: InvestorMemo = {
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
};

const InvestorPackage = () => {
  const [showShareModal, setShowShareModal] = useState(false);
  const [selectedMemo, setSelectedMemo] = useState<InvestorMemo | null>(null);
  const { data: deals, isLoading } = useQuery<DealResponse[]>({
    queryKey: ["deals"],
    queryFn: getDeals,
  });

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

  // Filter deals that have completed underwriting
  const dealsWithMemos = deals?.filter(
    (deal) => deal.status === "ready_for_underwriting"
  ) || [];

  // For now, show mock memo for Riverside Townhomes
  const investorMemo = MOCK_INVESTOR_MEMO;

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
      <Text size="3" color="gray">
        Investor-ready deal summaries and investment memorandums
      </Text>

      <Flex gap="4" wrap="wrap">
        {dealsWithMemos.map((deal) => {
          const memo = deal.deal_id === investorMemo.dealId ? investorMemo : null;
          if (!memo) return null;

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
                <Flex justify="between" align="start">
                  <Flex direction="column" gap="2">
                    <Text size="7" weight="bold">
                      {memo.dealName}
                    </Text>
                    <Text size="3" color="gray">
                      Investment Memorandum
                    </Text>
                  </Flex>
                  <Badge color="green" size="2">
                    Equity Investment
                  </Badge>
                </Flex>

                {/* Executive Summary */}
                <Card style={{ background: "#f8f9fa", padding: "20px" }}>
                  <Text size="4" weight="bold" mb="2">
                    Executive Summary
                  </Text>
                  <br />
                  <Text size="3" style={{ lineHeight: "1.6" }}>
                    {memo.executiveSummary}
                  </Text>
                </Card>

                {/* Property Details */}
                <Flex gap="4" wrap="wrap">
                  <Card style={{ flex: 1, minWidth: "200px", padding: "16px" }}>
                    <Text size="2" color="gray" mb="1">
                      Property Type:
                    </Text>
                    <br />
                    <Text size="3" weight="medium">
                      {memo.propertyDetails.propertyType}
                    </Text>
                  </Card>
                  <Card style={{ flex: 1, minWidth: "200px", padding: "16px" }}>
                    <Text size="2" color="gray" mb="1">
                      Units:
                    </Text>
                    <br />
                    <Text size="3" weight="medium">
                      {memo.propertyDetails.units}
                    </Text>
                  </Card>
                  <Card style={{ flex: 1, minWidth: "200px", padding: "16px" }}>
                    <Text size="2" color="gray" mb="1">
                      Year Built:
                    </Text>
                    <br />
                    <Text size="3" weight="medium">
                      {memo.propertyDetails.yearBuilt}
                    </Text>
                  </Card>
                  <Card style={{ flex: 1, minWidth: "200px", padding: "16px" }}>
                    <Text size="2" color="gray" mb="1">
                      Location:
                    </Text>
                    <br />
                    <Text size="2" weight="medium">
                      {memo.propertyDetails.address}
                    </Text>
                  </Card>
                </Flex>

                {/* Financial Highlights */}
                <Card style={{ background: "#f0f9ff", padding: "20px", border: "2px solid #0ea5e9" }}>
                  <Text size="4" weight="bold" mb="3">
                    Financial Highlights
                  </Text>
                  <Flex gap="4" wrap="wrap">
                    <Flex direction="column" gap="1" style={{ flex: 1, minWidth: "150px" }}>
                      <Text size="2" color="gray">
                        Net Operating Income
                      </Text>
                      <Text size="5" weight="bold">
                        {formatCurrency(memo.financialHighlights.noi)}
                      </Text>
                    </Flex>
                    <Flex direction="column" gap="1" style={{ flex: 1, minWidth: "150px" }}>
                      <Text size="2" color="gray">
                        DSCR
                      </Text>
                      <Text size="5" weight="bold">
                        {memo.financialHighlights.dscr.toFixed(2)}x
                      </Text>
                    </Flex>
                    <Flex direction="column" gap="1" style={{ flex: 1, minWidth: "150px" }}>
                      <Text size="2" color="gray">
                        Cap Rate
                      </Text>
                      <Text size="5" weight="bold">
                        {formatPercent(memo.financialHighlights.capRate)}
                      </Text>
                    </Flex>
                    <Flex direction="column" gap="1" style={{ flex: 1, minWidth: "150px" }}>
                      <Text size="2" color="gray">
                        Annual Cash Flow
                      </Text>
                      <Text size="5" weight="bold">
                        {formatCurrency(memo.financialHighlights.cashFlow)}
                      </Text>
                    </Flex>
                  </Flex>
                </Card>

                {/* Investment Thesis */}
                <Card style={{ padding: "20px" }}>
                  <Text size="4" weight="bold" mb="3">
                    Investment Thesis
                  </Text>
                  <Flex direction="column" gap="2">
                    {memo.investmentThesis.map((point, idx) => (
                      <Flex key={idx} align="start" gap="2">
                        <Text size="3" style={{ color: "#16a34a" }}>
                          ✓
                        </Text>
                        <Text size="3" style={{ flex: 1 }}>
                          {point}
                        </Text>
                      </Flex>
                    ))}
                  </Flex>
                </Card>

                {/* Risk Factors */}
                <Card style={{ background: "#fff7ed", padding: "20px", border: "1px solid #fb923c" }}>
                  <Text size="4" weight="bold" mb="3">
                    Risk Factors
                  </Text>
                  <Flex direction="column" gap="2">
                    {memo.riskFactors.map((risk, idx) => (
                      <Flex key={idx} align="start" gap="2">
                        <Text size="3" style={{ color: "#ea580c" }}>
                          ⚠
                        </Text>
                        <Text size="3" style={{ flex: 1 }}>
                          {risk}
                        </Text>
                      </Flex>
                    ))}
                  </Flex>
                </Card>

                {/* Use of Proceeds */}
                <Card style={{ padding: "20px" }}>
                  <Text size="4" weight="bold" mb="2">
                    Use of Proceeds
                  </Text>
                  <br />
                  <Text size="3" style={{ lineHeight: "1.6" }}>
                    {memo.useOfProceeds}
                  </Text>
                </Card>

                {/* Timeline */}
                <Card style={{ padding: "20px" }}>
                  <Text size="4" weight="bold" mb="2">
                    Investment Timeline
                  </Text>
                  <br />
                  <Text size="3" style={{ lineHeight: "1.6" }}>
                    {memo.timeline}
                  </Text>
                </Card>

                {/* Actions */}
                <Flex gap="3" justify="end" mt="2">
                  <Button variant="soft">Download PDF</Button>
                  <Button
                    onClick={() => {
                      setSelectedMemo(memo);
                      setShowShareModal(true);
                    }}
                  >
                    Share with Contacts
                  </Button>
                </Flex>
              </Flex>
            </Card>
          );
        })}
      </Flex>

      {/* Share with Contacts Modal */}
      {selectedMemo && (
        <ShareWithContactsModal
          open={showShareModal}
          onClose={() => {
            setShowShareModal(false);
            setSelectedMemo(null);
          }}
          dealName={selectedMemo.dealName}
          memoData={{
            executiveSummary: selectedMemo.executiveSummary,
            financialHighlights: selectedMemo.financialHighlights,
            propertyDetails: selectedMemo.propertyDetails,
          }}
        />
      )}
    </Flex>
  );
};

export default InvestorPackage;

