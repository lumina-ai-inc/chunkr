import { useState } from "react";
import { Flex, Text, Card, Button, Badge } from "@radix-ui/themes";
import { useQuery } from "react-query";
import {
  calculateUnderwriting,
  applyStressTest,
  UnderwritingResult,
  StressTestResult,
} from "../../services/underwritingApi";
import StressTestPanel from "./StressTestPanel";
// import toast";
import "./UnderwritingDashboard.css";

interface UnderwritingDashboardProps {
  dealId: string;
}

const UnderwritingDashboard = ({ dealId }: UnderwritingDashboardProps) => {
  const [stressScenario, setStressScenario] = useState({
    rentAdjustment: 0,
    expenseAdjustment: 0,
    interestRateAdjustment: 0,
  });

  const {
    data: underwritingResult,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery<UnderwritingResult>({
    queryKey: ["underwriting", dealId],
    queryFn: () => calculateUnderwriting(dealId),
  });

  const stressTestResult: StressTestResult | null =
    underwritingResult && (stressScenario.rentAdjustment !== 0 ||
      stressScenario.expenseAdjustment !== 0 ||
      stressScenario.interestRateAdjustment !== 0)
      ? applyStressTest({
          base_result: underwritingResult,
          rent_adjustment: stressScenario.rentAdjustment,
          expense_adjustment: stressScenario.expenseAdjustment,
          interest_rate_adjustment: stressScenario.interestRateAdjustment,
        })
      : null;

  const formatCurrency = (value?: number) => {
    if (value === undefined) return "N/A";
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value?: number) => {
    if (value === undefined) return "N/A";
    return `${value.toFixed(2)}%`;
  };

  const formatRatio = (value?: number) => {
    if (value === undefined) return "N/A";
    return value.toFixed(2);
  };

  const getDSCRColor = (dscr?: number) => {
    if (!dscr) return "gray";
    if (dscr < 1.0) return "red";
    if (dscr < 1.25) return "orange";
    return "green";
  };

  if (isLoading) {
    return (
      <Flex justify="center" align="center" p="8">
        <Text>Calculating underwriting...</Text>
      </Flex>
    );
  }

  if (isError) {
    return (
      <Flex justify="center" align="center" p="8" direction="column" gap="4">
        <Text color="red">
          {(error as any)?.message || "Error calculating underwriting"}
        </Text>
        <Button onClick={() => refetch()}>Retry</Button>
      </Flex>
    );
  }

  if (!underwritingResult) {
    return (
      <Flex justify="center" align="center" p="8">
        <Text color="gray">No underwriting data available</Text>
      </Flex>
    );
  }

  const displayResult = stressTestResult || {
    stressed_noi: underwritingResult.noi,
    stressed_dscr: underwritingResult.dscr,
    stressed_cash_flow: underwritingResult.cash_flow_after_debt,
  };

  return (
    <Flex
      direction="column"
      gap="4"
      p="24px"
      style={{ 
        overflowY: "auto", 
        overflowX: "hidden",
        flex: 1,
        minHeight: 0,
        width: "100%",
        maxWidth: "100%",
      }}
      className="underwriting-dashboard-container"
    >
      <Text size="6" weight="bold">
        Pro Forma Simulator
      </Text>

      {/* Warnings */}
      {underwritingResult.warnings.length > 0 && (
        <Card style={{ background: "#fff3cd", borderColor: "#ffc107" }}>
          <Flex direction="column" gap="2">
            <Text size="3" weight="bold">
              ⚠️ Warnings
            </Text>
            {underwritingResult.warnings.map((warning, idx) => (
              <Text key={idx} size="2">
                • {warning}
              </Text>
            ))}
          </Flex>
        </Card>
      )}

      {/* Key Metrics */}
      <Flex gap="4" wrap="wrap">
        <Card style={{ flex: 1, minWidth: "200px" }}>
          <Flex direction="column" gap="2">
            <Text size="2" color="gray">
              Net Operating Income
            </Text>
            <Text size="7" weight="bold">
              {formatCurrency(displayResult.stressed_noi)}
            </Text>
            {stressTestResult && (
              <Text
                size="2"
                color={
                  stressTestResult.comparison.noi_change >= 0 ? "green" : "red"
                }
              >
                {stressTestResult.comparison.noi_change >= 0 ? "+" : ""}
                {formatCurrency(stressTestResult.comparison.noi_change)} (
                {stressTestResult.comparison.noi_change_pct.toFixed(1)}%)
              </Text>
            )}
          </Flex>
        </Card>

        <Card style={{ flex: 1, minWidth: "200px" }}>
          <Flex direction="column" gap="2">
            <Text size="2" color="gray">
              DSCR
            </Text>
            <Flex align="baseline" gap="2">
              <Text size="7" weight="bold">
                {formatRatio(displayResult.stressed_dscr)}
              </Text>
              <Badge color={getDSCRColor(displayResult.stressed_dscr)}>
                {displayResult.stressed_dscr && displayResult.stressed_dscr >= 1.25
                  ? "Good"
                  : displayResult.stressed_dscr && displayResult.stressed_dscr >= 1.0
                  ? "Fair"
                  : "Poor"}
              </Badge>
            </Flex>
            {stressTestResult && stressTestResult.comparison.dscr_change !== undefined && (
              <Text
                size="2"
                color={
                  stressTestResult.comparison.dscr_change >= 0 ? "green" : "red"
                }
              >
                {stressTestResult.comparison.dscr_change >= 0 ? "+" : ""}
                {stressTestResult.comparison.dscr_change.toFixed(2)}
              </Text>
            )}
          </Flex>
        </Card>

        <Card style={{ flex: 1, minWidth: "200px" }}>
          <Flex direction="column" gap="2">
            <Text size="2" color="gray">
              Cash Flow After Debt
            </Text>
            <Text size="7" weight="bold">
              {formatCurrency(displayResult.stressed_cash_flow)}
            </Text>
            {stressTestResult &&
              stressTestResult.comparison.cash_flow_change !== undefined && (
                <Text
                  size="2"
                  color={
                    stressTestResult.comparison.cash_flow_change >= 0
                      ? "green"
                      : "red"
                  }
                >
                  {stressTestResult.comparison.cash_flow_change >= 0 ? "+" : ""}
                  {formatCurrency(stressTestResult.comparison.cash_flow_change)}
                </Text>
              )}
          </Flex>
        </Card>

        {underwritingResult.cap_rate !== undefined && (
          <Card style={{ flex: 1, minWidth: "200px" }}>
            <Flex direction="column" gap="2">
              <Text size="2" color="gray">
                Cap Rate
              </Text>
              <Text size="7" weight="bold">
                {formatPercent(underwritingResult.cap_rate)}
              </Text>
            </Flex>
          </Card>
        )}

        {underwritingResult.ltv !== undefined && (
          <Card style={{ flex: 1, minWidth: "200px" }}>
            <Flex direction="column" gap="2">
              <Text size="2" color="gray">
                LTV
              </Text>
              <Text size="7" weight="bold">
                {formatPercent(underwritingResult.ltv)}
              </Text>
            </Flex>
          </Card>
        )}
      </Flex>

      {/* Stress Testing */}
      <StressTestPanel
        onScenarioChange={(scenario) => setStressScenario(scenario)}
      />

      {/* Audit Trail */}
      <Card style={{ padding: "20px" }}>
        <Flex direction="column" gap="4">
          <Flex direction="column" gap="1">
            <Text size="4" weight="bold">
              Calculation Audit Trail
            </Text>
            <Text size="2" color="gray">
              Step-by-step breakdown of all underwriting calculations
            </Text>
          </Flex>
          <Flex direction="column" gap="3">
            {underwritingResult.audit_trail.map((step, idx) => (
              <Card key={idx} style={{ background: "#f8f9fa", border: "1px solid #e0e0e0", padding: "16px" }}>
                <Flex direction="column" gap="3">
                  <Text size="4" weight="bold" style={{ color: "#111" }}>
                    {step.metric}
                  </Text>
                  <Text size="2" color="gray" style={{ fontStyle: "italic" }}>
                    Formula: {step.formula}
                  </Text>
                  <Flex direction="column" gap="2" mt="1">
                    <Text size="2" weight="medium" color="gray">
                      Inputs:
                    </Text>
                    {step.inputs.map(([name, value], inputIdx) => (
                      <Flex key={inputIdx} justify="between" align="center" style={{ padding: "4px 0" }}>
                        <Text size="2" style={{ color: "#555" }}>
                          • {name}
                        </Text>
                        <Text size="2" weight="medium">
                          {step.metric.includes("DSCR") && name === "NOI" 
                            ? formatCurrency(value)
                            : step.metric.includes("DSCR")
                            ? formatRatio(value)
                            : formatCurrency(value)}
                        </Text>
                      </Flex>
                    ))}
                  </Flex>
                  <Flex justify="between" align="center" mt="2" p="3" style={{ background: "#fff", borderRadius: "6px", border: "1px solid #e0e0e0" }}>
                    <Text size="3" weight="bold">
                      Result:
                    </Text>
                    <Text size="5" weight="bold" style={{ color: "#16a34a" }}>
                      {step.metric.includes("DSCR") 
                        ? `${formatRatio(step.result)}x`
                        : formatCurrency(step.result)}
                    </Text>
                  </Flex>
                </Flex>
              </Card>
            ))}
          </Flex>
        </Flex>
      </Card>
    </Flex>
  );
};

export default UnderwritingDashboard;

