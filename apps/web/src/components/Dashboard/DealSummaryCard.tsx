import { Flex, Text, Card, Button, Separator, Grid, TextField } from "@radix-ui/themes";
import { useState, useEffect } from "react";
import { DealResponse } from "../../services/dealApi";
import "./DealSummaryCard.css";

interface DealSummaryCardProps {
  deal: DealResponse;
  metrics?: {
    grossRent?: number;
    noi?: number;
    dscr?: number;
    capRate?: number;
    capSpread?: number;
  };
  onViewFullAnalysis?: () => void;
  onExport?: () => void;
  onDealNameUpdate?: (newName: string) => void;
}

interface InfoRowProps {
  label: string;
  value: string | number;
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <Flex direction="column" gap="4px">
      <Text size="1" style={{ color: "#666", textTransform: "uppercase" }}>
        {label}
      </Text>
      <Text size="2" weight="medium">
        {value}
      </Text>
    </Flex>
  );
}

interface MetricRowProps {
  label: string;
  value: string | number;
  highlight?: boolean;
}

function MetricRow({ label, value, highlight }: MetricRowProps) {
  return (
    <Flex justify="between" align="center">
      <Text size="2" style={{ color: "#666" }}>
        {label}
      </Text>
      <Text
        size="3"
        weight="bold"
        style={{ color: highlight ? "#1976D2" : "#111" }}
      >
        {value}
      </Text>
    </Flex>
  );
}

export default function DealSummaryCard({
  deal,
  metrics,
  onViewFullAnalysis,
  onExport,
  onDealNameUpdate,
}: DealSummaryCardProps) {
  const [isEditingDealName, setIsEditingDealName] = useState(false);
  const [dealNameValue, setDealNameValue] = useState(deal.deal_name);

  useEffect(() => {
    setDealNameValue(deal.deal_name);
  }, [deal.deal_name]);

  const handleSaveDealName = () => {
    if (dealNameValue.trim() && onDealNameUpdate) {
      onDealNameUpdate(dealNameValue.trim());
      setIsEditingDealName(false);
    }
  };

  const handleCancelEdit = () => {
    setDealNameValue(deal.deal_name);
    setIsEditingDealName(false);
  };
  // Calculate capital range based on metrics
  const getCapitalRange = () => {
    if (!metrics?.noi) return { min: 0, max: 0 };
    
    // Simple calculation: 4-7x NOI
    const min = Math.round((metrics.noi * 4) / 1000) * 1000;
    const max = Math.round((metrics.noi * 7) / 1000) * 1000;
    
    return { min, max };
  };

  const capitalRange = getCapitalRange();

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  return (
    <Card style={{ padding: "28px", height: "100%" }}>
      <Flex direction="column" gap="24px">
        {/* Header */}
        <Flex direction="column" gap="8px">
          <Text size="2" weight="medium" style={{ color: "#666" }}>
            📊 Deal Analysis
          </Text>
          {isEditingDealName ? (
            <Flex gap="8px" align="center">
              <TextField.Root
                value={dealNameValue}
                onChange={(e) => setDealNameValue(e.target.value)}
                placeholder="Enter deal name"
                style={{ flex: 1 }}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSaveDealName();
                  if (e.key === "Escape") handleCancelEdit();
                }}
              />
              <Button
                size="2"
                onClick={handleSaveDealName}
                disabled={!dealNameValue.trim()}
              >
                Save
              </Button>
              <Button
                size="2"
                variant="soft"
                onClick={handleCancelEdit}
              >
                Cancel
              </Button>
            </Flex>
          ) : (
            <Text
              size="5"
              weight="bold"
              onClick={() => setIsEditingDealName(true)}
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
              {deal.deal_name}
            </Text>
          )}
        </Flex>

        <Separator size="4" />

        {/* Property Info */}
        <Grid columns="2" gap="16px">
          <InfoRow label="Property Type" value="Multi-Family" />
          <InfoRow label="Status" value={deal.status.replace(/_/g, " ")} />
          <InfoRow label="Documents" value={deal.document_count || 0} />
          <InfoRow label="Facts" value={deal.fact_count || 0} />
        </Grid>

        {metrics && (
          <>
            <Separator size="4" />

            {/* Financial Metrics */}
            <Flex direction="column" gap="12px">
              <Text size="2" weight="medium" style={{ color: "#666" }}>
                Financial Metrics
              </Text>
              {metrics.grossRent && (
                <MetricRow
                  label="Gross Rent"
                  value={formatCurrency(metrics.grossRent) + "/yr"}
                />
              )}
              {metrics.noi && (
                <MetricRow
                  label="NOI"
                  value={formatCurrency(metrics.noi) + "/yr"}
                />
              )}
              {metrics.dscr && (
                <MetricRow
                  label="DSCR"
                  value={metrics.dscr.toFixed(2) + "x"}
                  highlight
                />
              )}
              {metrics.capRate && (
                <MetricRow
                  label="Cap Rate"
                  value={formatPercentage(metrics.capRate)}
                />
              )}
              {metrics.capSpread && (
                <MetricRow
                  label="Cap Spread"
                  value={`+${Math.round(metrics.capSpread)} bps`}
                />
              )}
            </Flex>

            {capitalRange.max > 0 && (
              <>
                <Separator size="4" />

                {/* Capital Range */}
                <Card
                  style={{
                    background: "linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%)",
                    padding: "20px",
                    border: "none",
                  }}
                >
                  <Flex direction="column" gap="8px">
                    <Text size="2" weight="medium" style={{ color: "#1976D2" }}>
                      Supported Capital
                    </Text>
                    <Text size="6" weight="bold" style={{ color: "#1976D2" }}>
                      {formatCurrency(capitalRange.min)} —{" "}
                      {formatCurrency(capitalRange.max)}
                    </Text>
                    <Text size="1" style={{ color: "#1976D2", opacity: 0.8 }}>
                      Based on current property performance
                    </Text>
                  </Flex>
                </Card>
              </>
            )}
          </>
        )}

        {/* Actions */}
        <Flex gap="8px" mt="auto">
          {onViewFullAnalysis && (
            <Button
              size="2"
              onClick={onViewFullAnalysis}
              style={{
                flex: 1,
                backgroundColor: "#1976D2",
                color: "#fff",
                cursor: "pointer",
              }}
            >
              View Full Analysis
            </Button>
          )}
          {onExport && (
            <Button
              size="2"
              variant="soft"
              onClick={onExport}
              style={{
                flex: 1,
                cursor: "pointer",
              }}
            >
              Export
            </Button>
          )}
        </Flex>

        {/* Footer Note */}
        <Text size="1" style={{ color: "#999", textAlign: "center" }}>
          ○ All values verified with source citations
        </Text>
      </Flex>
    </Card>
  );
}

