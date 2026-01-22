import { useQuery } from 'react-query';
import { getDealFacts } from '../../services/dealApi';
import { Flex, Text, Card, Grid } from '@radix-ui/themes';
import { DealTypeBadge } from '../Dashboard/DealTypeBadge';

interface ValueAddMetricsProps {
  dealId: string;
}

function MetricRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <Flex direction="column" gap="4px">
      <Text size="2" style={{ color: "#666" }}>{label}</Text>
      <Text size="4" weight={highlight ? "bold" : "medium"}>{value}</Text>
    </Flex>
  );
}

export function ValueAddMetrics({ dealId }: ValueAddMetricsProps) {
  const { data: facts = [] } = useQuery(
    ['deal-facts', dealId],
    () => getDealFacts(dealId)
  );
  
  // Extract values
  const arv = parseFloat(facts.find(f => f.label === 'ARV (After Repair Value)')?.value || '0');
  const purchasePrice = parseFloat(facts.find(f => f.label === 'Purchase Price')?.value || '0');
  const rehabBudget = parseFloat(facts.find(f => f.label === 'Renovation Budget')?.value || '0');
  const closingCosts = parseFloat(facts.find(f => f.label === 'Closing Costs')?.value || '0');
  const hardCosts = parseFloat(facts.find(f => f.label === 'Hard Costs')?.value || '0');
  const softCosts = parseFloat(facts.find(f => f.label === 'Soft Costs')?.value || '0');
  const constructionTimeline = parseFloat(facts.find(f => f.label === 'Construction Timeline')?.value || '0');
  
  // Calculations
  const allInCost = purchasePrice + rehabBudget + closingCosts;
  const profitMargin = arv > 0 ? ((arv - allInCost) / arv) * 100 : 0;
  const roi = allInCost > 0 ? ((arv - allInCost) / allInCost) * 100 : 0;
  
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value.toFixed(2)}%`;
  };
  
  return (
    <Flex direction="column" gap="24px" p="24px" style={{ overflowY: "auto", flex: 1, minHeight: 0 }}>
      {/* Header with Badge */}
      <Flex justify="between" align="center">
        <Text size="6" weight="bold">Value-Add Analysis</Text>
        <DealTypeBadge dealType="value_add" />
      </Flex>
      
      {/* Pro Forma Section */}
      <Card style={{ padding: "20px" }}>
        <Text size="4" weight="bold" mb="16px">Pro Forma</Text>
        <Grid columns="2" gap="16px">
          <MetricRow label="Purchase Price" value={formatCurrency(purchasePrice)} />
          <MetricRow label="Rehab Budget" value={formatCurrency(rehabBudget)} />
          <MetricRow label="Closing Costs" value={formatCurrency(closingCosts)} />
          <MetricRow label="All-in Cost" value={formatCurrency(allInCost)} highlight />
        </Grid>
      </Card>
      
      {/* Key Metrics */}
      <Card style={{ padding: "20px" }}>
        <Text size="4" weight="bold" mb="16px">Key Metrics</Text>
        <Grid columns="2" gap="16px">
          <MetricRow label="ARV" value={formatCurrency(arv)} highlight />
          <MetricRow label="Profit Margin" value={formatPercent(profitMargin)} />
          <MetricRow label="ROI" value={formatPercent(roi)} highlight />
          <MetricRow label="All-in Cost" value={formatCurrency(allInCost)} />
        </Grid>
      </Card>

      {/* Construction Details */}
      {(hardCosts > 0 || softCosts > 0 || constructionTimeline > 0) && (
        <Card style={{ padding: "20px" }}>
          <Text size="4" weight="bold" mb="16px">Construction Details</Text>
          <Grid columns="2" gap="16px">
            {hardCosts > 0 && <MetricRow label="Hard Costs" value={formatCurrency(hardCosts)} />}
            {softCosts > 0 && <MetricRow label="Soft Costs" value={formatCurrency(softCosts)} />}
            {constructionTimeline > 0 && (
              <MetricRow 
                label="Construction Timeline" 
                value={`${Math.round(constructionTimeline)} days`} 
              />
            )}
          </Grid>
        </Card>
      )}
    </Flex>
  );
}
