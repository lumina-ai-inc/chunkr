interface UsageDetail {
  usage_type: "Fast" | "HighQuality";
  count: number;
  cost: number;
}

interface MonthlyUsage {
  month: string;
  total_cost: number;
  usage_details: UsageDetail[];
}

export type MonthlyUsageData = MonthlyUsage[];
