interface MonthlyUsage {
  user_id: string;
  email: string;
  last_paid_status?: boolean;
  month: string;
  subscription_cost: number;
  usage_limit: number;
  usage: number;
  overage_cost: number;
  tier: string;
  billing_cycle_start: string;
  billing_cycle_end: string;
}

export type MonthlyUsageData = MonthlyUsage[];
