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

export function calculateBillingDueDate(month: string): string {
  const [year, monthStr] = month.split("-");
  const currentMonth = new Date(parseInt(year), parseInt(monthStr) - 1, 1);
  const nextMonth = new Date(currentMonth);
  nextMonth.setMonth(nextMonth.getMonth() + 1);

  return nextMonth.toLocaleDateString("en-US", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}

export function calculateDiscountedBilling(
  totalUsage: number,
  discount: number,
  totalCost: number
): number {
  if (totalUsage === 0) return 0;

  const pricePerPage = totalCost / totalUsage;
  const discountedCost = Math.max(0, totalCost - discount * pricePerPage);

  return Math.max(0, discountedCost);
}
