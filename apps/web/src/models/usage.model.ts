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
  fastUsage: number,
  highQualityUsage: number,
  fastDiscount: number,
  highQualityDiscount: number,
  fastCost: number,
  highQualityCost: number
): number {
  const fastPricePerPage = fastUsage > 0 ? fastCost / fastUsage : 0;
  const highQualityPricePerPage =
    highQualityUsage > 0 ? highQualityCost / highQualityUsage : 0;

  const discountedFastCost = Math.max(
    0,
    fastCost - fastDiscount * fastPricePerPage
  );
  const discountedHighQualityCost = Math.max(
    0,
    highQualityCost - highQualityDiscount * highQualityPricePerPage
  );

  const totalDiscountedCost = discountedFastCost + discountedHighQualityCost;

  return Math.max(0, totalDiscountedCost);
}
