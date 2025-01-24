interface MonthlyUsage {
  user_id: string;
  email: string;
  last_paid_status: "paid" | "failed" | "cancelled" | null;
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

// export function calculateBillingDueDate(month: string): string {
//   const [year, monthStr] = month.split("-");
//   const currentMonth = new Date(parseInt(year), parseInt(monthStr) - 1, 1);
//   const nextMonth = new Date(currentMonth);
//   nextMonth.setMonth(nextMonth.getMonth() + 1);

//   return nextMonth.toLocaleDateString("en-US", {
//     month: "long",
//     day: "numeric",
//     year: "numeric",
//   });
// }

// export function calculateDiscountedBilling(
//   totalUsage: number,
//   discount: number,
//   totalCost: number
// ): number {
//   if (totalUsage === 0) return 0;

//   const pricePerPage = totalCost / totalUsage;
//   const discountedCost = Math.max(0, totalCost - discount * pricePerPage);

//   return Math.max(0, discountedCost);
// }
