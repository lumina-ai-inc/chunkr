import { Flex, Text } from "@radix-ui/themes";
import "./Usage.css";
import useUser from "../../hooks/useUser";
import PaymentSetup from "../Payments/PaymentSetup";
// import { useEffect, useState } from "react";
// import { useTasksQuery } from "../../hooks/useTaskQuery";
// import { TaskResponse } from "../../models/taskResponse.model";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";
import { useTasksQuery } from "../../hooks/useTaskQuery";
import BetterButton from "../BetterButton/BetterButton";
import UsageChart from "./UsageChart";

export default function UsagePage() {
  return (
    <Flex direction="column" className="usage-container">
      <Billing />

      <UsageOverview />
    </Flex>
  );
}

export function UsageOverview() {
  const { data: tasks } = useTasksQuery(
    undefined,
    undefined,
    "2025-01-17T00:00:00Z",
    "2025-01-20T23:59:59Z"
  );

  interface DailyCount {
    total: number;
    succeeded: number;
    failed: number;
    processing: number;
    starting: number;
  }

  interface DailyCounts {
    [date: string]: DailyCount;
  }

  const dailyCounts = tasks
    ? tasks.reduce((acc: DailyCounts, task) => {
        const date = task.created_at.split("T")[0];

        if (!acc[date]) {
          acc[date] = {
            total: 0,
            succeeded: 0,
            failed: 0,
            processing: 0,
            starting: 0,
          };
        }

        acc[date].total++;

        switch (task.status) {
          case "Succeeded":
            acc[date].succeeded++;
            break;
          case "Failed":
            acc[date].failed++;
            break;
          case "Processing":
            acc[date].processing++;
            break;
          case "Starting":
            acc[date].starting++;
            break;
        }

        return acc;
      }, {})
    : {};

  return (
    <Flex direction="column" className="account-container" gap="4">
      <Text size="5" weight="bold" style={{ color: "#FFF" }}>
        Usage Overview
      </Text>
      <UsageChart data={dailyCounts} />
    </Flex>
  );
}

export function Billing() {
  const { data: monthlyUsage, isLoading } = useMonthlyUsage();

  if (isLoading) {
    return null; // Or return a loading spinner if you prefer
  }

  const usage = monthlyUsage?.[0]?.usage || 0;
  const limit = monthlyUsage?.[0]?.usage_limit || 0;
  const percentage = limit > 0 ? Math.min((usage / limit) * 100, 100) : 0;
  const tier = monthlyUsage?.[0]?.tier || "Free";
  const overageAmount = monthlyUsage?.[0]?.overage_cost;

  // Format the date
  const startDate = monthlyUsage?.[0]?.billing_cycle_start
    ? new Date(monthlyUsage[0].billing_cycle_start).toLocaleDateString(
        "en-US",
        {
          month: "long",
          day: "numeric",
          year: "numeric",
        }
      )
    : "N/A";

  const endDate = monthlyUsage?.[0]?.billing_cycle_end
    ? new Date(monthlyUsage[0].billing_cycle_end).toLocaleDateString("en-US", {
        month: "long",
        day: "numeric",
        year: "numeric",
      })
    : "N/A";

  return (
    <Flex direction="column" className="billing-container" gap="4">
      <Flex direction="row" gap="4" align="center">
        <Text size="5" align="center" weight="bold" style={{ color: "#FFF" }}>
          Current Billing Cycle
        </Text>
        <Flex direction="row" gap="4" className="tier-badge">
          <Text size="2" style={{ color: "rgba(255,255,255,0.9)" }}>
            {tier} Plan
          </Text>
        </Flex>
      </Flex>

      <Text size="2" style={{ color: "#FFF" }}>
        View your usage and manage billing information + plans.
      </Text>
      <Flex direction="column" gap="4" style={{ flexWrap: "wrap" }}>
        {/* Tier Card */}
        <Flex direction="row" gap="4" style={{ flexWrap: "wrap" }}>
          <Flex direction="column" gap="4" className="usage-card">
            <Flex justify="between" align="center">
              <Text
                size="3"
                weight="bold"
                style={{ color: "rgba(255,255,255,0.9)" }}
              >
                Billing Dates
              </Text>
            </Flex>

            <Flex direction="column" gap="2">
              <Flex justify="between" align="center">
                <Text size="2" style={{ color: "rgba(255,255,255,0.6)" }}>
                  Start
                </Text>
                <Text size="2" style={{ color: "rgba(255,255,255,0.9)" }}>
                  {startDate}
                </Text>
              </Flex>
              <Flex justify="between" align="center">
                <Text size="2" style={{ color: "rgba(255,255,255,0.6)" }}>
                  End
                </Text>
                <Text size="2" style={{ color: "rgba(255,255,255,0.9)" }}>
                  {endDate}
                </Text>
              </Flex>
            </Flex>
          </Flex>
          <Flex direction="column" gap="4" className="usage-card">
            <Flex justify="between" align="center">
              <Text
                size="3"
                weight="bold"
                style={{ color: "rgba(255,255,255,0.9)" }}
              >
                Credits
              </Text>
              <Text
                size="1"
                weight="medium"
                style={{ color: "rgba(255,255,255,0.9)" }}
              >
                {usage.toLocaleString()} / {limit.toLocaleString()} Pages
              </Text>
            </Flex>

            <div className="usage-progress-bar">
              <div
                className="usage-progress-fill"
                style={{ width: `${percentage}%` }}
              />
            </div>

            <Flex justify="between" align="center">
              <Text size="1" style={{ color: "rgba(255,255,255,0.6)" }}>
                {percentage.toFixed(1)}% used
              </Text>
              <Flex className="usage-badge">
                <Text size="1" style={{ color: "rgba(255,255,255,0.8)" }}>
                  {limit - usage > 0
                    ? `${(limit - usage).toLocaleString()} pages remaining`
                    : "Limit reached"}
                </Text>
              </Flex>
            </Flex>
          </Flex>
        </Flex>

        <Flex direction="column" gap="4" className="usage-card">
          <Flex justify="between" align="center">
            <Text
              size="3"
              weight="bold"
              style={{ color: "rgba(255,255,255,0.9)" }}
            >
              Cost Breakdown
            </Text>
          </Flex>

          <Flex direction="column" gap="3">
            <Flex justify="between" align="center">
              <Text size="2" style={{ color: "rgba(255,255,255,0.6)" }}>
                Subscription Cost
              </Text>
              <Text size="2" style={{ color: "rgba(255,255,255,0.9)" }}>
                ${monthlyUsage?.[0]?.subscription_cost.toFixed(2)}
              </Text>
            </Flex>

            <Flex justify="between" align="center">
              <Text size="2" style={{ color: "rgba(255,255,255,0.6)" }}>
                Overage Cost
              </Text>
              <Text size="2" style={{ color: "rgba(255,255,255,0.9)" }}>
                ${monthlyUsage?.[0]?.overage_cost.toFixed(2)}
              </Text>
            </Flex>

            <Flex
              justify="between"
              align="center"
              className="total-cost-section"
            >
              <Text
                size="2"
                weight="bold"
                style={{ color: "rgba(255,255,255,0.8)" }}
              >
                Total Cost
              </Text>
              <Text
                size="4"
                weight="bold"
                style={{ color: "rgba(255,255,255,0.9)" }}
              >
                $
                {(
                  (monthlyUsage?.[0]?.subscription_cost || 0) +
                  (monthlyUsage?.[0]?.overage_cost || 0)
                ).toFixed(2)}
              </Text>
            </Flex>
          </Flex>
        </Flex>

        {monthlyUsage?.[0]?.tier !== "Free" && (
          <Flex direction="column" gap="4" className="usage-card">
            <Flex justify="between" align="center">
              <Text
                size="3"
                weight="bold"
                style={{ color: "rgba(255,255,255,0.9)" }}
              >
                Payment Status
              </Text>
              <div
                className={`payment-status-badge ${
                  monthlyUsage?.[0]?.last_paid_status === "failed"
                    ? "status-failed"
                    : monthlyUsage?.[0]?.last_paid_status === "cancelled"
                      ? "status-cancelled"
                      : "status-success"
                }`}
              >
                <Text size="2">
                  {monthlyUsage?.[0]?.last_paid_status === "failed"
                    ? "Failed"
                    : monthlyUsage?.[0]?.last_paid_status === "cancelled"
                      ? "Cancelled"
                      : "Paid"}
                </Text>
              </div>
            </Flex>

            <Flex direction="column" gap="3">
              <Text size="2" style={{ color: "rgba(255,255,255,0.6)" }}>
                {monthlyUsage?.[0]?.last_paid_status === "failed"
                  ? "Your last payment was unsuccessful. Please update your payment method."
                  : monthlyUsage?.[0]?.last_paid_status === "cancelled"
                    ? "Your subscription has been cancelled."
                    : "Your payment method is up to date."}
              </Text>

              <Flex direction="row" gap="2">
                <BetterButton
                  active={
                    monthlyUsage?.[0]?.last_paid_status === "failed"
                      ? true
                      : false
                  }
                  onClick={() => {
                    /* Add your payment management logic */
                  }}
                >
                  <Text size="2">Manage Payment Method</Text>
                </BetterButton>
              </Flex>
            </Flex>
          </Flex>
        )}
      </Flex>
    </Flex>
  );
}

export function PaymentMethod() {
  return (
    <Flex direction="column" className="payment-method-container">
      <Text size="5" weight="bold" style={{ color: "#FFF" }}>
        Payment Method
      </Text>
      <PaymentSetup />
    </Flex>
  );
}
