import { Flex, Text } from "@radix-ui/themes";
import "./Usage.css";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";
import BetterButton from "../BetterButton/BetterButton";
import { useAuth } from "react-oidc-context";
import { getBillingPortalSession } from "../../services/stripeService";
import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { format, subDays, startOfDay } from "date-fns";
import { getTasks } from "../../services/taskApi";
import { useQuery } from "react-query";
import { Status } from "../../models/taskResponse.model";
import Dropdown from "../Dropdown/Dropdown";

interface UsageProps {
  customerId?: string;
}

type TimeRange = "week" | "14days" | "month";

export default function UsagePage({ customerId }: UsageProps) {
  const { data: monthlyUsage, isLoading } = useMonthlyUsage();
  const auth = useAuth();
  const navigate = useNavigate();
  const [isLoadingPortal, setIsLoadingPortal] = useState(false);
  const [timeRange, setTimeRange] = useState<TimeRange>("week");

  const { startDate, endDate } = useMemo(() => {
    const daysToFetch =
      timeRange === "week" ? 7 : timeRange === "14days" ? 14 : 30;
    return {
      endDate: new Date().toISOString(),
      startDate: subDays(startOfDay(new Date()), daysToFetch).toISOString(),
    };
  }, [timeRange]);

  const { data: tasks } = useQuery(
    ["tasks", startDate, endDate],
    () => getTasks(undefined, undefined, startDate, endDate),
    {
      staleTime: 50000,
      refetchInterval:
        new Date(startDate) > new Date(Date.now() - 24 * 60 * 60 * 1000)
          ? 50000
          : false,
      refetchIntervalInBackground: false,
    }
  );

  console.log(tasks);

  const handleManagePayment = async () => {
    if (tier === "Free") {
      navigate("/");
      setTimeout(() => {
        window.location.hash = "pricing";
      }, 100);
      return;
    }

    try {
      setIsLoadingPortal(true);
      const { url } = await getBillingPortalSession(
        auth.user?.access_token || "",
        customerId || ""
      );

      // Redirect to Stripe Customer Portal
      window.location.href = url;
    } catch (error) {
      console.error("Error redirecting to billing portal:", error);
    } finally {
      setIsLoadingPortal(false);
    }
  };

  if (isLoading) {
    return null;
  }

  const usage = monthlyUsage?.[0]?.usage || 0;
  const limit = monthlyUsage?.[0]?.usage_limit || 0;
  const percentage = limit > 0 ? Math.min((usage / limit) * 100, 100) : 0;
  const tier = monthlyUsage?.[0]?.tier || "Free";
  const overage = Math.max(
    0,
    (monthlyUsage?.[0]?.usage || 0) - (monthlyUsage?.[0]?.usage_limit || 0)
  );

  const endDateFormatted = monthlyUsage?.[0]?.billing_cycle_end
    ? new Date(monthlyUsage[0].billing_cycle_end).toLocaleDateString("en-US", {
        month: "long",
        day: "numeric",
        year: "numeric",
      })
    : "N/A";

  const getChartData = () => {
    if (!tasks) return [];

    const dailyPages: {
      [key: string]: { date: string; successful: number; failed: number };
    } = {};

    tasks.forEach((task) => {
      const date = format(new Date(task.created_at), "MMM dd");
      if (!dailyPages[date]) {
        dailyPages[date] = { date, successful: 0, failed: 0 };
      }

      // Get page count from task output
      const pageCount = task.output?.page_count || 0;

      if (task.status === Status.Succeeded) {
        dailyPages[date].successful += pageCount;
      } else if (task.status === Status.Failed) {
        dailyPages[date].failed += pageCount;
      }
    });

    // Sort the data chronologically
    return Object.values(dailyPages).sort(
      (a, b) =>
        new Date(format(new Date(), "yyyy ") + a.date).getTime() -
        new Date(format(new Date(), "yyyy ") + b.date).getTime()
    );
  };

  return (
    <Flex direction="column" className="usage-container" gap="5">
      <Flex direction="row" gap="4" align="center">
        <Text size="5" align="center" weight="bold" style={{ color: "#FFF" }}>
          Current Billing Cycle
        </Text>
        <Flex direction="row" gap="4" className="tier-badge">
          <Text size="2" style={{ color: "rgba(255,255,255,0.9)" }}>
            {tier}
          </Text>
        </Flex>
      </Flex>

      <Flex direction="column" gap="5" mt="1" style={{ flexWrap: "wrap" }}>
        {/* Tier Card */}
        <Flex direction="row" gap="6" style={{ flexWrap: "wrap" }}>
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

          {overage > 0 && (
            <Flex direction="column" gap="4" className="usage-card">
              <Flex justify="between" align="center">
                <Text
                  size="3"
                  weight="bold"
                  style={{ color: "rgba(255,255,255,0.9)" }}
                >
                  Overage
                </Text>
                <Text
                  size="1"
                  weight="medium"
                  style={{ color: "rgba(255,255,255,0.9)" }}
                >
                  {overage.toLocaleString()} Pages
                </Text>
              </Flex>

              <div className="usage-progress-bar">
                <div
                  className="usage-progress-fill"
                  style={{ width: "100%", backgroundColor: "#FF4D4D" }}
                />
              </div>

              <Flex justify="between" align="center">
                <Text size="1" style={{ color: "rgba(255,255,255,0.6)" }}>
                  Above plan limit
                </Text>
                <Flex
                  className="usage-badge"
                  style={{ backgroundColor: "#FF4D4D33" }}
                >
                  <Text size="1" style={{ color: "#FF4D4D" }}>
                    Overage charges apply
                  </Text>
                </Flex>
              </Flex>
            </Flex>
          )}
        </Flex>

        <Flex direction="row" gap="6" style={{ flexWrap: "wrap" }}>
          <Flex direction="column" gap="4" className="usage-card">
            <Flex justify="between" align="center">
              <Text
                size="3"
                weight="bold"
                style={{ color: "rgba(255,255,255,0.9)" }}
              >
                {tier === "Free" ? "Upgrade Plan" : "Payment Status"}
              </Text>
              {tier !== "Free" && (
                <div
                  className={`payment-status-badge ${
                    monthlyUsage?.[0]?.last_paid_status === false
                      ? "status-failed"
                      : "status-success"
                  }`}
                >
                  <Text size="2">
                    {monthlyUsage?.[0]?.last_paid_status === false
                      ? "Failed"
                      : "Paid"}
                  </Text>
                </div>
              )}
            </Flex>

            <Flex direction="column" gap="3">
              <Text size="2" style={{ color: "rgba(255,255,255,0.6)" }}>
                {tier === "Free"
                  ? "Upgrade to a paid plan to unlock higher usage limits and additional features."
                  : monthlyUsage?.[0]?.last_paid_status === false
                  ? "Your last payment was unsuccessful. Please update your payment method."
                  : `Your payment method is up to date. Next bill due ${endDateFormatted}.`}
              </Text>

              <Flex direction="row" gap="2">
                <BetterButton
                  onClick={handleManagePayment}
                  disabled={isLoadingPortal}
                >
                  <Text size="2" className="white">
                    {isLoadingPortal
                      ? "Loading..."
                      : tier === "Free"
                      ? "Upgrade Plan"
                      : "Manage Plan"}
                  </Text>
                </BetterButton>
              </Flex>
            </Flex>
          </Flex>
        </Flex>
      </Flex>

      <Flex direction="row" gap="4" mt="5" align="center">
        <Text size="5" align="center" weight="bold" style={{ color: "#FFF" }}>
          Overview
        </Text>
        <Flex direction="row" gap="4" width="fit-content">
          <Dropdown
            value={
              timeRange === "week"
                ? "Last 7 Days"
                : timeRange === "14days"
                ? "Last 14 Days"
                : "Last 30 Days"
            }
            options={["Last 7 Days", "Last 14 Days", "Last 30 Days"]}
            onChange={(value) => {
              switch (value) {
                case "Last 7 Days":
                  setTimeRange("week");
                  break;
                case "Last 14 Days":
                  setTimeRange("14days");
                  break;
                case "Last 30 Days":
                  setTimeRange("month");
                  break;
              }
            }}
          />
        </Flex>
      </Flex>

      <Flex direction="column" gap="4">
        <Flex justify="between" align="center">
          <Text
            size="3"
            weight="bold"
            style={{ color: "rgba(255,255,255,0.9)" }}
          >
            Pages Processed
          </Text>
        </Flex>
        <Flex className="chart-container">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart
              data={getChartData()}
              margin={{ top: 20, right: 30, left: 0, bottom: 0 }}
            >
              <CartesianGrid
                horizontal={true}
                vertical={false}
                stroke="rgba(255,255,255,0.1)"
                strokeDasharray="3 3"
              />
              <XAxis
                dataKey="date"
                stroke="rgba(255,255,255,0.2)"
                tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                dy={10}
              />
              <YAxis
                stroke="rgba(255,255,255,0.2)"
                tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
                axisLine={false}
                tickLine={false}
                dx={-10}
              />
              <Tooltip
                contentStyle={{
                  background: "rgba(0,0,0,0.9)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "8px",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.5)",
                  padding: "12px",
                }}
                itemStyle={{
                  color: "rgba(255,255,255,0.95)",
                  fontSize: "12px",
                  fontWeight: "bold",
                }}
                labelStyle={{
                  color: "rgba(255,255,255,0.8)",
                  fontSize: "12px",
                  marginBottom: "4px",
                }}
                cursor={{ stroke: "rgba(255,255,255,0.1)", strokeWidth: 1 }}
              />
              <Line
                type="monotone"
                dataKey="successful"
                stroke="#00FF9D"
                strokeWidth={2.5}
                dot={false}
                name="Successfully Processed"
                activeDot={{
                  r: 4,
                  fill: "#00FF9D",
                  stroke: "#00FF9D",
                  strokeWidth: 2,
                }}
              />
              <Line
                type="monotone"
                dataKey="failed"
                stroke="#FF4D4D"
                strokeWidth={2.5}
                dot={false}
                name="Failed"
                activeDot={{
                  r: 4,
                  fill: "#FF4D4D",
                  stroke: "#FF4D4D",
                  strokeWidth: 2,
                }}
              />
            </LineChart>
          </ResponsiveContainer>
        </Flex>
      </Flex>
    </Flex>
  );
}
