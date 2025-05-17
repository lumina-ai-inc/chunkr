import { Flex, Text } from "@radix-ui/themes";
import "./Usage.css";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";
import BetterButton from "../BetterButton/BetterButton";
import { useAuth } from "react-oidc-context";
import { getBillingPortalSession } from "../../services/stripeService";
import { useState, useMemo } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  BarChart,
  Bar,
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
import toast from "react-hot-toast";

interface UsageProps {
  customerId?: string;
}

type TimeRange = "today" | "week" | "14days" | "month";

export default function UsagePage({ customerId }: UsageProps) {
  const { data: monthlyUsage, isLoading } = useMonthlyUsage();
  const auth = useAuth();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [timeRange, setTimeRange] = useState<TimeRange>(
    (searchParams.get("timeRange") as TimeRange) || "week"
  );
  const [isLoadingPortal, setIsLoadingPortal] = useState(false);

  const { startDate, endDate } = useMemo(() => {
    if (timeRange === "today") {
      return {
        startDate: startOfDay(new Date()).toISOString(),
        endDate: new Date().toISOString(),
      };
    }
    const daysToFetch =
      timeRange === "week" ? 7 : timeRange === "14days" ? 14 : 30;
    return {
      endDate: new Date().toISOString(),
      startDate: subDays(startOfDay(new Date()), daysToFetch).toISOString(),
    };
  }, [timeRange]);

  const { data: tasks, refetch: refetchTasks } = useQuery(
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
      toast.error(
        "Failed to open billing portal - refresh page and try again."
      );
    } finally {
      setIsLoadingPortal(false);
    }
  };

  const handleTimeRangeChange = (value: string) => {
    const newTimeRange = (() => {
      switch (value) {
        case "Today":
          return "today";
        case "Last 7 Days":
          return "week";
        case "Last 14 Days":
          return "14days";
        case "Last 30 Days":
          return "month";
        default:
          return "week";
      }
    })();

    // Preserve existing params while updating timeRange
    const params = new URLSearchParams(searchParams);
    params.set("timeRange", newTimeRange);
    params.set("view", "usage"); // Ensure we're marking this as the usage view
    setSearchParams(params);
    setTimeRange(newTimeRange);
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

    if (timeRange === "today") {
      // Handle hourly data for today
      const hourlyPages: {
        [key: string]: {
          hour: string;
          successful: number;
          failed: number;
          processing: number;
          starting: number;
        };
      } = {};

      // Initialize all hours
      for (let i = 0; i < 24; i++) {
        const hour = i.toString().padStart(2, "0");
        hourlyPages[hour] = {
          hour: `${hour}:00`,
          successful: 0,
          failed: 0,
          processing: 0,
          starting: 0,
        };
      }

      tasks.forEach((task) => {
        const hour = format(new Date(task.created_at), "HH");
        const pageCount = task.output?.page_count || 0;

        switch (task.status) {
          case Status.Succeeded:
            hourlyPages[hour].successful += pageCount;
            break;
          case Status.Failed:
            hourlyPages[hour].failed += pageCount;
            break;
          case Status.Processing:
            hourlyPages[hour].processing += pageCount;
            break;
          case Status.Starting:
            hourlyPages[hour].starting += pageCount;
            break;
        }
      });

      return Object.values(hourlyPages);
    }

    // Handle daily data
    const dailyPages: {
      [key: string]: {
        date: string;
        successful: number;
        failed: number;
        processing: number;
        starting: number;
      };
    } = {};

    tasks.forEach((task) => {
      const date = format(new Date(task.created_at), "MMM dd");
      if (!dailyPages[date]) {
        dailyPages[date] = {
          date,
          successful: 0,
          failed: 0,
          processing: 0,
          starting: 0,
        };
      }

      const pageCount = task.output?.page_count || 0;

      switch (task.status) {
        case Status.Succeeded:
          dailyPages[date].successful += pageCount;
          break;
        case Status.Failed:
          dailyPages[date].failed += pageCount;
          break;
        case Status.Processing:
          dailyPages[date].processing += pageCount;
          break;
        case Status.Starting:
          dailyPages[date].starting += pageCount;
          break;
      }
    });

    return Object.values(dailyPages).sort(
      (a, b) =>
        new Date(format(new Date(), "yyyy ") + a.date).getTime() -
        new Date(format(new Date(), "yyyy ") + b.date).getTime()
    );
  };

  // Helper to aggregate daily data for the stacked bar chart
  const getBarChartData = () => {
    if (!tasks) return [];
    const dailyPages: {
      [key: string]: {
        date: string;
        successful: number;
        failed: number;
        processing: number;
        starting: number;
      };
    } = {};

    tasks.forEach((task) => {
      const date = format(new Date(task.created_at), "MMM dd");
      if (!dailyPages[date]) {
        dailyPages[date] = {
          date,
          successful: 0,
          failed: 0,
          processing: 0,
          starting: 0,
        };
      }
      // Count one per task rather than summing pages
      const taskCount = 1;
      switch (task.status) {
        case Status.Succeeded:
          dailyPages[date].successful += taskCount;
          break;
        case Status.Failed:
          dailyPages[date].failed += taskCount;
          break;
        case Status.Processing:
          dailyPages[date].processing += taskCount;
          break;
        case Status.Starting:
          dailyPages[date].starting += taskCount;
          break;
      }
    });

    return Object.values(dailyPages).sort(
      (a, b) =>
        new Date(format(new Date(), "yyyy ") + a.date).getTime() -
        new Date(format(new Date(), "yyyy ") + b.date).getTime()
    );
  };

  return (
    <Flex direction="column" className="usage-container" gap="5">
      <Flex direction="row" gap="4" align="center">
        <Flex direction="row" align="center" justify="between" width="100%">
          <Flex direction="row" align="center" gap="4">
            <Text
              size="5"
              align="center"
              weight="bold"
              style={{ color: "#FFF" }}
            >
              Overview
            </Text>
            <BetterButton onClick={() => refetchTasks()}>
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M21.25 12C21.25 17.1086 17.1086 21.25 12 21.25C6.89137 21.25 2.75 17.1086 2.75 12C2.75 6.89137 6.89137 2.75 12 2.75C15.0183 2.75 17.7158 4.17505 19.4317 6.37837"
                  stroke="#FFFFFF"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M21.25 2.75V6.375C21.25 6.92728 20.8023 7.375 20.25 7.375H16.625"
                  stroke="#FFFFFF"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <Text size="2" className="white">
                Refresh
              </Text>
            </BetterButton>
          </Flex>
          <Flex direction="row" gap="4" width="fit-content">
            <Dropdown
              value={
                timeRange === "today"
                  ? "Today"
                  : timeRange === "week"
                  ? "Last 7 Days"
                  : timeRange === "14days"
                  ? "Last 14 Days"
                  : "Last 30 Days"
              }
              options={["Today", "Last 7 Days", "Last 14 Days", "Last 30 Days"]}
              onChange={handleTimeRangeChange}
            />
          </Flex>
        </Flex>
      </Flex>

      <Flex direction="row" gap="4" width="100%">
        <Flex direction="column" gap="2" style={{ flex: 1 }}>
          <Flex direction="row" gap="1" align="center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="25"
              height="24"
              fill="none"
              viewBox="0 0 25 24"
            >
              <g
                stroke="#FFF"
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="1.5"
                clipPath="url(#gRimTy0T6Ra)"
              >
                <path d="M9.25 6.75h6.5m-7.75 9h11.75v5.5H8c-1.52 0-2.75-1.23-2.75-2.75S6.48 15.75 8 15.75" />
                <path d="M5.25 18.5V5.75a3 3 0 0 1 3-3h10.5a1 1 0 0 1 1 1V16" />
              </g>
              <defs>
                <clipPath id="gRimTy0T6Ra">
                  <path fill="#fff" d="M.5 0h24v24H.5z" />
                </clipPath>
              </defs>
            </svg>

            <Text
              size="3"
              weight="bold"
              style={{ color: "rgba(255,255,255,0.9)" }}
            >
              Tasks Status Per Day (UTC)
            </Text>
          </Flex>
          <Flex className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={getBarChartData()}
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
                  allowDecimals={false}
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
                  cursor={false}
                />
                <Bar
                  stackId="a"
                  dataKey="starting"
                  fill="#4D9EFF"
                  name="Starting"
                />
                <Bar
                  stackId="a"
                  dataKey="processing"
                  fill="#FFB800"
                  name="Processing"
                />
                <Bar
                  stackId="a"
                  dataKey="failed"
                  fill="#FF4D4D"
                  name="Failed"
                />
                <Bar
                  stackId="a"
                  dataKey="successful"
                  fill="#00FF9D"
                  name="Successfully Processed"
                />
              </BarChart>
            </ResponsiveContainer>
          </Flex>
        </Flex>
        <Flex direction="column" gap="2" style={{ flex: 1 }}>
          <Flex direction="row" gap="1" align="center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="25"
              height="24"
              fill="none"
              viewBox="0 0 25 24"
            >
              <g
                stroke="#FFF"
                strokeLinecap="round"
                strokeWidth="1.5"
                clipPath="url(#WyWBQHSz7Ta)"
              >
                <path
                  strokeLinejoin="round"
                  d="M18.75 2.75H6.25a1 1 0 0 0-1 1v16.5a1 1 0 0 0 1 1h12.5a1 1 0 0 0 1-1V3.75a1 1 0 0 0-1-1"
                />
                <path strokeLinejoin="round" d="M12.75 5.75h-4.5v4.5h4.5z" />
                <path strokeMiterlimit="10" d="M8.25 13.75h8.5m-8.5 3.5h4.5" />
              </g>
              <defs>
                <clipPath id="WyWBQHSz7Ta">
                  <path fill="#fff" d="M.5 0h24v24H.5z" />
                </clipPath>
              </defs>
            </svg>
            <Text
              size="3"
              weight="bold"
              style={{ color: "rgba(255,255,255,0.9)" }}
            >
              Pages Processed Per Day (UTC)
            </Text>
          </Flex>
          <Flex className="chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
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
                  dataKey={timeRange === "today" ? "hour" : "date"}
                  stroke="rgba(255,255,255,0.2)"
                  tick={{ fill: "rgba(255,255,255,0.6)", fontSize: 12 }}
                  axisLine={false}
                  tickLine={false}
                  dy={10}
                />
                <YAxis
                  allowDecimals={false}
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
                  cursor={false}
                />
                <Bar
                  stackId="a"
                  dataKey="starting"
                  fill="#4D9EFF"
                  name="Starting"
                />
                <Bar
                  stackId="a"
                  dataKey="processing"
                  fill="#FFB800"
                  name="Processing"
                />
                <Bar
                  stackId="a"
                  dataKey="failed"
                  fill="#FF4D4D"
                  name="Failed"
                />
                <Bar
                  stackId="a"
                  dataKey="successful"
                  fill="#00FF9D"
                  name="Successfully Processed"
                />
              </BarChart>
            </ResponsiveContainer>
          </Flex>
        </Flex>
      </Flex>
      {tier !== "SelfHosted" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          <Flex direction="row" gap="4" mt="56px" align="center">
            <Text
              size="5"
              align="center"
              weight="bold"
              style={{ color: "#FFF" }}
            >
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
              <Flex
                direction="column"
                gap="4"
                className="usage-card"
                justify="between"
              >
                <Flex justify="between" align="center">
                  <Flex direction="row" gap="2" align="center">
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <g clip-path="url(#clip0_305_27868)">
                        <path
                          d="M12 21.25C17.1086 21.25 21.25 17.1086 21.25 12C21.25 6.89137 17.1086 2.75 12 2.75C6.89137 2.75 2.75 6.89137 2.75 12C2.75 17.1086 6.89137 21.25 12 21.25Z"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeMiterlimit="10"
                        />
                        <path
                          d="M9.88012 14.36C9.88012 15.53 10.8301 16.25 12.0001 16.25C13.1701 16.25 14.1201 15.53 14.1201 14.36C14.1201 13.19 13.3501 12.75 11.5301 11.66C10.6701 11.15 9.87012 10.82 9.87012 9.64C9.87012 8.46 10.8201 7.75 11.9901 7.75C13.1601 7.75 14.1101 8.7 14.1101 9.87"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M12 16.25V18.25"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M12 5.75V7.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </g>
                      <defs>
                        <clipPath id="clip0_305_27868">
                          <rect width="24" height="24" fill="white" />
                        </clipPath>
                      </defs>
                    </svg>
                    <Text
                      size="3"
                      weight="bold"
                      style={{ color: "rgba(255,255,255,0.9)" }}
                    >
                      Credits
                    </Text>
                  </Flex>

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

              <Flex
                direction="column"
                gap="4"
                className="usage-card"
                justify="between"
              >
                <Flex justify="between" align="center">
                  <Flex direction="row" gap="2" align="center">
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 25 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <g clip-path="url(#clip0_113_1447)">
                        <path
                          d="M12.5 6.25C17.61 6.25 21.75 10.39 21.75 15.5V18.25H3.25V15.5C3.25 10.39 7.39 6.25 12.5 6.25Z"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <circle cx="12.5" cy="14.75" r="1.25" fill="#121331" />
                        <path
                          d="M12.5 8.31V6.25"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M5.49979 12.8799L3.77979 12.4199"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M21.22 12.4199L19.5 12.8799"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M9.25 13.75L12.4999 14.7501"
                          stroke="#121331"
                          stroke-width="1.5"
                          stroke-linecap="round"
                          stroke-linejoin="round"
                        />
                      </g>
                      <defs>
                        <clipPath id="clip0_113_1447">
                          <rect
                            width="24"
                            height="24"
                            fill="white"
                            transform="translate(0.5)"
                          />
                        </clipPath>
                      </defs>
                    </svg>
                    <Text
                      size="3"
                      weight="bold"
                      style={{ color: "rgba(255,255,255,0.9)" }}
                    >
                      Overage
                    </Text>
                  </Flex>
                  <Text
                    size="1"
                    weight="medium"
                    style={{ color: "rgba(255,255,255,0.9)" }}
                  >
                    {overage.toLocaleString()} Pages
                  </Text>
                </Flex>

                <Text
                  size="2"
                  weight="bold"
                  style={{ color: "rgba(255,255,255,0.9)" }}
                >
                  {monthlyUsage?.[0]?.overage_cost
                    ? `$${Number(monthlyUsage[0].overage_cost).toLocaleString(
                        "en-US",
                        {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        }
                      )}`
                    : "$0.00"}
                </Text>

                {overage === 0 && (
                  <Flex justify="between" align="center">
                    <Text size="1" style={{ color: "rgba(255,255,255,0.6)" }}>
                      No overage incurred
                    </Text>
                    <Flex
                      className="usage-badge"
                      style={{
                        backgroundColor: "rgba(0, 255, 157, 0.1)",
                        border: "1px solid rgba(0, 255, 157, 0.2)",
                      }}
                    >
                      <Text size="1" style={{ color: "#00ff9d" }}>
                        Within your plan
                      </Text>
                    </Flex>
                  </Flex>
                )}

                {overage > 0 && (
                  <Flex justify="between" align="center">
                    <Text size="1" style={{ color: "rgba(255,255,255,0.6)" }}>
                      Plan credits used
                    </Text>
                    <Flex
                      className="usage-badge"
                      style={{ backgroundColor: "#FF4D4D33" }}
                    >
                      <Text size="1" style={{ color: "#ff824d" }}>
                        Overage charges apply
                      </Text>
                    </Flex>
                  </Flex>
                )}
              </Flex>
            </Flex>

            <Flex direction="row" gap="6" style={{ flexWrap: "wrap" }}>
              <Flex direction="column" gap="4" className="usage-card">
                <Flex justify="between" align="center">
                  <Flex direction="row" gap="2" align="center">
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 25 24"
                      fill="none"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <g clip-path="url(#clip0_113_1389)">
                        <path
                          d="M7.25 9V16.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M12.5 9V16.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M17.75 9V16.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M5.25 16.75H19.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeMiterlimit="10"
                          strokeLinecap="round"
                        />
                        <path
                          d="M4.25 20.25H20.75"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeMiterlimit="10"
                          strokeLinecap="round"
                        />
                        <path
                          d="M4.25 5.75L12.5 3.75L20.75 5.75V8.25H4.25V5.75Z"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                          strokeLinejoin="round"
                        />
                      </g>
                      <defs>
                        <clipPath id="clip0_113_1389">
                          <rect
                            width="24"
                            height="24"
                            fill="white"
                            transform="translate(0.5)"
                          />
                        </clipPath>
                      </defs>
                    </svg>
                    <Text
                      size="3"
                      weight="bold"
                      style={{ color: "rgba(255,255,255,0.9)" }}
                    >
                      {tier === "Free" ? "Upgrade Plan" : "Payment Status"}
                    </Text>
                  </Flex>
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
                      ? "Upgrade to a paid plan to unlock higher usage limits."
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
                          : "Manage Billing"}
                      </Text>
                    </BetterButton>
                  </Flex>
                </Flex>
              </Flex>
            </Flex>
          </Flex>
        </div>
      )}
    </Flex>
  );
}
