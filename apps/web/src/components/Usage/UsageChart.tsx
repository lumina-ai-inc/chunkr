"use client";

import * as React from "react";
import { Area, AreaChart, CartesianGrid, XAxis } from "recharts";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../../../@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface DailyCount {
  total: number;
  succeeded: number;
  failed: number;
  processing: number;
  starting: number;
  date: string;
}

interface DailyCounts {
  [date: string]: Omit<DailyCount, "date">;
}

interface UsageChartProps {
  data: DailyCounts;
}

const chartConfig = {
  tasks: {
    label: "Tasks",
  },
  succeeded: {
    label: "Succeeded",
    color: "hsl(142.1 76.2% 36.3%)", // #4CAF50
  },
  failed: {
    label: "Failed",
    color: "hsl(4 90% 58%)", // #f44336
  },
  processing: {
    label: "Processing",
    color: "hsl(207 90% 54%)", // #2196F3
  },
  starting: {
    label: "Starting",
    color: "hsl(45 100% 51%)", // #FFC107
  },
} satisfies ChartConfig;

export default function UsageChart({ data }: UsageChartProps) {
  const [timeRange, setTimeRange] = React.useState("14d");

  const transformData = () => {
    const chartData = Object.entries(data).map(([date, counts]) => ({
      date,
      ...counts,
    }));

    // Sort by date
    chartData.sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    // Filter by selected time range
    const referenceDate = new Date();
    const daysToSubtract = parseInt(timeRange);
    const startDate = new Date(referenceDate);
    startDate.setDate(startDate.getDate() - daysToSubtract);

    return chartData.filter((item) => {
      const itemDate = new Date(item.date);
      return itemDate >= startDate;
    });
  };

  return (
    <Card>
      <CardHeader className="flex items-center gap-2 space-y-0 border-b py-5 sm:flex-row">
        <div className="grid flex-1 gap-1 text-center sm:text-left">
          <CardTitle>Task Statistics</CardTitle>
          <CardDescription>Showing task status over time</CardDescription>
        </div>
        <Select value={timeRange} onValueChange={setTimeRange}>
          <SelectTrigger className="w-[160px] rounded-lg sm:ml-auto">
            <SelectValue placeholder="Last 14 days" />
          </SelectTrigger>
          <SelectContent className="rounded-xl">
            <SelectItem value="30d">Last 30 days</SelectItem>
            <SelectItem value="14d">Last 14 days</SelectItem>
            <SelectItem value="7d">Last 7 days</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[300px] w-full"
        >
          <AreaChart data={transformData()}>
            <defs>
              {Object.entries(chartConfig).map(([key, config]) => {
                if (key === "tasks") return null;
                return (
                  <linearGradient
                    key={key}
                    id={`fill${key}`}
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor={config.color}
                      stopOpacity={0.8}
                    />
                    <stop
                      offset="95%"
                      stopColor={config.color}
                      stopOpacity={0.1}
                    />
                  </linearGradient>
                );
              })}
            </defs>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              tickFormatter={(value) => {
                return new Date(value).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                });
              }}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => {
                    return new Date(value).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                    });
                  }}
                  indicator="dot"
                />
              }
            />
            {Object.entries(chartConfig).map(([key, config]) => {
              if (key === "tasks") return null;
              return (
                <Area
                  key={key}
                  dataKey={key}
                  type="natural"
                  fill={`url(#fill${key})`}
                  stroke={config.color}
                  stackId="a"
                />
              );
            })}
            <ChartLegend content={<ChartLegendContent />} />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}
