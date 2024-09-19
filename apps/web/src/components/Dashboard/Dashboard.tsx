import { useEffect, useState } from "react";
import { Text, Flex, ScrollArea } from "@radix-ui/themes";
import "./Dashboard.css";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";
import TaskCard from "../TaskCard/TaskCard";
import DashBoardHeader from "./DashBoardHeader";
import Loader from "../../pages/Loader/Loader";
import { useTasksQuery } from "../../hooks/useTaskQuery";
import { useNavigate } from "react-router-dom";
import { TaskResponse } from "../../models/task.model";
import ApiKeyDialog from "../ApiDialog.tsx/ApiKeyDialog";
import {
  calculateBillingDueDate,
  calculateDiscountedBilling,
  MonthlyUsageData,
} from "../../models/usage.model";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";

export default function Dashboard() {
  const [showApiKey, setShowApiKey] = useState(false);
  const [monthlyUsage, setMonthlyUsage] = useState<MonthlyUsageData>([]);

  const user = useSelector((state: RootState) => state.user.data);
  const navigate = useNavigate();

  const { data: monthlyUsageData } = useMonthlyUsage();

  useEffect(() => {
    if (monthlyUsageData) {
      setMonthlyUsage(monthlyUsageData);
    }
  }, [monthlyUsageData]);

  const { data: tasks, isLoading, isError } = useTasksQuery(1, 10);

  if (!user) {
    return <Loader />;
  }

  if (!monthlyUsage) {
    return <Loader />;
  }

  const fastUsage =
    monthlyUsage?.[0]?.usage_details.find(
      (usage) => usage.usage_type === "Fast"
    )?.count || 0;

  const highQualityUsage =
    monthlyUsage?.[0]?.usage_details.find(
      (usage) => usage.usage_type === "HighQuality"
    )?.count || 0;

  const fastDiscount =
    user?.usage?.find((u) => u.usage_type === "Fast")?.discounts?.[0]?.amount ||
    0;
  const highQualityDiscount =
    user?.usage?.find((u) => u.usage_type === "HighQuality")?.discounts?.[0]
      ?.amount || 0;

  const fastDiscountedUsage = Math.max(0, fastDiscount - fastUsage);
  const highQualityDiscountedUsage = Math.max(
    0,
    highQualityDiscount - highQualityUsage
  );

  const fastCost =
    monthlyUsage?.[0]?.usage_details.find((u) => u.usage_type === "Fast")
      ?.cost || 0;
  const highQualityCost =
    monthlyUsage?.[0]?.usage_details.find((u) => u.usage_type === "HighQuality")
      ?.cost || 0;

  const adjustedBillingAmount = calculateDiscountedBilling(
    fastUsage,
    highQualityUsage,
    fastDiscount,
    highQualityDiscount,
    fastCost,
    highQualityCost
  );

  // const adjustedFastUsage = Math.max(0, fastUsage - fastDiscount);
  // const adjustedHighQualityUsage = Math.max(
  //   0,
  //   highQualityUsage - highQualityDiscount
  // );

  // const fastCost =
  //   monthlyUsage?.[0]?.usage_details.find((u) => u.usage_type === "Fast")
  //     ?.cost || 0;
  // const highQualityCost =
  //   monthlyUsage?.[0]?.usage_details.find((u) => u.usage_type === "HighQuality")
  //     ?.cost || 0;

  // const fastCostPerPage = fastUsage > 0 ? fastCost / fastUsage : 0;
  // const highQualityCostPerPage =
  //   highQualityUsage > 0 ? highQualityCost / highQualityUsage : 0;

  // const adjustedBillingAmount = Math.max(
  //   0,
  //   fastCost -
  //     fastDiscount * fastCostPerPage +
  //     (highQualityCost - highQualityDiscount * highQualityCostPerPage)
  // );

  const billingAmount = monthlyUsage?.[0]?.total_cost || 0;
  console.log(billingAmount);

  const billingDueDate = monthlyUsage?.[0]?.month
    ? calculateBillingDueDate(monthlyUsage[0].month)
    : null;

  const handleTaskClick = (task: TaskResponse) => {
    navigate(`/task/${task.task_id}?pageCount=${task.page_count}`);
  };

  return (
    <div className="dashboard-container">
      <Flex className="dashboard-header">
        <DashBoardHeader {...user} />
      </Flex>
      <ScrollArea scrollbars="vertical" style={{ height: "100%" }}>
        <Flex direction="row" className="dashboard-content">
          <Flex direction="column" className="dashboard-content-left">
            <Text size="9" weight="bold" style={{ color: "var(--cyan-1)" }}>
              Dashboard
            </Text>
            <Text
              size="6"
              mt="2"
              mb="4"
              weight="medium"
              style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
            >
              {user?.first_name} {user?.last_name}
            </Text>
            <ApiKeyDialog
              user={user}
              showApiKey={showApiKey}
              setShowApiKey={setShowApiKey}
            />
            {user?.tier === "Free" && (
              <Flex className="callout">
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "var(--amber-4)" }}
                >
                  Add a payment method
                </Text>

                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="16"
                  height="16"
                  viewBox="0 0 16 16"
                  fill="none"
                >
                  <rect
                    width="16"
                    height="16"
                    fill="white"
                    fill-opacity="0.01"
                  />
                  <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M8.68955 3.35623C8.89782 3.14794 9.23551 3.14794 9.44378 3.35623L13.7105 7.6229C13.9187 7.83117 13.9187 8.16886 13.7105 8.37713L9.44378 12.6439C9.23551 12.8521 8.89782 12.8521 8.68955 12.6439C8.48126 12.4355 8.48126 12.0978 8.68955 11.8895L12.0458 8.53335H2.66666C2.37212 8.53335 2.13333 8.29456 2.13333 8.00002C2.13333 7.70547 2.37212 7.46668 2.66666 7.46668H12.0458L8.68955 4.11047C8.48126 3.90219 8.48126 3.56451 8.68955 3.35623Z"
                    fill="#FFEE9C"
                  />
                </svg>
              </Flex>
            )}
            <Flex direction="row" gap="6" justify="between" mt="5">
              <Flex
                flexGrow="1"
                direction="column"
                p="5"
                style={{
                  border: "2px solid hsla(180, 100%, 100%, 0.1)",
                  borderRadius: "8px",
                }}
              >
                <Flex direction="row" justify="between">
                  <Flex
                    direction="row"
                    gap="2"
                    align="center"
                    style={{
                      padding: "6px 12px",
                      borderRadius: "4px",
                      width: "fit-content",
                      border: "1px solid hsla(180, 100%, 100%, 0.1)",
                      backgroundColor: "hsla(180, 100%, 100%, 0.05)",
                    }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 16 16"
                      fill="none"
                    >
                      <rect
                        width="16"
                        height="16"
                        fill="white"
                        fill-opacity="0.01"
                      />
                      <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M9.27645 0.0430443C9.5025 0.139774 9.63313 0.378514 9.5927 0.621051L8.62955 6.40001H13.3333C13.5353 6.40001 13.72 6.51414 13.8103 6.69484C13.9007 6.87552 13.8812 7.09173 13.7599 7.25334L7.35997 15.7867C7.21245 15.9834 6.94956 16.0537 6.7235 15.957C6.49744 15.8603 6.36681 15.6215 6.40723 15.379L7.37039 9.6H2.66666C2.46465 9.6 2.27998 9.48587 2.18963 9.30518C2.0993 9.1245 2.11878 8.90828 2.23999 8.74667L8.63997 0.213374C8.7875 0.0166672 9.05039 -0.0536855 9.27645 0.0430443ZM3.73332 8.53334H7.99997C8.15675 8.53334 8.30558 8.60231 8.40691 8.72194C8.50825 8.84156 8.55182 8.99971 8.52605 9.15435L7.81895 13.3969L12.2667 7.46668H7.99997C7.8432 7.46668 7.69437 7.3977 7.59302 7.27808C7.49169 7.15845 7.44812 7.00031 7.4739 6.84566L8.18099 2.60311L3.73332 8.53334Z"
                        fill="hsla(180, 100%, 100%, 1)"
                      />
                    </svg>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "hsla(180, 100%, 100%, 1)" }}
                    >
                      Fast ($0.005/page)
                    </Text>
                  </Flex>
                  {user?.tier === "PayAsYouGo" && fastDiscountedUsage < 0 && (
                    <Flex
                      direction="row"
                      gap="2"
                      align="center"
                      style={{
                        padding: "6px 12px",
                        borderRadius: "4px",
                        width: "fit-content",
                        border: "1px solid hsla(180, 100%, 100%, 0.1)",
                        backgroundColor: "hsla(180, 100%, 100%, 0.9)",
                      }}
                    >
                      <Text
                        size="2"
                        weight="medium"
                        style={{ color: "hsla(0, 0%, 0%, 1)" }}
                      >
                        {fastDiscount} free pages
                      </Text>
                    </Flex>
                  )}
                </Flex>

                {user?.tier === "PayAsYouGo" && (
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    {fastUsage}
                    <Text
                      size="4"
                      weight="medium"
                      style={{ color: "hsla(180, 100%, 100%, 0.7)" }}
                    >
                      {" "}
                      / 1M pages
                    </Text>
                  </Text>
                )}
                {user?.tier === "Free" && (
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    {fastUsage}
                    <Text
                      size="4"
                      weight="medium"
                      style={{ color: "hsla(180, 100%, 100%, 0.7)" }}
                    >
                      {" "}
                      / 1000 pages
                    </Text>
                  </Text>
                )}
                {user?.tier === "SelfHosted" && (
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    {fastUsage} pages
                  </Text>
                )}
              </Flex>
              <Flex
                flexGrow="1"
                direction="column"
                p="5"
                style={{
                  border: "2px solid hsla(180, 100%, 100%, 0.1)",
                  borderRadius: "8px",
                }}
              >
                <Flex direction="row" justify="between">
                  <Flex
                    direction="row"
                    gap="2"
                    align="center"
                    style={{
                      padding: "6px 12px",
                      borderRadius: "4px",
                      width: "fit-content",
                      border: "1px solid hsla(180, 100%, 100%, 0.1)",
                      backgroundColor: "hsla(180, 100%, 100%, 0.05)",
                    }}
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 16 16"
                      fill="none"
                    >
                      <rect
                        width="16"
                        height="16"
                        fill="white"
                        fill-opacity="0.01"
                      />
                      <path
                        fill-rule="evenodd"
                        clip-rule="evenodd"
                        d="M0.935547 8.00221C0.935547 4.0994 4.0994 0.935547 8.00222 0.935547C11.905 0.935547 15.0689 4.0994 15.0689 8.00221C15.0689 11.905 11.905 15.0689 8.00222 15.0689C4.0994 15.0689 0.935547 11.905 0.935547 8.00221ZM1.97225 7.4667C2.22784 4.55082 4.55082 2.22784 7.4667 1.97225V4.80003C7.4667 5.09458 7.70547 5.33337 8.00003 5.33337C8.29458 5.33337 8.53337 5.09458 8.53337 4.80003V1.97186C11.4513 2.22553 13.7764 4.54935 14.0322 7.4667H11.2C10.9055 7.4667 10.6667 7.70547 10.6667 8.00003C10.6667 8.29458 10.9055 8.53337 11.2 8.53337H14.0325C13.7788 11.4527 11.4527 13.7788 8.53337 14.0325V11.2C8.53337 10.9055 8.29458 10.6667 8.00003 10.6667C7.70547 10.6667 7.4667 10.9055 7.4667 11.2V14.0322C4.54935 13.7764 2.22553 11.4513 1.97186 8.53337H4.80003C5.09458 8.53337 5.33337 8.29458 5.33337 8.00003C5.33337 7.70547 5.09458 7.4667 4.80003 7.4667H1.97225Z"
                        fill="hsla(180, 100%, 100%, 1)"
                      />
                    </svg>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "hsla(180, 100%, 100%, 1)" }}
                    >
                      High Quality ($0.01/page)
                    </Text>
                  </Flex>
                  {user?.tier === "PayAsYouGo" &&
                    highQualityDiscountedUsage < 0 && (
                      <Flex
                        direction="row"
                        gap="2"
                        align="center"
                        style={{
                          padding: "6px 12px",
                          borderRadius: "4px",
                          width: "fit-content",
                          border: "1px solid hsla(180, 100%, 100%, 0.1)",
                          backgroundColor: "hsla(180, 100%, 100%, 0.9)",
                        }}
                      >
                        <Text
                          size="2"
                          weight="medium"
                          style={{ color: "hsla(0, 0%, 0%, 1)" }}
                        >
                          {highQualityDiscount} free pages
                        </Text>
                      </Flex>
                    )}
                </Flex>
                {user?.tier === "PayAsYouGo" && (
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    {highQualityUsage}
                    <Text
                      size="4"
                      weight="medium"
                      style={{ color: "hsla(180, 100%, 100%, 0.7)" }}
                    >
                      {" "}
                      <>/ 1M pages</>
                    </Text>
                  </Text>
                )}
                {user?.tier === "Free" && (
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    {highQualityUsage}
                    <Text
                      size="4"
                      weight="medium"
                      style={{ color: "hsla(180, 100%, 100%, 0.7)" }}
                    >
                      {" "}
                      <>/ 500 pages</>
                    </Text>
                  </Text>
                )}
                {user?.tier === "SelfHosted" && (
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    {highQualityUsage} pages
                  </Text>
                )}
              </Flex>

              {user?.tier === "PayAsYouGo" && (
                <Flex
                  flexGrow="1"
                  p="5"
                  direction="column"
                  style={{
                    border: "2px solid hsla(180, 100%, 100%, 0.1)",
                    borderRadius: "8px",
                  }}
                >
                  <Flex direction="row" justify="between">
                    <Flex
                      direction="row"
                      gap="2"
                      align="center"
                      style={{
                        padding: "6px 12px",
                        borderRadius: "4px",
                        width: "fit-content",
                        border: "1px solid hsla(180, 100%, 100%, 0.1)",
                        backgroundColor: "hsla(180, 100%, 100%, 0.05)",
                      }}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="16"
                        height="16"
                        viewBox="0 0 16 16"
                        fill="none"
                      >
                        <rect
                          width="16"
                          height="16"
                          fill="white"
                          fill-opacity="0.01"
                        />
                        <path
                          fill-rule="evenodd"
                          clip-rule="evenodd"
                          d="M4.48 1.06668H4.4559C4.12733 1.06667 3.84393 1.06666 3.61183 1.08873C3.36614 1.11209 3.12347 1.16389 2.89733 1.30246C2.68205 1.43438 2.50105 1.6154 2.36911 1.83068C2.23054 2.05681 2.17874 2.29949 2.15538 2.54518C2.13331 2.77726 2.13332 3.06065 2.13333 3.38922V3.41335V12.5867V12.6108C2.13332 12.9394 2.13331 13.2227 2.15538 13.4548C2.17874 13.7005 2.23054 13.9432 2.36911 14.1693C2.50105 14.3847 2.68205 14.5657 2.89733 14.6976C3.12347 14.8362 3.36614 14.8879 3.61183 14.9113C3.84393 14.9333 4.12733 14.9333 4.4559 14.9333H4.48H11.52H11.5441C11.8726 14.9333 12.1561 14.9333 12.3882 14.9113C12.6338 14.8879 12.8765 14.8362 13.1026 14.6976C13.318 14.5657 13.499 14.3847 13.6309 14.1693C13.7695 13.9432 13.8212 13.7005 13.8446 13.4548C13.8667 13.2227 13.8667 12.9393 13.8667 12.6108V12.5867V3.41335V3.38925C13.8667 3.06067 13.8667 2.77727 13.8446 2.54518C13.8212 2.29949 13.7695 2.05681 13.6309 1.83068C13.499 1.6154 13.318 1.43438 13.1026 1.30246C12.8765 1.16389 12.6338 1.11209 12.3882 1.08873C12.1561 1.06666 11.8726 1.06667 11.5441 1.06668H11.52H4.48ZM3.45467 2.21194C3.48504 2.19333 3.5452 2.16654 3.7128 2.15061C3.88834 2.13391 4.12051 2.13335 4.48 2.13335H11.52C11.8795 2.13335 12.1117 2.13391 12.2873 2.15061C12.4548 2.16654 12.515 2.19333 12.5454 2.21194C12.6171 2.25592 12.6774 2.31625 12.7214 2.38801C12.7401 2.41838 12.7668 2.47854 12.7827 2.64615C12.7995 2.82169 12.8 3.05386 12.8 3.41335V12.5867C12.8 12.9461 12.7995 13.1784 12.7827 13.3539C12.7668 13.5215 12.7401 13.5817 12.7214 13.6121C12.6774 13.6837 12.6171 13.7441 12.5454 13.7881C12.515 13.8067 12.4548 13.8335 12.2873 13.8494C12.1117 13.8661 11.8795 13.8667 11.52 13.8667H4.48C4.12051 13.8667 3.88834 13.8661 3.7128 13.8494C3.5452 13.8335 3.48504 13.8067 3.45467 13.7881C3.3829 13.7441 3.32257 13.6837 3.27859 13.6121C3.25999 13.5817 3.2332 13.5215 3.21726 13.3539C3.20057 13.1784 3.2 12.9461 3.2 12.5867V3.41335C3.2 3.05386 3.20057 2.82169 3.21726 2.64615C3.2332 2.47854 3.25999 2.41838 3.27859 2.38801C3.32257 2.31625 3.3829 2.25592 3.45467 2.21194ZM5.33333 10.6667C5.03878 10.6667 4.8 10.9055 4.8 11.2C4.8 11.4945 5.03878 11.7333 5.33333 11.7333H8.53333C8.82788 11.7333 9.06667 11.4945 9.06667 11.2C9.06667 10.9055 8.82788 10.6667 8.53333 10.6667H5.33333ZM4.8 8.00001C4.8 7.70547 5.03878 7.46668 5.33333 7.46668H10.6667C10.9612 7.46668 11.2 7.70547 11.2 8.00001C11.2 8.29456 10.9612 8.53335 10.6667 8.53335H5.33333C5.03878 8.53335 4.8 8.29456 4.8 8.00001ZM5.33333 4.26668C5.03878 4.26668 4.8 4.50547 4.8 4.80001C4.8 5.09456 5.03878 5.33335 5.33333 5.33335H10.6667C10.9612 5.33335 11.2 5.09456 11.2 4.80001C11.2 4.50547 10.9612 4.26668 10.6667 4.26668H5.33333Z"
                          fill="hsla(180, 100%, 100%, 1)"
                        />
                      </svg>
                      <Text
                        size="2"
                        weight="medium"
                        style={{ color: "hsla(180, 100%, 100%, 1)" }}
                      >
                        Billing
                      </Text>
                    </Flex>
                    {billingDueDate && (
                      <Text
                        size="2"
                        weight="medium"
                        style={{ color: "hsla(180, 100%, 100%, 1)" }}
                      >
                        Due {billingDueDate}
                      </Text>
                    )}
                  </Flex>
                  <Text
                    size="8"
                    mt="4"
                    weight="bold"
                    style={{ color: "hsla(180, 100%, 100%, 1)" }}
                  >
                    ${billingAmount}
                  </Text>
                </Flex>
              )}
            </Flex>
            {user?.tier === "PayAsYouGo" &&
              (fastDiscount > 0 || highQualityDiscount > 0) && (
                <Flex
                  direction="row"
                  p="4"
                  mt="4"
                  align="center"
                  gap="2"
                  style={{
                    backgroundColor: "hsla(180, 100%, 100%, 0.1)",
                    borderRadius: "4px",
                    color: "hsla(180, 100%, 100%, 1)",
                  }}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="16"
                    height="16"
                    viewBox="0 0 16 16"
                    fill="none"
                  >
                    <rect
                      width="16"
                      height="16"
                      fill="white"
                      fill-opacity="0.01"
                    />
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M7.9999 0.935349C4.09837 0.935349 0.935547 4.09817 0.935547 7.9997C0.935547 11.9012 4.09837 15.0641 7.9999 15.0641C11.9014 15.0641 15.0642 11.9012 15.0642 7.9997C15.0642 4.09817 11.9014 0.935349 7.9999 0.935349ZM1.94887 7.9997C1.94887 4.65782 4.65802 1.94868 7.9999 1.94868C11.3418 1.94868 14.0509 4.65782 14.0509 7.9997C14.0509 11.3415 11.3418 14.0508 7.9999 14.0508C4.65802 14.0508 1.94887 11.3415 1.94887 7.9997ZM8.79991 4.79999C8.79991 5.24181 8.44174 5.59999 7.99991 5.59999C7.55809 5.59999 7.19991 5.24181 7.19991 4.79999C7.19991 4.35815 7.55809 3.99999 7.99991 3.99999C8.44174 3.99999 8.79991 4.35815 8.79991 4.79999ZM6.40003 6.39999H6.93337H8.00003C8.29459 6.39999 8.53336 6.63876 8.53336 6.93332V10.6667H9.0667H9.60003V11.7333H9.0667H8.00003H6.93337H6.40003V10.6667H6.93337H7.4667V7.46665H6.93337H6.40003V6.39999Z"
                      fill="white"
                    />
                  </svg>
                  <Text size="2" weight="medium">
                    A discount of{" "}
                    {fastDiscount > 0 && highQualityDiscount > 0
                      ? `${fastDiscount} fast and ${highQualityDiscount} high-quality`
                      : fastDiscount > 0
                        ? `${fastDiscount} fast`
                        : `${highQualityDiscount} high-quality`}{" "}
                    will be applied to your account for this billing period.
                    Your adjusted bill is ${adjustedBillingAmount}.
                  </Text>
                </Flex>
              )}
          </Flex>

          <Flex direction="column" className="dashboard-content-right">
            <ScrollArea scrollbars="vertical" style={{ height: "100%" }}>
              <Flex direction="column" gap="5">
                <Flex direction="row" align="center" gap="2" mb="2">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                  >
                    <rect
                      width="24"
                      height="24"
                      fill="white"
                      fill-opacity="0.01"
                    />
                    <path
                      fill-rule="evenodd"
                      clip-rule="evenodd"
                      d="M12.4069 2.91125C12.1559 2.7629 11.844 2.7629 11.5929 2.91125L2.79294 8.11125C2.54937 8.25517 2.39993 8.51706 2.39993 8.79999C2.39993 9.08292 2.54937 9.3448 2.79294 9.48872L11.5929 14.6887C11.844 14.8371 12.1559 14.8371 12.4069 14.6887L21.2069 9.48872C21.4506 9.3448 21.6 9.08292 21.6 8.79999C21.6 8.51706 21.4506 8.25517 21.2069 8.11125L12.4069 2.91125ZM11.9999 13.0708L4.77248 8.79999L11.9999 4.52922L19.2274 8.79999L11.9999 13.0708ZM3.60691 13.3113C3.22653 13.0865 2.73597 13.2126 2.51118 13.593C2.28641 13.9734 2.41256 14.464 2.79294 14.6887L11.5929 19.8888C11.844 20.0371 12.1559 20.0371 12.4069 19.8888L21.2069 14.6887C21.5874 14.464 21.7134 13.9734 21.4886 13.593C21.2638 13.2126 20.7733 13.0865 20.393 13.3113L11.9999 18.2707L3.60691 13.3113Z"
                      fill="var(--cyan-2)"
                    />
                  </svg>
                  <Text
                    size="6"
                    weight="bold"
                    style={{ color: "var(--cyan-2)" }}
                  >
                    Tasks
                  </Text>
                </Flex>
                {isLoading ? (
                  <Loader />
                ) : isError ? (
                  <div>Error</div>
                ) : (
                  tasks?.map((task) => (
                    <TaskCard
                      key={task.task_id}
                      {...task}
                      onClick={() => handleTaskClick(task)}
                    />
                  ))
                )}
              </Flex>
            </ScrollArea>
          </Flex>
        </Flex>
      </ScrollArea>
    </div>
  );
}
