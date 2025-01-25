import { Flex, Text } from "@radix-ui/themes";
import "./Usage.css";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";
import BetterButton from "../BetterButton/BetterButton";
import { useAuth } from "react-oidc-context";
import { getBillingPortalSession } from "../../services/stripeService";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

interface UsageProps {
  customerId?: string;
}

export default function UsagePage({ customerId }: UsageProps) {
  return (
    <Flex direction="column" className="usage-container">
      <Billing customerId={customerId} />
    </Flex>
  );
}

interface BillingProps {
  customerId?: string;
}

export function Billing({ customerId }: BillingProps) {
  const { data: monthlyUsage, isLoading } = useMonthlyUsage();
  const auth = useAuth();
  const navigate = useNavigate();
  const [isLoadingPortal, setIsLoadingPortal] = useState(false);

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
    <Flex direction="column" className="billing-container" gap="5">
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

      <Text size="2" style={{ color: "#FFF" }}>
        Track your monthly usage, manage your subscription plan, and view
        detailed billing information.
      </Text>
      <Flex direction="column" gap="6" mt="5" style={{ flexWrap: "wrap" }}>
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
        </Flex>

        <Flex direction="row" gap="6" style={{ flexWrap: "wrap" }}>
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
                    : "Your payment method is up to date."}
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
    </Flex>
  );
}
