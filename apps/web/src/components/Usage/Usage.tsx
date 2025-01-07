import { Text, Flex } from "@radix-ui/themes";
import "./Usage.css";
import useMonthlyUsage from "../../hooks/useMonthlyUsage";
import useUser from "../../hooks/useUser";
import {
  calculateDiscountedBilling,
  calculateBillingDueDate,
} from "../../models/usage.model";
import Loader from "../../pages/Loader/Loader";
import BetterButton from "../BetterButton/BetterButton";
import { Dialog } from "@radix-ui/themes";
import PaymentSetup from "../Payments/PaymentSetup";

// UsageCard Component
function UsageCard({
  title,
  description,
  usage,
  limit,
  discount,
}: {
  title: string;
  description: string;
  usage: number;
  limit: number;
  discount: number;
}) {
  const percentage = limit ? Math.min((usage / limit) * 100, 100) : 0;

  return (
    <div className="usage-card">
      <Flex direction="column" gap="3">
        <Flex justify="between" align="center">
          <Text size="5" weight="bold" className="white">
            {title}
          </Text>
          <Text
            className="usage-badge"
            weight="medium"
            style={{ color: "rgba(255, 255, 255, 0.9)" }}
          >
            {usage.toLocaleString()} /{" "}
            {limit === 0 ? "∞" : limit.toLocaleString()}
          </Text>
        </Flex>
        <Flex className="usage-progress-bar" mt="1">
          <div
            className="usage-progress-fill"
            style={{ width: `${percentage}%` }}
          />
        </Flex>
        <Text size="2" style={{ color: "rgba(255, 255, 255, 0.7)" }}>
          {description}
        </Text>
        {discount > 0 && (
          <Text
            className="usage-discount-badge"
            style={{ color: "rgba(255, 255, 255, 0.7)" }}
          >
            {discount} % discount applied
          </Text>
        )}
      </Flex>
    </div>
  );
}

// BillingCard Component
function BillingCard({
  billingAmount,
  adjustedBillingAmount,
  dueDate,
}: {
  billingAmount: number;
  adjustedBillingAmount: number;
  dueDate: Date | null;
}) {
  return (
    <div className="usage-card">
      <Flex direction="column" gap="3">
        <Text size="5" weight="bold" className="white">
          Current Bill
        </Text>
        <Flex direction="column" gap="2">
          <Text size="4">
            ${adjustedBillingAmount.toFixed(2)}
            {billingAmount !== adjustedBillingAmount && (
              <Text size="2" style={{ textDecoration: "line-through" }}>
                ${billingAmount.toFixed(2)}
              </Text>
            )}
          </Text>
          {dueDate && (
            <Text size="2" color="gray">
              Due by {dueDate.toLocaleDateString()}
            </Text>
          )}
        </Flex>
      </Flex>
    </div>
  );
}

// UsageAlert Component
function UsageAlert({ tier }: { tier: string }) {
  return (
    <div className="usage-alert">
      <Text size="2" style={{ color: "rgba(255, 255, 255, 0.9)" }}>
        {tier === "Free"
          ? "You've reached your free tier limit. Upgrade to continue processing."
          : "You've reached your usage limit. Please check your billing details."}
      </Text>
    </div>
  );
}

// New PaymentCard Component
function PaymentCard({
  showPaymentSetup,
  setShowPaymentSetup,
  customerSessionSecret,
  customerSessionClientSecret,
  handleAddPaymentMethod,
  tier,
}: {
  showPaymentSetup: boolean;
  setShowPaymentSetup: (show: boolean) => void;
  customerSessionSecret: string | null;
  customerSessionClientSecret: string | null;
  handleAddPaymentMethod: () => Promise<void>;
  tier: string;
}) {
  return (
    <div className="usage-card payment-card">
      <Flex direction="column" gap="3">
        <Text size="5" weight="bold" className="white">
          Payment Method
        </Text>
        <Text size="2" style={{ color: "rgba(255, 255, 255, 0.7)" }}>
          {tier === "Free"
            ? "Add a payment method to unlock unlimited usage"
            : "Manage your payment methods and billing details"}
        </Text>
        <BetterButton padding="4px 12px" onClick={handleAddPaymentMethod}>
          <Text
            size="2"
            weight="medium"
            style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
          >
            {tier === "Free" ? "Add Payment Method" : "Manage Payments"}
          </Text>
        </BetterButton>
      </Flex>

      {showPaymentSetup && customerSessionSecret && (
        <Dialog.Root open={showPaymentSetup} onOpenChange={setShowPaymentSetup}>
          <Dialog.Content
            style={{
              backgroundColor: "hsla(189, 64%, 3%, 1)",
              boxShadow: "0 0 0 1px hsla(0, 0%, 100%, 0.1)",
              border: "1px solid hsla(0, 0%, 100%, 0.1)",
              outline: "none",
              borderRadius: "8px",
            }}
          >
            <PaymentSetup
              customerSessionSecret={customerSessionSecret}
              clientSecret={customerSessionClientSecret as string}
            />
          </Dialog.Content>
        </Dialog.Root>
      )}
    </div>
  );
}

// Main Usage Component
export default function Usage({
  showPaymentSetup,
  setShowPaymentSetup,
  customerSessionSecret,
  customerSessionClientSecret,
  handleAddPaymentMethod,
}: {
  showPaymentSetup: boolean;
  setShowPaymentSetup: (show: boolean) => void;
  customerSessionSecret: string | null;
  customerSessionClientSecret: string | null;
  handleAddPaymentMethod: () => Promise<void>;
}) {
  const { data: user } = useUser();
  const { data: monthlyUsage } = useMonthlyUsage();

  if (!user || !monthlyUsage) {
    return <Loader />;
  }

  const totalUsage = monthlyUsage.reduce((total, month) => {
    return (
      total + month.usage_details.reduce((sum, detail) => sum + detail.count, 0)
    );
  }, 0);

  const creditLimit = user.usage?.[0]?.usage_limit || 0;
  const discount = user.usage?.[0]?.discounts?.[0]?.amount || 0;
  const totalCost = monthlyUsage[0]?.total_cost || 0;

  const adjustedBillingAmount = Number(
    calculateDiscountedBilling(totalUsage, discount, totalCost).toFixed(3)
  );

  const billingAmount = Math.max(0, Number(totalCost.toFixed(3)));
  const billingDueDate = monthlyUsage[0]?.month
    ? calculateBillingDueDate(monthlyUsage[0].month)
    : null;

  const showLimitAlert =
    (user.tier === "Free" || user.tier === "PayAsYouGo") &&
    totalUsage >= creditLimit;

  return (
    <Flex direction="column" gap="6" className="usage-container">
      <Text size="8" weight="bold" className="white">
        Current Billing Cycle
      </Text>
      <Flex direction="row" gap="6" wrap="wrap">
        <UsageCard
          title="Credits Used"
          description={`${user.tier} tier • ${creditLimit.toLocaleString()} pages/month`}
          usage={totalUsage}
          limit={creditLimit}
          discount={discount}
        />

        {user.tier === "PayAsYouGo" && (
          <BillingCard
            billingAmount={billingAmount}
            adjustedBillingAmount={adjustedBillingAmount}
            dueDate={billingDueDate ? new Date(billingDueDate) : null}
          />
        )}

        <PaymentCard
          showPaymentSetup={showPaymentSetup}
          setShowPaymentSetup={setShowPaymentSetup}
          customerSessionSecret={customerSessionSecret}
          customerSessionClientSecret={customerSessionClientSecret}
          handleAddPaymentMethod={handleAddPaymentMethod}
          tier={user.tier}
        />
      </Flex>

      {showLimitAlert && <UsageAlert tier={user.tier} />}
    </Flex>
  );
}
