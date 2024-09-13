import { Flex, Text, Dialog } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import { User } from "../../models/user.model";
import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "react-oidc-context";
import { createSetupIntent } from "../../services/stripeService";
import PaymentSetup from "../Payments/PaymentSetup";

export default function DashBoardHeader(user: User) {
  const [showPaymentSetup, setShowPaymentSetup] = useState(false);
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const auth = useAuth();
  const accessToken = auth.user?.access_token;

  const handleAddPaymentMethod = async () => {
    try {
      const secret = await createSetupIntent(accessToken as string);
      setClientSecret(secret);
      setShowPaymentSetup(true);
    } catch (error) {
      console.error("Error creating Stripe Setup Intent:", error);
    }
  };

  const handleLogout = () => {
    auth.signoutRedirect();
    window.location.href = "/";
  };

  const handleSupport = () => {
    window.open(
      "https://cal.com/mehulc/15min",
      "_blank",
      "noopener,noreferrer"
    );
  };

  const handleApiDocs = () => {
    window.open(
      `${import.meta.env.VITE_API_URL}/redoc`,
      "_blank",
      "noopener,noreferrer"
    );
  };

  return (
    <Flex direction="row" align="center" justify="between" width="100%" px="7">
      <Link to="/" style={{ textDecoration: "none" }}>
        <div className="logo-container">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
          >
            <path
              d="M5.88 9.78C6.42518 9.99822 7.02243 10.0516 7.59768 9.93364C8.17294 9.81564 8.70092 9.53139 9.11616 9.11616C9.53139 8.70092 9.81564 8.17294 9.93364 7.59768C10.0516 7.02243 9.99822 6.42518 9.78 5.88C10.4143 5.70922 10.975 5.33496 11.3761 4.81468C11.7771 4.29441 11.9963 3.65689 12 3C13.78 3 15.5201 3.52784 17.0001 4.51677C18.4802 5.50571 19.6337 6.91131 20.3149 8.55585C20.9961 10.2004 21.1743 12.01 20.8271 13.7558C20.4798 15.5016 19.6226 17.1053 18.364 18.364C17.1053 19.6226 15.5016 20.4798 13.7558 20.8271C12.01 21.1743 10.2004 20.9961 8.55585 20.3149C6.91131 19.6337 5.50571 18.4802 4.51677 17.0001C3.52784 15.5201 3 13.78 3 12C3.65689 11.9963 4.29441 11.7771 4.81468 11.3761C5.33496 10.975 5.70922 10.4143 5.88 9.78Z"
              stroke="white"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
          </svg>
          <Text size="4" weight="bold" className="logo-title">
            chunkr
          </Text>
        </div>
      </Link>
      <Flex direction="row" gap="4">
        {user?.tier === "Free" && (
          <BetterButton padding="4px 12px" onClick={handleAddPaymentMethod}>
            <Text
              size="1"
              weight="medium"
              style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
            >
              Add Payment Method
            </Text>
          </BetterButton>
        )}
        {/* {user?.tier !== "Free" && (
          <BetterButton padding="4px 12px">
            <Text
              size="1"
              weight="medium"
              style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
            >
              Manage Payments
            </Text>
          </BetterButton>
        )} */}
        <BetterButton padding="4px 12px" onClick={handleApiDocs}>
          <Text
            size="1"
            weight="medium"
            style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
          >
            API Docs
          </Text>
        </BetterButton>
        <BetterButton padding="4px 12px" onClick={handleSupport}>
          <Text
            size="1"
            weight="medium"
            style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
          >
            Support
          </Text>
        </BetterButton>
        <BetterButton padding="4px 12px" onClick={handleLogout}>
          <Text
            size="1"
            weight="medium"
            style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
          >
            Logout
          </Text>
        </BetterButton>
      </Flex>
      {showPaymentSetup && clientSecret && (
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
            <PaymentSetup clientSecret={clientSecret} />
          </Dialog.Content>
        </Dialog.Root>
      )}
    </Flex>
  );
}
