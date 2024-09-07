import { Flex, Text, Dialog } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import { User } from "../../models/user.model";
import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "react-oidc-context";
import { createSetupIntent } from "../../services/stripeService";

export default function DashBoardHeader(user: User) {
  const [showApiKey, setShowApiKey] = useState(false);
  const auth = useAuth();
  const accessToken = auth.user?.access_token;

  const createStripeSetupIntent = async () => {
    try {
      const clientSecret = await createSetupIntent(accessToken as string);
      console.log("Stripe Setup Intent Client Secret:", clientSecret);
      // Use the clientSecret here to set up the payment method
      // For example, you might want to pass it to a Stripe Elements component
    } catch (error) {
      console.error("Error creating Stripe Setup Intent:", error);
    }
  };

  const handleLogout = () => {
    auth.removeUser();
    window.location.href = "/";
  };

  return (
    <Flex direction="row" align="center" justify="between">
      <Link to="/">
        <Flex
          direction="row"
          align="center"
          justify="between"
          gap="2"
          style={{
            border: "1px solid var(--cyan-12)",
            borderRadius: "99px",
            backgroundColor: "var(--cyan-12)",
            padding: "4px 12px",
            cursor: "pointer",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
          >
            <rect width="16" height="16" fill="white" fill-opacity="0.01" />
            <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M12.8743 1.88529C13.3127 1.30076 12.2245 0.484595 11.7861 1.06912C11.3477 1.65363 12.4361 2.4698 12.8743 1.88529ZM8.95214 1.14998C7.63291 1.01397 6.16779 1.24104 4.88995 1.82332C3.61122 2.406 2.48251 3.36329 1.90767 4.71287C1.46745 5.74639 1.08496 6.69987 1.0676 7.93977C1.05037 9.17167 1.39223 10.6336 2.27385 12.7232C2.78551 13.9359 4.30894 14.4476 5.48155 14.7284C6.83558 15.0527 8.32914 15.1375 9.24203 15.0125C9.98949 14.9101 10.8211 14.7503 11.5186 14.1835C12.6909 13.2308 13.761 12.1552 14.3994 11.0286C15.0443 9.89035 15.2806 8.63154 14.6364 7.41679C14.467 7.09731 14.2037 7.01583 13.9106 6.92512C13.7999 6.89083 13.6848 6.85523 13.569 6.80497C12.3958 6.29583 11.6956 5.70627 11.6411 5.36401C11.59 5.04328 11.6502 4.8156 11.7132 4.57793C11.7471 4.45012 11.7817 4.31943 11.8001 4.16983C11.8227 3.98652 11.8202 3.5554 11.4129 3.33611C11.0985 3.1668 10.8975 2.83566 10.7101 2.52697C10.6639 2.45093 10.6186 2.37624 10.5726 2.30568C10.2681 1.83858 9.82153 1.2396 8.95214 1.14998ZM5.33225 2.79397C4.22915 3.29662 3.33391 4.08637 2.88902 5.13086C2.45128 6.15859 2.14833 6.94264 2.13417 7.9547C2.1199 8.97477 2.40046 10.2793 3.25664 12.3086C3.61793 13.1649 4.92978 13.4994 5.72998 13.6911C6.98592 13.9919 8.34347 14.0589 9.09726 13.9556C9.82689 13.8557 10.3954 13.7218 10.846 13.3556C11.9776 12.4359 12.9291 11.4597 13.4713 10.5028C13.7972 9.92756 14.415 8.21632 13.4711 7.87978C11.4708 7.16666 10.9461 6.47113 10.6561 5.77352C10.4631 5.30927 10.4839 4.82434 10.6212 4.35077C10.6417 4.28011 10.6617 4.2111 10.7023 4.14807C10.4731 3.98587 10.2949 3.78874 10.1489 3.59626C10.0573 3.4755 9.97003 3.32905 9.87943 3.17713C9.6199 2.74188 9.33352 2.26162 8.84275 2.21102C7.71081 2.09432 6.43624 2.2909 5.33225 2.79397ZM14.0886 4.02635C14.3371 3.44182 13.285 2.81179 12.8877 3.3084C12.3538 3.9758 13.7631 4.79233 14.0886 4.02635ZM15.325 2.41099C15.9646 2.42659 16.1654 1.19043 15.5433 1.01266C14.7281 0.779758 14.4979 2.39081 15.325 2.41099ZM15.2381 5.43038C15.8554 5.44004 16.0493 4.67436 15.4487 4.56425C14.6619 4.42 14.4398 5.41788 15.2381 5.43038ZM7.368 4.5725C6.90024 4.97871 6.45568 4.42864 6.37848 3.9856C6.25971 3.30393 6.75942 2.73979 7.41144 3.10566C7.929 3.39607 7.75123 4.23969 7.368 4.5725ZM5.27419 6.27396C5.76944 5.77871 4.90205 4.8748 4.38853 5.38831C4.08894 5.68791 4.27665 6.03067 4.51993 6.27396C4.72822 6.48224 5.06591 6.48224 5.27419 6.27396ZM7.4423 7.16783C7.79926 7.88175 9.05391 7.24436 8.52707 6.55774C8.38953 6.37848 8.19901 6.34602 8.12955 6.3382C7.69298 6.28907 7.21439 6.712 7.4423 7.16783ZM7.45384 9.30625C7.6167 9.72672 8.01197 9.93994 8.44709 9.77418C8.82596 9.62985 8.88732 8.95668 8.71653 8.61255C8.5582 8.29348 8.14886 8.18979 7.82663 8.31378C7.45079 8.45841 7.31383 8.94622 7.45148 9.30015L7.45384 9.30625ZM11.0811 9.43019C11.4747 9.2337 12.2258 9.41622 12.2258 9.93279C12.2258 10.3604 11.6893 10.8976 11.2676 10.9681C10.8601 11.0363 10.3522 10.6409 10.4176 10.2157C10.4687 9.88308 10.7965 9.57227 11.0811 9.43019ZM8.07151 11.5661C7.82358 11.4669 7.61976 11.5841 7.58866 11.602L7.58697 11.603C7.36834 11.7279 7.16142 11.9935 7.11968 12.2498C7.02952 12.8036 7.60745 12.9322 8.02838 12.9053C8.75456 12.859 8.60317 11.7788 8.07151 11.5661ZM5.20587 11.6194C5.40523 11.4905 5.50035 11.2387 5.42179 11.003C5.25916 10.45 4.61291 10.36 4.22998 10.805C4.04625 11.0186 4.06132 11.3359 4.25828 11.5312L4.26128 11.5349L4.26443 11.5388C4.5016 11.8338 4.94532 11.9314 5.20587 11.6194ZM3.25008 8.33406C3.55357 8.63754 3.84082 8.69404 4.25006 8.59583C4.91548 8.43612 4.41401 6.65919 3.25013 7.33115C2.89001 7.53907 3.0583 8.14228 3.25008 8.33406Z"
              fill="#CAF1F6"
            />
          </svg>

          <Text size="2" weight="medium" style={{ color: "var(--cyan-4)" }}>
            chunk_my_docs
          </Text>
        </Flex>
      </Link>
      <Flex direction="row" gap="4">
        <Dialog.Root open={showApiKey} onOpenChange={setShowApiKey}>
          <Dialog.Trigger>
            <BetterButton padding="4px 12px">
              <Text size="2" weight="medium" style={{ color: "var(--cyan-4)" }}>
                API Key
              </Text>
            </BetterButton>
          </Dialog.Trigger>
          <Dialog.Content
            style={{
              backgroundColor: "hsl(189, 70%, 3%)",
              boxShadow: "0 0 0 1px var(--cyan-12)",
              border: "1px solid var(--cyan-12)",
              outline: "none",
              borderRadius: "8px",
            }}
          >
            <Flex direction="row" align="center" gap="2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
              >
                <rect width="24" height="24" fill="white" fill-opacity="0.01" />
                <path
                  fill-rule="evenodd"
                  clip-rule="evenodd"
                  d="M15.9428 4.29713C16.1069 3.88689 15.9073 3.42132 15.4971 3.25723C15.0869 3.09315 14.6213 3.29267 14.4572 3.70291L8.05722 19.7029C7.89312 20.1131 8.09266 20.5787 8.50288 20.7427C8.91312 20.9069 9.37869 20.7074 9.54278 20.2971L15.9428 4.29713ZM6.16568 8.23433C6.47811 8.54675 6.47811 9.05327 6.16568 9.36569L3.53138 12L6.16568 14.6343C6.47811 14.9467 6.47811 15.4533 6.16568 15.7657C5.85326 16.0781 5.34674 16.0781 5.03432 15.7657L1.83432 12.5657C1.52189 12.2533 1.52189 11.7467 1.83432 11.4343L5.03432 8.23433C5.34674 7.92191 5.85326 7.92191 6.16568 8.23433ZM17.8342 8.23433C18.1467 7.92191 18.6533 7.92191 18.9658 8.23433L22.1658 11.4343C22.4781 11.7467 22.4781 12.2533 22.1658 12.5657L18.9658 15.7657C18.6533 16.0781 18.1467 16.0781 17.8342 15.7657C17.5219 15.4533 17.5219 14.9467 17.8342 14.6343L20.4686 12L17.8342 9.36569C17.5219 9.05327 17.5219 8.54675 17.8342 8.23433Z"
                  fill="#CAF1F6"
                />
              </svg>
              <Text size="6" weight="bold" style={{ color: "var(--cyan-4)" }}>
                API Key
              </Text>
            </Flex>

            <Flex direction="column" gap="2" mt="4">
              <Flex
                direction="row"
                align="center"
                justify="between"
                gap="4"
                p="2"
                style={{
                  border: "1px solid var(--cyan-12)",
                  borderRadius: "4px",
                }}
              >
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "var(--cyan-4)" }}
                >
                  {user.api_keys[0]}
                </Text>
              </Flex>
              <Flex direction="row" gap="4" mt="2">
                <BetterButton
                  onClick={() => {
                    navigator.clipboard.writeText(user.api_keys[0]);
                  }}
                >
                  <Text
                    size="2"
                    weight="medium"
                    style={{ color: "var(--cyan-4)" }}
                  >
                    Copy
                  </Text>
                </BetterButton>
                <Dialog.Close>
                  <BetterButton>
                    <Text
                      size="2"
                      weight="medium"
                      style={{ color: "var(--cyan-4)" }}
                    >
                      Close
                    </Text>
                  </BetterButton>
                </Dialog.Close>
              </Flex>
            </Flex>
          </Dialog.Content>
        </Dialog.Root>
        {user?.tier === "Free" ? (
          <BetterButton padding="4px 12px" onClick={createStripeSetupIntent}>
            <Text size="2" weight="medium" style={{ color: "var(--cyan-4)" }}>
              Add Payment Method
            </Text>
          </BetterButton>
        ) : (
          <BetterButton padding="4px 12px">
            <Text size="2" weight="medium" style={{ color: "var(--cyan-4)" }}>
              Manage Payments
            </Text>
          </BetterButton>
        )}
        <BetterButton padding="4px 12px" onClick={handleLogout}>
          <Text size="2" weight="medium" style={{ color: "var(--cyan-4)" }}>
            Logout
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
