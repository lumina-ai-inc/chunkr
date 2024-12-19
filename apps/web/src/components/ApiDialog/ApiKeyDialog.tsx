import { Flex, Text, Dialog } from "@radix-ui/themes";
import BetterButton from "../BetterButton/BetterButton";
import { User } from "../../models/user.model";
import "./ApiKeyDialog.css";

interface ApiKeyDialogProps {
  user: User;
  showApiKey: boolean;
  setShowApiKey: (show: boolean) => void;
  phone?: boolean;
}

export default function ApiKeyDialog({
  user,
  showApiKey,
  setShowApiKey,
  phone = false,
}: ApiKeyDialogProps) {
  return (
    <Dialog.Root open={showApiKey} onOpenChange={setShowApiKey}>
      <Dialog.Trigger>
        {phone ? (
          <Text size="2" weight="regular" mt="2px" style={{ color: "#000000" }}>
            API Key
          </Text>
        ) : (
          <BetterButton>
            <Text
              size="1"
              weight="medium"
              mt="2px"
              style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
            >
              API Key
            </Text>
          </BetterButton>
        )}
      </Dialog.Trigger>
      <Dialog.Content
        style={{
          backgroundColor: "hsla(0, 0%, 0%)",
          boxShadow: "0 0 0 1px hsla(0, 0%, 100%, 0.1)",
          border: "1px solid hsla(0, 0%, 100%, 0.1)",
          outline: "none",
          borderRadius: "8px",
          width: "fit-content",
        }}
      >
        <Flex direction="row" align="center" gap="4">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
          >
            <rect width="24" height="24" fill="white" fillOpacity="0.01" />
            <path
              fillRule="evenodd"
              clipRule="evenodd"
              d="M15.9428 4.29713C16.1069 3.88689 15.9073 3.42132 15.4971 3.25723C15.0869 3.09315 14.6213 3.29267 14.4572 3.70291L8.05722 19.7029C7.89312 20.1131 8.09266 20.5787 8.50288 20.7427C8.91312 20.9069 9.37869 20.7074 9.54278 20.2971L15.9428 4.29713ZM6.16568 8.23433C6.47811 8.54675 6.47811 9.05327 6.16568 9.36569L3.53138 12L6.16568 14.6343C6.47811 14.9467 6.47811 15.4533 6.16568 15.7657C5.85326 16.0781 5.34674 16.0781 5.03432 15.7657L1.83432 12.5657C1.52189 12.2533 1.52189 11.7467 1.83432 11.4343L5.03432 8.23433C5.34674 7.92191 5.85326 7.92191 6.16568 8.23433ZM17.8342 8.23433C18.1467 7.92191 18.6533 7.92191 18.9658 8.23433L22.1658 11.4343C22.4781 11.7467 22.4781 12.2533 22.1658 12.5657L18.9658 15.7657C18.6533 16.0781 18.1467 16.0781 17.8342 15.7657C17.5219 15.4533 17.5219 14.9467 17.8342 14.6343L20.4686 12L17.8342 9.36569C17.5219 9.05327 17.5219 8.54675 17.8342 8.23433Z"
              fill="hsla(0, 0%, 100%, 0.9)"
            />
          </svg>
          <Text
            size="6"
            weight="bold"
            style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
          >
            API Key
          </Text>
        </Flex>

        <Flex direction="column" gap="2" mt="5">
          <Flex
            direction="row"
            align="center"
            justify="between"
            gap="4"
            p="4"
            style={{
              border: "1px solid hsla(0, 0%, 100%, 0.2)",
              borderRadius: "4px",
            }}
          >
            <Text
              size="2"
              weight="medium"
              style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
            >
              {user.api_keys[0]}
            </Text>
          </Flex>
          <Flex direction="row" gap="4" mt="3">
            <BetterButton
              onClick={() => {
                navigator.clipboard.writeText(user.api_keys[0]);
              }}
            >
              <Text
                size="2"
                weight="medium"
                style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
              >
                Copy
              </Text>
            </BetterButton>
            <Dialog.Close>
              <BetterButton>
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "hsla(0, 0%, 100%, 0.9)" }}
                >
                  Close
                </Text>
              </BetterButton>
            </Dialog.Close>
          </Flex>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  );
}
