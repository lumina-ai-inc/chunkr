import { useRef, useEffect } from "react";
import { Text, Flex, Separator } from "@radix-ui/themes";
import "./Account.css";
import BetterButton from "../BetterButton/BetterButton";
import { useAuth } from "react-oidc-context";

export default function Account({ onClose }: { onClose: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const auth = useAuth();
  const user = auth.user;

  console.log(user);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        onClose();
      }
    }

    // Use setTimeout to add the event listener on the next tick
    setTimeout(() => {
      document.addEventListener("mousedown", handleClickOutside);
    }, 0);

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [onClose]);

  return (
    <div className="account-container">
      <div
        className="account-module-container"
        ref={containerRef}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="close-button-container" onClick={onClose}>
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
              d="M18.8506 6.45054C19.21 6.09126 19.21 5.50874 18.8506 5.14946C18.4914 4.79018 17.9089 4.79018 17.5497 5.14946L12.0001 10.6989L6.45067 5.14946C6.09138 4.79018 5.50887 4.79018 5.14959 5.14946C4.79031 5.50874 4.79031 6.09126 5.14959 6.45054L10.6991 12L5.14959 17.5495C4.79031 17.9088 4.79031 18.4912 5.14959 18.8506C5.50887 19.2098 6.09138 19.2098 6.45067 18.8506L12.0001 13.3011L17.5497 18.8506C17.9089 19.2098 18.4914 19.2098 18.8506 18.8506C19.21 18.4912 19.21 17.9088 18.8506 17.5495L13.3012 12L18.8506 6.45054Z"
              fill="#CAF1F6"
            />
          </svg>
        </div>
        <Flex direction="column" gap="4px">
          <Text size="2" weight="bold" className="cyan-1">
            ACCOUNT
          </Text>
          <Text size="6" weight="bold" className="cyan-8">
            m-chadd100@gmail.com
          </Text>
        </Flex>
        <Flex direction="column" gap="16px" mt="32px">
          <Text size="2" weight="bold" className="cyan-1">
            USAGE
          </Text>
          <UsageLoader />
        </Flex>
        <Flex
          direction="row"
          gap="16px"
          mt="16px"
          align="center"
          className="api-container"
        >
          <Text size="2" weight="bold" className="cyan-4">
            Key
          </Text>
          <Separator
            size="2"
            orientation="vertical"
            style={{ backgroundColor: "var(--cyan-5)" }}
          />
          <Flex direction="row" justify="between" align="center" width="100%">
            <Text size="2" weight="regular" className="cyan-4">
              Uxve-ahg3g-8jjd-hhga
            </Text>
            <div className="copy-icon">
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
                  d="M1.06665 10.1334C1.06665 11.0171 1.78299 11.7335 2.66665 11.7335H4.26665V10.6668H2.66665C2.3721 10.6668 2.13332 10.428 2.13332 10.1334V2.66677C2.13332 2.37222 2.3721 2.13344 2.66665 2.13344H10.1333C10.4279 2.13344 10.6667 2.37222 10.6667 2.66677V4.26673H5.86665C4.983 4.26673 4.26665 4.98307 4.26665 5.86673V13.3334C4.26665 14.217 4.983 14.9334 5.86665 14.9334H13.3333C14.2169 14.9334 14.9333 14.217 14.9333 13.3334V5.86673C14.9333 4.98307 14.2169 4.26673 13.3333 4.26673H11.7333V2.66677C11.7333 1.78311 11.0169 1.06677 10.1333 1.06677H2.66665C1.78299 1.06677 1.06665 1.78311 1.06665 2.66677V10.1334ZM5.33332 5.86673C5.33332 5.57218 5.5721 5.3334 5.86665 5.3334H13.3333C13.6278 5.3334 13.8667 5.57218 13.8667 5.86673V13.3334C13.8667 13.628 13.6278 13.8667 13.3333 13.8667H5.86665C5.5721 13.8667 5.33332 13.628 5.33332 13.3334V5.86673Z"
                  fill="var(--cyan-5)"
                />
              </svg>
            </div>
          </Flex>
        </Flex>
        <Flex
          direction="row"
          gap="16px"
          mt="24px"
          width="100%"
          justify="between"
        >
          <div className="badge-container">
            <Text size="2" weight="medium" className="cyan-1">
              Free
            </Text>
          </div>
          <Flex direction="row" gap="16px">
            <BetterButton>
              <Text size="2" weight="medium" className="cyan-1">
                Manage Payment
              </Text>
            </BetterButton>
            <BetterButton onClick={() => auth.signoutRedirect()}>
              <Text size="2" weight="medium" className="cyan-1">
                Logout
              </Text>
            </BetterButton>
          </Flex>
        </Flex>
      </div>
    </div>
  );
}

function UsageLoader() {
  return (
    <div className="usage-loader-container">
      <div className="usage-loader-background">
        <div className="usage-loader-progress"></div>
      </div>
      <Text size="2" weight="medium" className="cyan-5">
        75 / 100 pages used
      </Text>
    </div>
  );
}
