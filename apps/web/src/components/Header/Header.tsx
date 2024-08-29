import { useState } from "react";
import { DropdownMenu, Flex, Text, Button, Code } from "@radix-ui/themes";
import { Link } from "react-router-dom";
import "./Header.css";
import Account from "../Auth/Account";

interface HeaderProps {
  py?: string;
  px?: string;
  download?: boolean;
  home?: boolean;
}

export default function Header({
  py = "40px",
  px = "80px",
  download = false,
  home = false,
}: HeaderProps) {
  const [showAccount, setShowAccount] = useState(false);

  return (
    <Flex direction="row" justify="between" py={py} px={px} className="header">
      <Link to="/" style={{ textDecoration: "none" }}>
        <Flex className="logo" direction="row" gap="4" align="center">
          <div className="logo-container">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="40"
              height="40"
              viewBox="0 0 40 40"
              fill="none"
            >
              <path
                d="M0 4C0 1.79086 1.79086 0 4 0H36C38.2091 0 40 1.79086 40 4V36C40 38.2091 38.2091 40 36 40H4C1.79086 40 0 38.2091 0 36V4Z"
                fill="#05161A"
              />
              <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M27.3115 10.8279C27.9691 9.95115 26.3368 8.7269 25.6792 9.60368C25.0216 10.4804 26.6541 11.7047 27.3115 10.8279ZM21.4282 9.72498C19.4494 9.52096 17.2517 9.86157 15.3349 10.735C13.4168 11.609 11.7238 13.0449 10.8615 15.0693C10.2012 16.6196 9.62744 18.0498 9.60141 19.9097C9.57556 21.7575 10.0884 23.9504 11.4108 27.0848C12.1783 28.9038 14.4634 29.6714 16.2223 30.0926C18.2534 30.579 20.4937 30.7062 21.863 30.5187C22.9842 30.3651 24.2317 30.1254 25.2779 29.2752C27.0363 27.8462 28.6414 26.2328 29.599 24.5429C30.5664 22.8355 30.921 20.9473 29.9546 19.1252C29.7005 18.646 29.3056 18.5237 28.8659 18.3877C28.6998 18.3363 28.5272 18.2828 28.3534 18.2075C26.5938 17.4437 25.5434 16.5594 25.4616 16.046C25.385 15.5649 25.4754 15.2234 25.5698 14.8669C25.6206 14.6752 25.6725 14.4792 25.7002 14.2548C25.7341 13.9798 25.7302 13.3331 25.1194 13.0042C24.6477 12.7502 24.3462 12.2535 24.0651 11.7905C23.9959 11.6764 23.9279 11.5644 23.8589 11.4585C23.4021 10.7579 22.7323 9.85941 21.4282 9.72498ZM15.9984 12.191C14.3437 12.9449 13.0009 14.1296 12.3335 15.6963C11.6769 17.2379 11.2225 18.414 11.2013 19.932C11.1799 21.4622 11.6007 23.4189 12.885 26.4629C13.4269 27.7474 15.3947 28.2491 16.595 28.5366C18.4789 28.9878 20.5152 29.0883 21.6459 28.9334C22.7403 28.7835 23.5931 28.5827 24.269 28.0334C25.9664 26.6539 27.3936 25.1896 28.2069 23.7542C28.6958 22.8913 29.6226 20.3245 28.2067 19.8197C25.2062 18.75 24.4192 17.7067 23.9842 16.6603C23.6946 15.9639 23.7259 15.2365 23.9319 14.5262C23.9626 14.4202 23.9926 14.3167 24.0534 14.2221C23.7097 13.9788 23.4424 13.6831 23.2234 13.3944C23.086 13.2132 22.955 12.9936 22.8192 12.7657C22.4299 12.1128 22.0003 11.3924 21.2641 11.3165C19.5662 11.1415 17.6544 11.4364 15.9984 12.191ZM29.133 14.0395C29.5056 13.1627 27.9275 12.2177 27.3315 12.9626C26.5307 13.9637 28.6446 15.1885 29.133 14.0395ZM30.9875 11.6165C31.9469 11.6399 32.2482 9.78565 31.3149 9.51899C30.0922 9.16964 29.7469 11.5862 30.9875 11.6165ZM30.8571 16.1456C31.783 16.1601 32.0739 15.0116 31.1731 14.8464C29.9928 14.63 29.6597 16.1268 30.8571 16.1456ZM19.052 14.8588C18.3504 15.4681 17.6835 14.643 17.5677 13.9784C17.3896 12.9559 18.1391 12.1097 19.1172 12.6585C19.8935 13.0941 19.6268 14.3595 19.052 14.8588ZM15.9113 17.4109C16.6542 16.6681 15.3531 15.3122 14.5828 16.0825C14.1334 16.5319 14.415 17.046 14.7799 17.4109C15.0923 17.7234 15.5989 17.7234 15.9113 17.4109ZM19.1635 18.7517C19.6989 19.8226 21.5809 18.8665 20.7906 17.8366C20.5843 17.5677 20.2985 17.519 20.1943 17.5073C19.5395 17.4336 18.8216 18.068 19.1635 18.7517ZM19.1808 21.9594C19.4251 22.5901 20.018 22.9099 20.6706 22.6613C21.2389 22.4448 21.331 21.435 21.0748 20.9188C20.8373 20.4402 20.2233 20.2847 19.74 20.4707C19.1762 20.6876 18.9708 21.4193 19.1772 21.9502L19.1808 21.9594ZM24.6216 22.1453C25.212 21.8506 26.3387 22.1243 26.3387 22.8992C26.3387 23.5406 25.5339 24.3464 24.9014 24.4522C24.2901 24.5544 23.5282 23.9613 23.6264 23.3235C23.7031 22.8246 24.1947 22.3584 24.6216 22.1453ZM20.1073 25.3491C19.7354 25.2003 19.4296 25.3762 19.383 25.403L19.3805 25.4045C19.0525 25.5918 18.7421 25.9902 18.6795 26.3747C18.5443 27.2054 19.4112 27.3982 20.0426 27.3579C21.1318 27.2885 20.9048 25.6682 20.1073 25.3491ZM15.8088 25.4291C16.1078 25.2357 16.2505 24.8581 16.1327 24.5045C15.8887 23.675 14.9194 23.54 14.345 24.2075C14.0694 24.5278 14.092 25.0038 14.3874 25.2968L14.3919 25.3024L14.3966 25.3082C14.7524 25.7507 15.418 25.8971 15.8088 25.4291ZM12.8751 20.5011C13.3304 20.9563 13.7612 21.0411 14.3751 20.8937C15.3732 20.6542 14.621 17.9888 12.8752 18.9967C12.335 19.3086 12.5875 20.2134 12.8751 20.5011Z"
                fill="#F2FAFB"
              />
            </svg>
          </div>

          <Code size="7" weight="medium" className="logo-title">
            chunk_my_docs
          </Code>
        </Flex>
      </Link>

      <Flex className="nav" direction="row" gap="40px" align="center">
        {download && !home && (
          <Text size="4" weight="medium" className="cyan-9 nav-item">
            Download JSON
          </Text>
        )}

        <a href="https://twitter.com/lumina_ai_inc" target="_blank">
          <Text size="4" weight="medium" className="nav-item">
            Contact
          </Text>
        </a>

        <a href="https://twitter.com/lumina_ai_inc" target="_blank">
          <Text size="4" weight="medium" className="nav-item">
            Twitter
          </Text>
        </a>

        <a
          href="https://github.com/lumina-ai-inc/chunk-my-docs"
          target="_blank"
        >
          <Text size="4" weight="medium" className="nav-item">
            Github
          </Text>
        </a>

        <Link to="/pricing" style={{ textDecoration: "none" }}>
          <Text size="4" weight="medium" className="nav-item">
            Pricing
          </Text>
        </Link>

        <Text size="4" weight="medium" className="nav-item">
          Docs
        </Text>

        <Text
          size="4"
          weight="medium"
          className="nav-item"
          onClick={() => setShowAccount(!showAccount)}
          style={{ cursor: "pointer" }}
        >
          Account
        </Text>

        <div className="dropdown-container">
          <DropdownMenu.Root>
            <DropdownMenu.Trigger>
              <Button>Menu</Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              {download && (
                <DropdownMenu.Item>
                  <Text>Download JSON</Text>
                </DropdownMenu.Item>
              )}
              <DropdownMenu.Item>
                <Text>Pricing</Text>
              </DropdownMenu.Item>
              <DropdownMenu.Item>
                <Text>Docs</Text>
              </DropdownMenu.Item>
              <a
                href="https://github.com/lumina-ai-inc/chunk-my-docs"
                target="_blank"
              >
                <DropdownMenu.Item>
                  <Text>Github</Text>
                </DropdownMenu.Item>
              </a>
              <a href="https://twitter.com/lumina_ai_inc" target="_blank">
                <DropdownMenu.Item>
                  <Text>Twitter</Text>
                </DropdownMenu.Item>
              </a>
              <a href="https://twitter.com/lumina_ai_inc" target="_blank">
                <DropdownMenu.Item>
                  <Text>Contact</Text>
                </DropdownMenu.Item>
              </a>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </Flex>
      {showAccount && <Account onClose={() => setShowAccount(false)} />}
    </Flex>
  );
}
