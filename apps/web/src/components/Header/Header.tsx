import { DropdownMenu, Flex, Text, Button } from "@radix-ui/themes";
import { Link } from "react-router-dom";
import "./Header.css";

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
  return (
    <Flex direction="row" justify="between" py={py} px={px} className="header">
      <Link to="/" style={{ textDecoration: "none" }}>
        <Flex className="logo" direction="row" gap="4" align="center">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="40"
            height="40"
            viewBox="0 0 40 40"
            fill="none"
            className="logo-icon"
          >
            <rect width="40" height="40" fill="white" fill-opacity="0.01" />
            <path
              fill-rule="evenodd"
              clip-rule="evenodd"
              d="M32.1859 4.71323C33.2819 3.25193 30.5614 1.2115 29.4654 2.67281C28.3694 4.13409 31.0902 6.17451 32.1859 4.71323ZM22.3804 2.87497C19.0823 2.53495 15.4195 3.10262 12.2249 4.55833C9.02807 6.01502 6.20628 8.40825 4.76919 11.7822C3.66865 14.366 2.71241 16.7497 2.66902 19.8494C2.62594 22.9292 3.4806 26.584 5.68465 31.808C6.9638 34.8397 10.7724 36.1189 13.7039 36.8211C17.089 37.6317 20.8229 37.8437 23.1051 37.5312C24.9737 37.2752 27.0528 36.8757 28.7966 35.4587C31.7272 33.0771 34.4024 30.388 35.9984 27.5715C37.6107 24.7259 38.2016 21.5789 36.5909 18.542C36.1675 17.7433 35.5093 17.5396 34.7765 17.3128C34.4997 17.2271 34.212 17.1381 33.9224 17.0124C30.9896 15.7396 29.2389 14.2657 29.1027 13.41C28.9749 12.6082 29.1256 12.039 29.2829 11.4448C29.3677 11.1253 29.4541 10.7986 29.5003 10.4246C29.5568 9.96633 29.5504 8.88851 28.5323 8.3403C27.7461 7.91702 27.2437 7.08915 26.7752 6.31745C26.6598 6.12734 26.5465 5.94062 26.4315 5.76422C25.6703 4.59646 24.5539 3.09902 22.3804 2.87497ZM13.3307 6.98494C10.5729 8.24155 8.33479 10.2159 7.22257 12.8272C6.12823 15.3965 5.37084 17.3566 5.33545 19.8868C5.29977 22.4369 6.00116 25.6981 8.14161 30.7715C9.04484 32.9123 12.3245 33.7485 14.325 34.2277C17.4648 34.9797 20.8587 35.1472 22.7432 34.8891C24.5672 34.6392 25.9885 34.3045 27.1149 33.3891C29.944 31.0899 32.3227 28.6493 33.6782 26.2569C34.4931 24.8189 36.0376 20.5408 33.6779 19.6995C28.6771 17.9167 27.3654 16.1778 26.6403 14.4338C26.1577 13.2732 26.2098 12.0609 26.5531 10.8769C26.6043 10.7003 26.6544 10.5278 26.7557 10.3702C26.1829 9.9647 25.7373 9.47187 25.3724 8.99067C25.1434 8.68875 24.9251 8.32265 24.6986 7.94283C24.0498 6.85473 23.3338 5.65406 22.1069 5.52755C19.2771 5.23582 16.0906 5.72726 13.3307 6.98494ZM35.2216 10.0659C35.8427 8.60457 33.2125 7.0295 32.2192 8.27102C30.8845 9.93953 34.4078 11.9808 35.2216 10.0659ZM38.3126 6.0275C39.9115 6.06649 40.4136 2.97609 38.8582 2.53166C36.8203 1.94941 36.2448 5.97705 38.3126 6.0275ZM38.0952 13.576C39.6384 13.6001 40.1232 11.6859 38.6219 11.4106C36.6547 11.05 36.0995 13.5447 38.0952 13.576ZM18.42 11.4313C17.2506 12.4468 16.1392 11.0716 15.9462 9.96401C15.6493 8.25985 16.8986 6.8495 18.5286 7.76417C19.8225 8.49019 19.3781 10.5992 18.42 11.4313ZM13.1855 15.6849C14.4236 14.4468 12.2551 12.187 10.9713 13.4708C10.2224 14.2198 10.6916 15.0767 11.2999 15.6849C11.8206 16.2056 12.6648 16.2056 13.1855 15.6849ZM18.6058 17.9196C19.4982 19.7044 22.6348 18.1109 21.3177 16.3944C20.9739 15.9462 20.4975 15.8651 20.3239 15.8455C19.2325 15.7227 18.036 16.78 18.6058 17.9196ZM18.6346 23.2656C19.0418 24.3168 20.0299 24.8499 21.1177 24.4355C22.0649 24.0746 22.2183 22.3917 21.7913 21.5314C21.3955 20.7337 20.3722 20.4745 19.5666 20.7845C18.627 21.146 18.2846 22.3656 18.6287 23.2504L18.6346 23.2656ZM27.7027 23.5755C28.6867 23.0843 30.5646 23.5406 30.5646 24.832C30.5646 25.901 29.2232 27.244 28.1691 27.4203C27.1501 27.5907 25.8804 26.6022 26.044 25.5392C26.1719 24.7077 26.9912 23.9307 27.7027 23.5755ZM20.1788 28.9152C19.559 28.6672 19.0494 28.9603 18.9717 29.0051L18.9675 29.0075C18.4209 29.3197 17.9036 29.9837 17.7992 30.6245C17.5738 32.0091 19.0187 32.3304 20.071 32.2632C21.8864 32.1475 21.5079 29.4469 20.1788 28.9152ZM13.0147 29.0485C13.5131 28.7261 13.7509 28.0968 13.5545 27.5075C13.1479 26.125 11.5323 25.9001 10.575 27.0125C10.1156 27.5464 10.1533 28.3397 10.6457 28.828L10.6532 28.8373L10.6611 28.8469C11.254 29.5845 12.3633 29.8285 13.0147 29.0485ZM8.12521 20.8352C8.88393 21.5939 9.60207 21.7351 10.6252 21.4896C12.2887 21.0903 11.0351 16.648 8.12535 18.3279C7.22505 18.8477 7.64577 20.3557 8.12521 20.8352Z"
              fill="#F2FAFB"
              className="logo-path"
            />
          </svg>
          <Text size="7" weight="medium" className="logo-title cyan-2">
            Chunk My Docs
          </Text>
        </Flex>
      </Link>

      <Flex className="nav" direction="row" gap="40px" align="center">
        {download && (
          <Text size="5" weight="medium" className="cyan-2 nav-item">
            Download JSON
          </Text>
        )}
        {home && (
          <Text size="5" weight="medium" className="cyan-2 nav-item">
            Demo PDF
          </Text>
        )}
        <Text size="5" weight="medium" className="cyan-2 nav-item">
          Pricing
        </Text>
        <Text size="5" weight="medium" className="cyan-2 nav-item">
          Docs
        </Text>
        <a
          href="https://github.com/lumina-ai-inc/chunk-my-docs"
          target="_blank"
        >
          <Text size="5" weight="medium" className="cyan-2 nav-item">
            Github
          </Text>
        </a>

        <a href="https://twitter.com/lumina_ai_inc" target="_blank">
          <Text size="5" weight="medium" className="cyan-2 nav-item">
            Twitter
          </Text>
        </a>

        <a href="https://twitter.com/lumina_ai_inc" target="_blank">
          <Text size="5" weight="medium" className="cyan-2 nav-item">
            Contact
          </Text>
        </a>
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
              {home && (
                <DropdownMenu.Item>
                  <Text>Demo PDF</Text>
                </DropdownMenu.Item>
              )}
              <DropdownMenu.Item>
                <Text>Pricing</Text>
              </DropdownMenu.Item>
              <DropdownMenu.Item>
                <Text>Docs</Text>
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </Flex>
    </Flex>
  );
}
