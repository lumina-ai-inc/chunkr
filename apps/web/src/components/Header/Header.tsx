import { useState } from "react";
import { DropdownMenu, Flex, Text, Button } from "@radix-ui/themes";
import { Link, useParams } from "react-router-dom";
import { useSelector } from "react-redux";
import { RootState } from "../../store/store";
import "./Header.css";
import Dashboard from "../Dashboard/Dashboard";
import { useAuth } from "react-oidc-context";
import { downloadJSON } from "../../utils/utils";
import ApiKeyDialog from "../ApiDialog.tsx/ApiKeyDialog";
import { useTaskQuery } from "../../hooks/useTaskQuery";

interface HeaderProps {
  py?: string;
  px?: string;
  download?: boolean;
  home?: boolean;
}

export default function Header({
  download = false,
  home = false,
}: HeaderProps) {
  const [showAccount, setShowAccount] = useState(false);
  const [showApiKey, setShowApiKey] = useState(false);
  const auth = useAuth();
  const isAuthenticated = auth.isAuthenticated;
  const { taskId } = useParams<{ taskId: string }>();
  const { data: taskResponse } = useTaskQuery(taskId);
  const user = useSelector((state: RootState) => state.user.data);

  const handleDownloadJSON = () => {
    if (taskResponse?.output) {
      downloadJSON(
        taskResponse.output,
        `${taskResponse.file_name?.slice(0, -4)}.json`
      );
    }
  };

  return (
    <Flex direction="row" justify="between" py="16px" className="header">
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

      <Flex
        className="nav-center"
        direction="row"
        gap="40px"
        ml={isAuthenticated ? "104px" : "0"}
        align="center"
      >
        <a
          href={`${import.meta.env.VITE_API_URL}/redoc`}
          target="_blank"
          className="nav-item"
        >
          <Text size="2" weight="medium" className="nav-item">
            API Docs
          </Text>
        </a>

        <a
          href="https://github.com/lumina-ai-inc/chunk-my-docs"
          target="_blank"
          className="nav-item"
        >
          <Text size="2" weight="medium" className="nav-item">
            Github
          </Text>
        </a>

        <a
          href="https://github.com/lumina-ai-inc/chunk-my-docs"
          target="_blank"
          className="nav-item"
        >
          <Text size="2" weight="medium" className="nav-item">
            Contact
          </Text>
        </a>

        {download && !home && taskResponse?.output && (
          <Text
            size="2"
            weight="medium"
            className="nav-item-download"
            onClick={handleDownloadJSON}
            style={{ cursor: "pointer" }}
          >
            Download JSON
          </Text>
        )}

        {/* <Link
          to="/pricing"
          style={{ textDecoration: "none" }}
          className="nav-item"
        >
          <Text size="2" weight="medium" className="nav-item">
            Pricing
          </Text>
        </Link> */}
      </Flex>

      <Flex className="nav" direction="row" gap="24px" align="center">
        <Flex
          direction="row"
          gap="4"
          align="center"
          className="auth-container"
          justify="end"
        >
          {isAuthenticated && user && (
            <ApiKeyDialog
              user={user}
              showApiKey={showApiKey}
              setShowApiKey={setShowApiKey}
            />
          )}
          {isAuthenticated ? (
            <Link
              to="/dashboard"
              style={{ textDecoration: "none" }}
              className="nav-item"
            >
              <Flex
                direction="row"
                gap="3"
                align="center"
                className="nav-item-right"
              >
                <Text
                  size="2"
                  weight="medium"
                  onClick={() => setShowAccount(!showAccount)}
                  style={{ cursor: "pointer" }}
                >
                  Dashboard
                </Text>
              </Flex>
            </Link>
          ) : (
            <Flex
              direction="row"
              gap="3"
              align="center"
              className="nav-item-right"
            >
              <Text
                size="2"
                weight="medium"
                onClick={() => auth.signinRedirect()}
                style={{ cursor: "pointer" }}
              >
                Login
              </Text>
            </Flex>
          )}
        </Flex>

        <div className="dropdown-container">
          <DropdownMenu.Root>
            <DropdownMenu.Trigger style={{ backgroundColor: "transparent" }}>
              <Button>Menu</Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content>
              {download && !home && (
                <DropdownMenu.Item>
                  <Text>Download JSON</Text>
                </DropdownMenu.Item>
              )}
              <DropdownMenu.Item asChild>
                <a href="https://twitter.com/lumina_ai_inc" target="_blank">
                  <Text>Contact</Text>
                </a>
              </DropdownMenu.Item>
              <DropdownMenu.Item asChild>
                <a href="https://twitter.com/lumina_ai_inc" target="_blank">
                  <Text>Twitter</Text>
                </a>
              </DropdownMenu.Item>
              <DropdownMenu.Item asChild>
                <a
                  href="https://github.com/lumina-ai-inc/chunk-my-docs"
                  target="_blank"
                >
                  <Text>Github</Text>
                </a>
              </DropdownMenu.Item>
              <DropdownMenu.Item asChild>
                <Link to="/pricing">
                  <Text>Pricing</Text>
                </Link>
              </DropdownMenu.Item>
              <DropdownMenu.Item>
                <Text>Docs</Text>
              </DropdownMenu.Item>
              {isAuthenticated ? (
                <DropdownMenu.Item
                  onSelect={() => setShowAccount(!showAccount)}
                >
                  <Text>Dashboard</Text>
                </DropdownMenu.Item>
              ) : (
                <DropdownMenu.Item onSelect={() => auth.signinRedirect()}>
                  <Text>Login</Text>
                </DropdownMenu.Item>
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </Flex>
      {showAccount && <Dashboard />}
    </Flex>
  );
}
