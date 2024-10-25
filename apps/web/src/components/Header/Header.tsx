import { useState, useEffect } from "react";
import { DropdownMenu, Flex, Text, Button } from "@radix-ui/themes";
import { Link, useParams } from "react-router-dom";
import "./Header.css";
import Dashboard from "../../pages/Dashboard/Dashboard";
import { useAuth } from "react-oidc-context";
import { downloadJSON } from "../../utils/utils";
import ApiKeyDialog from "../ApiDialog.tsx/ApiKeyDialog";
import { useTaskQuery } from "../../hooks/useTaskQuery";
import useUser from "../../hooks/useUser";
import { User } from "../../models/user.model";
import { getRepoStats } from "../../services/githubApi";
import BetterButton from "../BetterButton/BetterButton";
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
  const { data: user } = useUser();
  const [repoStats, setRepoStats] = useState({ stars: 0, forks: 0 });

  useEffect(() => {
    const fetchStats = async () => {
      const stats = await getRepoStats();
      setRepoStats(stats);
    };
    fetchStats();
  }, []);

  const handleDownloadJSON = () => {
    if (taskResponse?.output) {
      downloadJSON(
        taskResponse.output,
        `${taskResponse.file_name?.slice(0, -4)}.json`
      );
    }
  };

  const handleGithubRedirect = () => {
    window.open("https://github.com/lumina-ai-inc/chunk-my-docs", "_blank");
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
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
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
        ml={isAuthenticated ? "88px" : "40px"}
        align="center"
      >
        <a href={"https://docs.chunkr.ai"} target="_blank" className="nav-item">
          <Text size="2" weight="medium" className="nav-item">
            Docs
          </Text>
        </a>

        <a
          href="https://cal.com/mehulc/30min"
          target="_blank"
          className="nav-item"
        >
          <Text size="2" weight="medium" className="nav-item">
            Contact
          </Text>
        </a>

        <Link to="/pricing" style={{ textDecoration: "none" }}>
          <Text size="2" weight="medium" className="nav-item">
            Pricing
          </Text>
        </Link>

        {/* <a
          href="https://github.com/lumina-ai-inc/chunk-my-docs"
          target="_blank"
          className="nav-item"
        >
          <Flex align="baseline" gap="2" className="nav-item github-stats">
            <Text size="2" weight="medium">
              Github
            </Text>
          </Flex>
        </a> */}

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
      </Flex>

      <Flex className="nav" direction="row" gap="24px" align="center">
        <Flex
          direction="row"
          gap="2"
          align="center"
          className="auth-container"
          justify="end"
        >
          <BetterButton onClick={handleGithubRedirect}>
            <Flex direction="row" gap="2" align="center">
              <svg
                width="18"
                height="18"
                viewBox="0 0 18 18"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <g clip-path="url(#clip0_132_8)">
                  <path
                    fill-rule="evenodd"
                    clip-rule="evenodd"
                    d="M8.97318 0C4.01125 0 0 4.04082 0 9.03986C0 13.0359 2.57014 16.4184 6.13561 17.6156C6.58139 17.7056 6.74467 17.4211 6.74467 17.1817C6.74467 16.9722 6.72998 16.2538 6.72998 15.5053C4.23386 16.0442 3.71406 14.4277 3.71406 14.4277C3.31292 13.3801 2.71855 13.1108 2.71855 13.1108C1.90157 12.557 2.77806 12.557 2.77806 12.557C3.68431 12.6169 4.15984 13.4849 4.15984 13.4849C4.96194 14.8618 6.25445 14.4727 6.77443 14.2332C6.84863 13.6495 7.08649 13.2454 7.33904 13.021C5.3482 12.8114 3.25359 12.0332 3.25359 8.56084C3.25359 7.57304 3.60992 6.76488 4.17453 6.13635C4.08545 5.9119 3.77339 4.9838 4.2638 3.74161C4.2638 3.74161 5.02145 3.5021 6.7298 4.66953C7.4612 4.47165 8.21549 4.37099 8.97318 4.37014C9.73084 4.37014 10.5032 4.47502 11.2164 4.66953C12.9249 3.5021 13.6826 3.74161 13.6826 3.74161C14.173 4.9838 13.8607 5.9119 13.7717 6.13635C14.3511 6.76488 14.6928 7.57304 14.6928 8.56084C14.6928 12.0332 12.5982 12.7963 10.5924 13.021C10.9194 13.3053 11.2015 13.844 11.2015 14.6972C11.2015 15.9094 11.1868 16.8823 11.1868 17.1816C11.1868 17.4211 11.3503 17.7056 11.7959 17.6158C15.3613 16.4182 17.9315 13.0359 17.9315 9.03986C17.9462 4.04082 13.9202 0 8.97318 0Z"
                    fill="white"
                  />
                </g>
                <defs>
                  <clipPath id="clip0_132_8">
                    <rect width="18" height="17.6327" fill="white" />
                  </clipPath>
                </defs>
              </svg>
              <Text size="1" weight="bold" className="nav-item">
                {repoStats.stars >= 1000
                  ? `${(repoStats.stars / 1000).toFixed(1)}K`
                  : repoStats.stars}
              </Text>
            </Flex>
          </BetterButton>
          <a
            href="https://github.com/lumina-ai-inc/chunk-my-docs"
            target="_blank"
            className="nav-item"
            style={{
              display: "flex",
              flexDirection: "row",
              gap: "8px",
              alignItems: "center",
              cursor: "pointer",
            }}
          ></a>

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
              <Button
                className="nav-item-right"
                onClick={() => setShowAccount(!showAccount)}
              >
                <Text size="2" weight="medium" style={{ cursor: "pointer" }}>
                  Dashboard
                </Text>
              </Button>
            </Link>
          ) : (
            <Button
              className="nav-item-right"
              onClick={() => auth.signinRedirect()}
            >
              <Text size="2" weight="medium" style={{ cursor: "pointer" }}>
                Login
              </Text>
            </Button>
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
                <a
                  href="https://cal.com/mehulc/30min"
                  target="_blank"
                  style={{ textDecoration: "none" }}
                >
                  <Text>Contact</Text>
                </a>
              </DropdownMenu.Item>
              <DropdownMenu.Item asChild>
                <a
                  href="https://twitter.com/lumina_ai_inc"
                  target="_blank"
                  style={{ textDecoration: "none" }}
                >
                  <Text>Twitter</Text>
                </a>
              </DropdownMenu.Item>
              <DropdownMenu.Item asChild>
                <a
                  href="https://github.com/lumina-ai-inc/chunk-my-docs"
                  target="_blank"
                  style={{ textDecoration: "none" }}
                >
                  <Text>Github</Text>
                </a>
              </DropdownMenu.Item>
              <DropdownMenu.Item asChild>
                <Link to="/pricing" style={{ textDecoration: "none" }}>
                  <Text>Pricing</Text>
                </Link>
              </DropdownMenu.Item>
              {isAuthenticated && (
                <DropdownMenu.Item>
                  <ApiKeyDialog
                    user={user as User}
                    showApiKey={showApiKey}
                    setShowApiKey={setShowApiKey}
                    phone={true}
                  />
                </DropdownMenu.Item>
              )}
              <DropdownMenu.Item>
                <a
                  href={"https://docs.chunkr.ai/introduction"}
                  target="_blank"
                  style={{ textDecoration: "none", color: "inherit" }}
                >
                  <Text weight="regular">Docs</Text>
                </a>
              </DropdownMenu.Item>
              {isAuthenticated ? (
                <DropdownMenu.Item>
                  <Link
                    to="/dashboard"
                    style={{ textDecoration: "none", color: "inherit" }}
                    onClick={() => setShowAccount(!showAccount)}
                  >
                    <Text weight="regular">Dashboard</Text>
                  </Link>
                </DropdownMenu.Item>
              ) : (
                <DropdownMenu.Item
                  className="signup-button"
                  onSelect={() => auth.signinRedirect()}
                >
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
