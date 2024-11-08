import { useState, useEffect } from "react";
import { DropdownMenu, Flex, Text, Button } from "@radix-ui/themes";
import { Link } from "react-router-dom";
import "./Header.css";
import Dashboard from "../../pages/Dashboard/Dashboard";
import { useAuth } from "react-oidc-context";
// import { downloadJSON } from "../../utils/utils";
import ApiKeyDialog from "../ApiDialog.tsx/ApiKeyDialog";
// import { useTaskQuery } from "../../hooks/useTaskQuery";
import useUser from "../../hooks/useUser";
import { User } from "../../models/user.model";
import { getRepoStats } from "../../services/githubApi";
// import BetterButton from "../BetterButton/BetterButton";
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
  // const { taskId } = useParams<{ taskId: string }>();
  // const { data: taskResponse } = useTaskQuery(taskId);
  const { data: user } = useUser();
  const [repoStats, setRepoStats] = useState({ stars: 0, forks: 0 });

  useEffect(() => {
    const fetchStats = async () => {
      const stats = await getRepoStats();
      setRepoStats(stats);
    };
    fetchStats();
  }, []);

  // const handleDownloadJSON = () => {
  //   if (taskResponse?.output) {
  //     downloadJSON(
  //       taskResponse.output,
  //       `${taskResponse.file_name?.slice(0, -4)}.json`
  //     );
  //   }
  // };

  // const handleGithubRedirect = () => {
  //   window.open("https://github.com/lumina-ai-inc/chunk-my-docs", "_blank");
  // };

  return (
    <Flex direction="row" justify="between" py="12px" className="header">
      <Link to="/" style={{ textDecoration: "none" }}>
        <div className="logo-container">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="30"
            height="30"
            viewBox="0 0 30 30"
            fill="none"
          >
            <path
              d="M7.35 12.225C8.03148 12.4978 8.77803 12.5646 9.4971 12.4171C10.2162 12.2695 10.8761 11.9142 11.3952 11.3952C11.9142 10.8761 12.2695 10.2162 12.4171 9.4971C12.5646 8.77803 12.4978 8.03148 12.225 7.35C13.0179 7.13652 13.7188 6.6687 14.2201 6.01836C14.7214 5.36802 14.9954 4.57111 15 3.75C17.225 3.75 19.4001 4.4098 21.2502 5.64597C23.1002 6.88213 24.5422 8.63914 25.3936 10.6948C26.2451 12.7505 26.4679 15.0125 26.0338 17.1948C25.5998 19.3771 24.5283 21.3816 22.955 22.955C21.3816 24.5283 19.3771 25.5998 17.1948 26.0338C15.0125 26.4679 12.7505 26.2451 10.6948 25.3936C8.63914 24.5422 6.88213 23.1002 5.64597 21.2502C4.4098 19.4001 3.75 17.225 3.75 15C4.57111 14.9954 5.36802 14.7214 6.01836 14.2201C6.6687 13.7188 7.13652 13.0179 7.35 12.225Z"
              stroke="url(#paint0_linear_236_740)"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            />
            <defs>
              <linearGradient
                id="paint0_linear_236_740"
                x1="15"
                y1="3.75"
                x2="15"
                y2="26.25"
                gradientUnits="userSpaceOnUse"
              >
                <stop stop-color="white" />
                <stop offset="1" stop-color="#DCE4DD" />
              </linearGradient>
            </defs>
          </svg>
          <Text size="6" weight="bold" className="logo-title" trim="start">
            chunkr
          </Text>
        </div>
      </Link>

      <Flex className="nav" direction="row" gap="24px" align="center">
        <Flex
          direction="row"
          gap="4"
          align="center"
          className="auth-container"
          justify="end"
        >
          <Flex direction="row" gap="2" py="12px" px="16px" align="center">
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

          <Flex direction="row" gap="2" py="12px" px="16px" align="center">
            <a
              href="https://cal.com/mehulc/30min"
              target="_blank"
              className="nav-item"
            >
              <Text size="2" weight="medium" className="nav-item">
                Contact
              </Text>
            </a>
          </Flex>

          <Flex direction="row" gap="2" py="12px" px="16px" align="center">
            <Link to="/pricing" style={{ textDecoration: "none" }}>
              <Text size="2" weight="medium" className="nav-item">
                Pricing
              </Text>
            </Link>
          </Flex>

          <Flex direction="row" gap="2" py="12px" px="16px" align="center">
            <a
              href={"https://docs.chunkr.ai"}
              target="_blank"
              className="nav-item"
            >
              <Text size="2" weight="medium" className="nav-item">
                Docs
              </Text>
            </a>
          </Flex>

          <Flex direction="row" gap="2" py="12px" px="16px" align="center">
            <Link to="/pricing" style={{ textDecoration: "none" }}>
              <Text size="2" weight="medium" className="nav-item">
                Upload
              </Text>
            </Link>
          </Flex>

          <Flex direction="row" gap="2" py="12px" px="16px" align="center">
            <Link to="/pricing" style={{ textDecoration: "none" }}>
              <Text size="2" weight="medium" className="nav-item">
                API Keys
              </Text>
            </Link>
          </Flex>

          {/* <BetterButton onClick={handleGithubRedirect}></BetterButton> */}

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
                gap="2"
                py="12px"
                px="16px"
                align="center"
                onClick={() => setShowAccount(!showAccount)}
              >
                <Text size="2" weight="medium" style={{ cursor: "pointer" }}>
                  Dashboard
                </Text>
              </Flex>
            </Link>
          ) : (
            <Flex
              direction="row"
              gap="2"
              py="12px"
              px="16px"
              align="center"
              onClick={() => auth.signinRedirect()}
            >
              <Text
                size="2"
                weight="medium"
                style={{ cursor: "pointer", color: "white" }}
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
                  className="signup-button-dropdown"
                  onSelect={() => auth.signinRedirect()}
                >
                  <Text
                    weight="regular"
                    size="2"
                    style={{ cursor: "pointer", textAlign: "left" }}
                  >
                    Login
                  </Text>
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
