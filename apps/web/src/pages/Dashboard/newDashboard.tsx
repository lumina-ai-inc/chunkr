import { Flex, Text } from "@radix-ui/themes";
import "./newDashboard.css";
import BetterButton from "../../components/BetterButton/BetterButton";
import TableWrapper from "../../components/Table/Table";
import { useAuth } from "react-oidc-context";
import useUser from "../../hooks/useUser";
export default function NewDashboard() {
  const auth = useAuth();
  const user = useUser();

  return (
    <Flex direction="row" width="100%" height="100vh">
      <Flex
        className="dashboard-nav-container"
        align="start"
        direction="column"
      >
        <Flex className="dashboard-nav-header">
          <Flex gap="8px" align="center" justify="center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="28"
              height="28"
              viewBox="0 0 22 22"
              fill="none"
            >
              <path
                d="M5.39 8.965C5.88975 9.16504 6.43722 9.21401 6.96454 9.10584C7.49186 8.99767 7.97584 8.73711 8.35648 8.35648C8.73711 7.97584 8.99767 7.49186 9.10584 6.96454C9.21401 6.43722 9.16504 5.88975 8.965 5.39C9.54645 5.23345 10.0604 4.89038 10.4281 4.41346C10.7957 3.93655 10.9966 3.35215 11 2.75C12.6317 2.75 14.2267 3.23385 15.5835 4.14038C16.9402 5.0469 17.9976 6.33537 18.622 7.84286C19.2464 9.35035 19.4098 11.0092 19.0915 12.6095C18.7732 14.2098 17.9874 15.6798 16.8336 16.8336C15.6798 17.9874 14.2098 18.7732 12.6095 19.0915C11.0092 19.4098 9.35035 19.2464 7.84286 18.622C6.33537 17.9976 5.0469 16.9402 4.14038 15.5835C3.23385 14.2267 2.75 12.6317 2.75 11C3.35215 10.9966 3.93655 10.7957 4.41346 10.4281C4.89038 10.0604 5.23345 9.54645 5.39 8.965Z"
                stroke="url(#paint0_linear_293_747)"
                stroke-width="2.2"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <defs>
                <linearGradient
                  id="paint0_linear_293_747"
                  x1="11"
                  y1="2.75"
                  x2="11"
                  y2="19.25"
                  gradientUnits="userSpaceOnUse"
                >
                  <stop stop-color="white" />
                  <stop offset="1" stop-color="#DCE4DD" />
                </linearGradient>
              </defs>
            </svg>
            <Text size="5" weight="bold" mb="2px" style={{ color: "#FFF" }}>
              chunkr
            </Text>
          </Flex>
        </Flex>
        <Flex className="nav-body" direction="column" justify="between">
          <Flex direction="column">
            <Flex className="dashboard-nav-items" direction="column">
              <Flex className="dashboard-nav-item">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 22 22"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g clip-path="url(#clip0_305_31798)">
                    <path
                      d="M12.75 7.5H21.25"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M12.75 16.5H21.25"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M8.25 4.75H2.75V10.25H8.25V4.75Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M8.25 13.75H2.75V19.25H8.25V13.75Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </g>
                  <defs>
                    <clipPath id="clip0_305_31798">
                      <rect width="24" height="24" fill="white" />
                    </clipPath>
                  </defs>
                </svg>
                <Text size="3" weight="medium" style={{ color: "#FFF" }}>
                  Tasks
                </Text>
              </Flex>
              <Flex className="dashboard-nav-item">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 22 22"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g clip-path="url(#clip0_113_1401)">
                    <path
                      d="M5.25 20.25H6.75C7.30228 20.25 7.75 19.8023 7.75 19.25L7.75 13.75C7.75 13.1977 7.30228 12.75 6.75 12.75H5.25C4.69772 12.75 4.25 13.1977 4.25 13.75L4.25 19.25C4.25 19.8023 4.69772 20.25 5.25 20.25Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M18.25 20.25H19.75C20.3023 20.25 20.75 19.8023 20.75 19.25V9.75C20.75 9.19772 20.3023 8.75 19.75 8.75H18.25C17.6977 8.75 17.25 9.19771 17.25 9.75V19.25C17.25 19.8023 17.6977 20.25 18.25 20.25Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M11.75 20.25H13.25C13.8023 20.25 14.25 19.8023 14.25 19.25L14.25 5.75C14.25 5.19771 13.8023 4.75 13.25 4.75H11.75C11.1977 4.75 10.75 5.19771 10.75 5.75L10.75 19.25C10.75 19.8023 11.1977 20.25 11.75 20.25Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </g>
                  <defs>
                    <clipPath id="clip0_113_1401">
                      <rect
                        width="24"
                        height="24"
                        fill="white"
                        transform="translate(0.5)"
                      />
                    </clipPath>
                  </defs>
                </svg>
                <Text size="3" weight="medium" style={{ color: "#FFF" }}>
                  Usage
                </Text>
              </Flex>
              <Flex className="dashboard-nav-item">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 22 22"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g clip-path="url(#clip0_305_27941)">
                    <path
                      d="M2.75 5.75C2.75 5.19771 3.19772 4.75 3.75 4.75H20.25C20.8023 4.75 21.25 5.19772 21.25 5.75V18.25C21.25 18.8023 20.8023 19.25 20.25 19.25H3.75C3.19772 19.25 2.75 18.8023 2.75 18.25V5.75Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M15.75 13.25H18.25"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M8.25 13.25L5.75 10.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M5.75 13.25L8.25 10.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M13.25 13.25L10.75 10.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M10.75 13.25L13.25 10.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </g>
                  <defs>
                    <clipPath id="clip0_305_27941">
                      <rect width="24" height="24" fill="white" />
                    </clipPath>
                  </defs>
                </svg>
                <Text size="3" weight="medium" style={{ color: "#FFF" }}>
                  API Keys
                </Text>
              </Flex>
              <Flex className="dashboard-nav-item">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 22 22"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <g clip-path="url(#clip0_113_1457)">
                    <path
                      d="M21.75 8.75H3.25V12.25H21.75V8.75Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                    <path
                      d="M6.25 16.25H10.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <path
                      d="M21.75 18.25V5.75C21.75 5.19772 21.3023 4.75 20.75 4.75L4.25 4.75C3.69772 4.75 3.25 5.19772 3.25 5.75V18.25C3.25 18.8023 3.69772 19.25 4.25 19.25H20.75C21.3023 19.25 21.75 18.8023 21.75 18.25Z"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    />
                  </g>
                  <defs>
                    <clipPath id="clip0_113_1457">
                      <rect
                        width="24"
                        height="24"
                        fill="white"
                        transform="translate(0.5)"
                      />
                    </clipPath>
                  </defs>
                </svg>
                <Text size="3" weight="medium" style={{ color: "#FFF" }}>
                  Account & Billing
                </Text>
              </Flex>
            </Flex>
          </Flex>

          <Flex className="profile-section" direction="column">
            <Flex className="profile-menu" direction="column">
              <Flex className="profile-menu-item">
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "rgba(255,255,255,0.8)" }}
                >
                  Manage Payments
                </Text>
              </Flex>
              <Flex className="profile-menu-item">
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "rgba(255,255,255,0.8)" }}
                >
                  Support
                </Text>
              </Flex>
              <Flex className="profile-menu-item">
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "rgba(255,255,255,0.8)" }}
                >
                  Terms of Service
                </Text>
              </Flex>
              <Flex className="profile-menu-item">
                <Text
                  size="2"
                  weight="medium"
                  style={{ color: "rgba(255,255,255,0.8)" }}
                >
                  Logout
                </Text>
              </Flex>
            </Flex>
            <Flex className="profile-info">
              <Flex gap="12px" align="center">
                <Flex direction="column" gap="2px">
                  <Text size="2" weight="bold" style={{ color: "#FFF" }}>
                    mchadda100@gmail.com
                  </Text>
                  <Text size="1" style={{ color: "rgba(255,255,255,0.5)" }}>
                    Free Plan
                  </Text>
                </Flex>
              </Flex>
            </Flex>
          </Flex>
        </Flex>
      </Flex>
      <Flex direction="column" className="main-container">
        <Flex className="main-header">
          <Text size="5" weight="medium" style={{ color: "#FFF" }}>
            Dashboard
          </Text>
          <Flex gap="12px">
            <BetterButton>
              <Text size="1" weight="medium" style={{ color: "#FFF" }}>
                Create Task
              </Text>
            </BetterButton>
            <BetterButton>
              <Text size="1" weight="medium" style={{ color: "#FFF" }}>
                Docs
              </Text>
            </BetterButton>
            <BetterButton>
              <Text size="1" weight="medium" style={{ color: "#FFF" }}>
                Discord
              </Text>
            </BetterButton>
            <BetterButton>
              <Text size="1" weight="medium" style={{ color: "#FFF" }}>
                Github
              </Text>
            </BetterButton>
          </Flex>
        </Flex>
        <Flex className="main-body" p="24px">
          {auth && user && <TableWrapper auth={auth} />}
        </Flex>
      </Flex>
    </Flex>
  );
}
