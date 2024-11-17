import { Flex, Text } from "@radix-ui/themes";
import "./newDashboard.css";

export default function NewDashboard() {
  return (
    <Flex direction="row" width="100%" height="100vh">
      <Flex className="nav-container">
        <Flex className="nav-header">
          <Flex className="logo-container" gap="6px">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="22"
              height="22"
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
            <Text size="2" weight="bold" style={{ color: "#FFF" }}>
              chunkr
            </Text>
          </Flex>
          <Flex className="nav-container-toggle">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="18"
              height="18"
              viewBox="0 0 18 18"
              fill="none"
            >
              <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M15.744 18.0004H2.24812C1.00725 18.0004 0 16.9931 0 15.7523V2.24813C0 1.00763 1.00725 0 2.24812 0H15.744C16.9849 0 17.9921 1.00763 17.9921 2.24813V15.7523C17.9921 16.9931 16.9849 18.0004 15.744 18.0004ZM7 16.5V1.50038H2.6205C2.00175 1.50038 1.49962 2.0025 1.49962 2.62125V15.3791C1.49962 15.9979 2.00175 16.5 2.6205 16.5H7ZM15.3716 1.50038H9V9V16.5H15.3716C15.9904 16.5 16.4925 15.9979 16.4925 15.3791V2.62125C16.4925 2.0025 15.9904 1.50038 15.3716 1.50038Z"
                fill="white"
              />
              <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M3.56088 11.5406L6.12137 8.98012L3.56088 6.41925L3 7.5L4.5 8.98012L3 10.5L3.28044 11.0203L3.56088 11.5406Z"
                fill="white"
              />
            </svg>
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
}
