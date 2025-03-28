import "./Loader.css";
import { Flex } from "@radix-ui/themes";
export default function Loader() {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "100%",
        width: "100%",
      }}
    >
      <Flex
        direction="column"
        align="center"
        justify="center"
        className="loader-icon"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="48"
          height="48"
          viewBox="0 0 30 30"
          fill="none"
          style={{
            filter: "drop-shadow(0 0 4px rgba(255, 255, 255, 0.0))",
          }}
        >
          <defs>
            <linearGradient
              id="paint0_linear_236_740"
              x1="15"
              y1="3.75"
              x2="15"
              y2="26.25"
              gradientUnits="userSpaceOnUse"
            >
              <stop stopColor="white" />
              <stop offset="1" stopColor="rgba(220, 228, 221, 0.9)" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="0.4" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          <path
            d="M7.35 12.225C8.03148 12.4978 8.77803 12.5646 9.4971 12.4171C10.2162 12.2695 10.8761 11.9142 11.3952 11.3952C11.9142 10.8761 12.2695 10.2162 12.4171 9.4971C12.5646 8.77803 12.4978 8.03148 12.225 7.35C13.0179 7.13652 13.7188 6.6687 14.2201 6.01836C14.7214 5.36802 14.9954 4.57111 15 3.75C17.225 3.75 19.4001 4.4098 21.2502 5.64597C23.1002 6.88213 24.5422 8.63914 25.3936 10.6948C26.2451 12.7505 26.4679 15.0125 26.0338 17.1948C25.5998 19.3771 24.5283 21.3816 22.955 22.955C21.3816 24.5283 19.3771 25.5998 17.1948 26.0338C15.0125 26.4679 12.7505 26.2451 10.6948 25.3936C8.63914 24.5422 6.88213 23.1002 5.64597 21.2502C4.4098 19.4001 3.75 17.225 3.75 15C4.57111 14.9954 5.36802 14.7214 6.01836 14.2201C6.6687 13.7188 7.13652 13.0179 7.35 12.225Z"
            stroke="url(#paint0_linear_236_740)"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter="url(#glow)"
          />
        </svg>
      </Flex>
    </div>
  );
}
