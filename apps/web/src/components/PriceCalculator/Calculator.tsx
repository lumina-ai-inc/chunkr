import { Flex, Text, Slider } from "@radix-ui/themes";
import { useState } from "react";
import "./Calculator.css";

export default function Calculator() {
  const [fastPages, setFastPages] = useState(0);
  const [highQualityPages, setHighQualityPages] = useState(0);

  const formatNumber = (num: number) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };

  const calculateTotalCost = () => {
    const fastCost = fastPages * 0.005;
    const highQualityCost = highQualityPages * 0.01;
    return (fastCost + highQualityCost).toFixed(2);
  };

  return (
    <Flex
      direction="column"
      width="100%"
      gap="8"
      style={{
        backgroundColor: "hsl(0, 0%, 100%, 0.1)",
        borderRadius: "8px",
        /* border: 2px solid #fff; */
        boxShadow:
          "0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
        backdropFilter: "blur(24px)",
        border: "1px solid hsla(0, 0%, 100%, 0.1)",
      }}
    >
      <Flex direction="column" width="100%" className="calculator-container">
        <Flex
          direction="row"
          gap="4"
          style={{
            padding: "10px 12px",
            borderRadius: "8px",
            backgroundColor: "hsla(180, 100%, 100%, 0.1)",
            width: "fit-content",
          }}
        >
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
              d="M7.99995 2.4C7.99995 1.95817 7.64178 1.6 7.19995 1.6C6.75813 1.6 6.39995 1.95817 6.39995 2.4V11.2C6.39995 11.2267 6.40126 11.2532 6.40382 11.2792C4.5761 11.6483 3.19995 13.2635 3.19995 15.2C3.19995 17.1365 4.5761 18.7517 6.40382 19.1208C6.40126 19.1469 6.39995 19.1733 6.39995 19.2V21.6C6.39995 22.0418 6.75813 22.4 7.19995 22.4C7.64178 22.4 7.99995 22.0418 7.99995 21.6V19.2C7.99995 19.1733 7.99864 19.1469 7.99608 19.1208C9.82379 18.7517 11.2 17.1365 11.2 15.2C11.2 13.2635 9.82379 11.6483 7.99608 11.2792C7.99864 11.2532 7.99995 11.2267 7.99995 11.2V2.4ZM17.6 2.4C17.6 1.95817 17.2417 1.6 16.8 1.6C16.3582 1.6 16 1.95817 16 2.4V4.8C16 4.82673 16.0012 4.85317 16.0038 4.87923C14.1761 5.24835 12.8 6.86347 12.8 8.8C12.8 10.7365 14.1761 12.3516 16.0038 12.7208C16.0012 12.7468 16 12.7733 16 12.8V21.6C16 22.0418 16.3582 22.4 16.8 22.4C17.2417 22.4 17.6 22.0418 17.6 21.6V12.8C17.6 12.7733 17.5987 12.7468 17.5961 12.7208C19.4238 12.3516 20.8 10.7365 20.8 8.8C20.8 6.86347 19.4238 5.24835 17.5961 4.87923C17.5987 4.85317 17.6 4.82673 17.6 4.8V2.4ZM7.19995 12.8C5.87446 12.8 4.79995 13.8745 4.79995 15.2C4.79995 16.5254 5.87446 17.6 7.19995 17.6C8.52544 17.6 9.59995 16.5254 9.59995 15.2C9.59995 13.8745 8.52544 12.8 7.19995 12.8ZM14.4 8.8C14.4 7.47451 15.4745 6.4 16.8 6.4C18.1254 6.4 19.2 7.47451 19.2 8.8C19.2 10.1255 18.1254 11.2 16.8 11.2C15.4745 11.2 14.4 10.1255 14.4 8.8Z"
              fill="white"
            />
          </svg>
          <Text
            weight="bold"
            size="4"
            style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
          >
            Metered API
          </Text>
        </Flex>

        <Flex direction="column" mt="8" mb="5" gap="2">
          <Text
            size="9"
            weight="bold"
            style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
          >
            ${calculateTotalCost()}
          </Text>
          <Flex direction="row" gap="2" align="end">
            <Text
              size="6"
              weight="medium"
              style={{ color: "hsl(0, 0%, 100%, 0.9)" }}
            >
              Estimated cost
            </Text>
            <Text
              size="2"
              weight="light"
              style={{
                color: "hsl(0, 0%, 100%, 0.6)",
                fontStyle: "italic",
                lineHeight: "1.5",
              }}
            >
              *Billed monthly
            </Text>
          </Flex>
        </Flex>

        <Flex direction="column" gap="4">
          <Flex direction="column" width="100%">
            <Flex direction="column" gap="4" width="100%" mt="5">
              <Flex direction="column">
                <ExplanationSection
                  title="Fast"
                  price="0.005 / page"
                  free="1000 free"
                  description="Quick processing for standard documents. Ideal for bulk operations and time-sensitive tasks."
                  icon={
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="18"
                      viewBox="0 0 16 18"
                      fill="none"
                    >
                      <rect
                        width="16"
                        height="16"
                        transform="translate(0 1)"
                        fill="white"
                        fillOpacity="0.01"
                      />
                      <path
                        d="M9.31576 0.951108L9.27642 1.04304L9.31576 0.951107C9.04732 0.836241 8.73514 0.919785 8.55994 1.15337L2.15996 9.68667L2.15996 9.68667C2.01603 9.87858 1.99289 10.1353 2.10016 10.3499L2.10017 10.3499C2.20745 10.5645 2.42675 10.7 2.66663 10.7H7.25232L6.30857 16.3625C6.26056 16.6506 6.41569 16.9341 6.68413 17.0489L6.68417 17.049C6.95257 17.1637 7.26475 17.0803 7.43994 16.8467L13.8399 8.31337C13.8399 8.31336 13.8399 8.31335 13.8399 8.31334C13.9839 8.12143 14.0071 7.86468 13.8998 7.65012C13.7925 7.43554 13.5731 7.30001 13.3333 7.30001H8.74757L9.69132 1.63749C9.73933 1.34948 9.5842 1.06597 9.31576 0.951108ZM7.5167 8.34272L7.5167 8.34272C7.63704 8.48476 7.81378 8.56668 7.99994 8.56668H12.0666L7.98499 14.0088L8.62466 10.1708C8.65527 9.98716 8.60352 9.79935 8.48319 9.6573C8.36286 9.51525 8.18611 9.43334 7.99994 9.43334H3.9333L8.01491 3.99119L7.37524 7.82922C7.37524 7.82922 7.37524 7.82922 7.37524 7.82922C7.34462 8.01287 7.39637 8.20066 7.5167 8.34272Z"
                        fill="#000"
                        stroke="#000"
                        strokeWidth="0.2"
                      />
                    </svg>
                  }
                />
                <Text
                  size="2"
                  weight="light"
                  style={{ color: "hsla(180, 100%, 100%, 0.8)" }}
                  mt="4"
                >
                  {formatNumber(fastPages)} Pages
                </Text>
              </Flex>
              <Slider
                min={0}
                max={1000000}
                step={100}
                value={[fastPages]}
                onValueChange={(value) => setFastPages(value[0])}
                style={{
                  backgroundColor: "hsla(180, 100%, 100%, 0.1)",
                  color: "hsla(180, 100%, 100%, 0.1)",
                  borderRadius: "8px",
                }}
              />
            </Flex>

            <Flex direction="row" justify="between" width="100%" mt="3">
              <Text
                size="2"
                weight="bold"
                style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
              >
                0
              </Text>
              <Text
                size="2"
                weight="bold"
                style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
              >
                500K
              </Text>
              <Text
                size="2"
                weight="bold"
                style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
              >
                1M
              </Text>
            </Flex>
          </Flex>

          <Flex direction="column" width="100%" mt="6">
            <Flex direction="column" gap="4" width="100%" mt="4">
              <Flex direction="column">
                <ExplanationSection
                  title="High Quality"
                  price="0.01 / page"
                  free="500 free"
                  description="Enhanced processing for complex documents. Perfect for when accuracy and detail are crucial."
                  icon={
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="16"
                      height="16"
                      viewBox="0 0 16 16"
                      fill="none"
                    >
                      <rect
                        width="16"
                        height="16"
                        fill="white"
                        fillOpacity="0.01"
                      />
                      <path
                        d="M8.00222 0.835547C4.04417 0.835547 0.835547 4.04417 0.835547 8.00221C0.835547 11.9602 4.04417 15.1689 8.00222 15.1689C11.9602 15.1689 15.1689 11.9602 15.1689 8.00221C15.1689 4.04417 11.9602 0.835547 8.00222 0.835547ZM4.80003 7.3667H2.0824C2.37738 4.58645 4.58645 2.37738 7.3667 2.0824V4.80003C7.3667 5.14981 7.65024 5.43337 8.00003 5.43337C8.34981 5.43337 8.63337 5.14981 8.63337 4.80003V2.08193C11.4157 2.37508 13.6269 4.58499 13.9221 7.3667H11.2C10.8503 7.3667 10.5667 7.65024 10.5667 8.00003C10.5667 8.34982 10.8503 8.63337 11.2 8.63337H13.9225C13.6292 11.4171 11.4171 13.6292 8.63337 13.9225V11.2C8.63337 10.8503 8.34982 10.5667 8.00003 10.5667C7.65024 10.5667 7.3667 10.8503 7.3667 11.2V13.9221C4.58499 13.6269 2.37508 11.4157 2.08193 8.63337H4.80003C5.14981 8.63337 5.43337 8.34981 5.43337 8.00003C5.43337 7.65024 5.14981 7.3667 4.80003 7.3667Z"
                        fill="#000"
                        stroke="#000"
                        strokeWidth="0.2"
                      />
                    </svg>
                  }
                />

                <Text
                  size="2"
                  weight="light"
                  style={{ color: "hsla(180, 100%, 100%, 0.8)" }}
                  mt="4"
                >
                  {formatNumber(highQualityPages)} Pages
                </Text>
              </Flex>
              <Slider
                min={0}
                max={1000000}
                step={100}
                value={[highQualityPages]}
                onValueChange={(value) => setHighQualityPages(value[0])}
                style={{
                  backgroundColor: "hsla(180, 100%, 100%, 0.1)",
                  borderRadius: "8px",
                }}
              />
            </Flex>

            <Flex direction="row" justify="between" width="100%" mt="3">
              <Text
                size="2"
                weight="bold"
                style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
              >
                0
              </Text>
              <Text
                size="2"
                weight="bold"
                style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
              >
                500K
              </Text>
              <Text
                size="2"
                weight="bold"
                style={{ color: "hsla(180, 100%, 100%, 0.9)" }}
              >
                1M
              </Text>
            </Flex>
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
}

function ExplanationSection({
  title,
  price,
  description,
  free,
  icon,
}: {
  title: string;
  price: string;
  description: string;
  free: string;
  icon: React.ReactNode;
}) {
  return (
    <Flex direction="column" gap="3">
      <Flex
        direction="row"
        justify="between"
        align="center"
        width="fit-content"
        gap="2"
        px="12px"
        py="4px"
        style={{
          borderRadius: "99px",
          border: "2px solid hsla(180, 100%, 100%, 0.15)",
          backgroundColor: "hsla(180, 100%, 100%, 0.9)",
        }}
      >
        {icon}
        <Text size="2" weight="bold" style={{ color: "hsla(0, 0%, 0%, 0.9)" }}>
          {title}
        </Text>
      </Flex>
      <Text size="7" weight="bold" style={{ color: "hsl(0, 0%, 100%, 0.98)" }}>
        ${price}{" "}
        <Text
          size="3"
          weight="regular"
          style={{ color: "hsl(0, 0%, 100%, 0.6)" }}
        >
          {free}
        </Text>
      </Text>
      <Text
        size="4"
        weight="regular"
        style={{ color: "hsl(0, 0%, 100%, 0.9)" }}
        trim="both"
      >
        {description}
      </Text>
    </Flex>
  );
}
