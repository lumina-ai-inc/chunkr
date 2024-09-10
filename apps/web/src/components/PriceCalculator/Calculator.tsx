import { Flex, Text, Slider } from "@radix-ui/themes";
import { useState } from "react";

export default function Calculator() {
  const [fastPages, setFastPages] = useState(0);
  const [highQualityPages, setHighQualityPages] = useState(0);

  const formatNumber = (num: number) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };

  const calculateTotalCost = () => {
    const fastCost = fastPages * 0.002;
    const highQualityCost = highQualityPages * 0.01;
    return (fastCost + highQualityCost).toFixed(2);
  };

  return (
    <Flex
      direction="column"
      width="100%"
      gap="8"
      style={{
        border: "2px solid var(--cyan-5)",
        borderRadius: "8px",
        backgroundColor:
          "hsla(189.47368421052633, 76%, 4.901960784313726%, 0.6)",
        boxShadow: "0px 0px 20px 0px rgba(0, 0, 0, 0.3)",
      }}
    >
      <Flex direction="column" width="100%" p="8">
        <Text weight="bold" size="4" className="cyan-4" trim="start">
          API Calculator
        </Text>

        <Flex direction="column" mt="6" mb="5" gap="2">
          <Text size="9" weight="bold" className="cyan-2">
            ${calculateTotalCost()}
          </Text>
          <Flex direction="row" gap="2" align="end">
            <Text size="6" weight="medium" className="cyan-8">
              Estimated cost
            </Text>
            <Text
              size="2"
              weight="light"
              className="cyan-5"
              style={{ fontStyle: "italic", lineHeight: "1.5" }}
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
                        fill-opacity="0.01"
                      />
                      <path
                        d="M9.31576 0.951108L9.27642 1.04304L9.31576 0.951107C9.04732 0.836241 8.73514 0.919785 8.55994 1.15337L2.15996 9.68667L2.15996 9.68667C2.01603 9.87858 1.99289 10.1353 2.10016 10.3499L2.10017 10.3499C2.20745 10.5645 2.42675 10.7 2.66663 10.7H7.25232L6.30857 16.3625C6.26056 16.6506 6.41569 16.9341 6.68413 17.0489L6.68417 17.049C6.95257 17.1637 7.26475 17.0803 7.43994 16.8467L13.8399 8.31337C13.8399 8.31336 13.8399 8.31335 13.8399 8.31334C13.9839 8.12143 14.0071 7.86468 13.8998 7.65012C13.7925 7.43554 13.5731 7.30001 13.3333 7.30001H8.74757L9.69132 1.63749C9.73933 1.34948 9.5842 1.06597 9.31576 0.951108ZM7.5167 8.34272L7.5167 8.34272C7.63704 8.48476 7.81378 8.56668 7.99994 8.56668H12.0666L7.98499 14.0088L8.62466 10.1708C8.65527 9.98716 8.60352 9.79935 8.48319 9.6573C8.36286 9.51525 8.18611 9.43334 7.99994 9.43334H3.9333L8.01491 3.99119L7.37524 7.82922C7.37524 7.82922 7.37524 7.82922 7.37524 7.82922C7.34462 8.01287 7.39637 8.20066 7.5167 8.34272Z"
                        fill="#0D3C48"
                        stroke="#0D3C48"
                        stroke-width="0.2"
                      />
                    </svg>
                  }
                />
                <Text size="2" weight="light" className="cyan-5" mt="4">
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
                  backgroundColor: "var(--cyan-12)",
                  borderRadius: "8px",
                }}
              />
            </Flex>

            <Flex direction="row" justify="between" width="100%" mt="3">
              <Text size="2" weight="bold" className="cyan-5">
                0
              </Text>
              <Text size="2" weight="bold" className="cyan-5">
                500K
              </Text>
              <Text size="2" weight="bold" className="cyan-5">
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
                        fill-opacity="0.01"
                      />
                      <path
                        d="M8.00222 0.835547C4.04417 0.835547 0.835547 4.04417 0.835547 8.00221C0.835547 11.9602 4.04417 15.1689 8.00222 15.1689C11.9602 15.1689 15.1689 11.9602 15.1689 8.00221C15.1689 4.04417 11.9602 0.835547 8.00222 0.835547ZM4.80003 7.3667H2.0824C2.37738 4.58645 4.58645 2.37738 7.3667 2.0824V4.80003C7.3667 5.14981 7.65024 5.43337 8.00003 5.43337C8.34981 5.43337 8.63337 5.14981 8.63337 4.80003V2.08193C11.4157 2.37508 13.6269 4.58499 13.9221 7.3667H11.2C10.8503 7.3667 10.5667 7.65024 10.5667 8.00003C10.5667 8.34982 10.8503 8.63337 11.2 8.63337H13.9225C13.6292 11.4171 11.4171 13.6292 8.63337 13.9225V11.2C8.63337 10.8503 8.34982 10.5667 8.00003 10.5667C7.65024 10.5667 7.3667 10.8503 7.3667 11.2V13.9221C4.58499 13.6269 2.37508 11.4157 2.08193 8.63337H4.80003C5.14981 8.63337 5.43337 8.34981 5.43337 8.00003C5.43337 7.65024 5.14981 7.3667 4.80003 7.3667Z"
                        fill="#0D3C48"
                        stroke="#0D3C48"
                        stroke-width="0.2"
                      />
                    </svg>
                  }
                />

                <Text size="2" weight="light" className="cyan-5" mt="4">
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
                  backgroundColor: "var(--cyan-12)",
                  borderRadius: "8px",
                }}
              />
            </Flex>

            <Flex direction="row" justify="between" width="100%" mt="3">
              <Text size="2" weight="bold" className="cyan-5">
                0
              </Text>
              <Text size="2" weight="bold" className="cyan-5">
                500K
              </Text>
              <Text size="2" weight="bold" className="cyan-5">
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
          border: "2px solid var(--cyan-2)",
          backgroundColor: "var(--cyan-1)",
        }}
      >
        {icon}
        <Text size="2" weight="bold" className="cyan-12">
          {title}
        </Text>
      </Flex>
      <Text size="7" weight="bold" className="cyan-1">
        ${price}{" "}
        <Text size="3" weight="regular" className="cyan-5">
          {free}
        </Text>
      </Text>
      <Text size="4" weight="regular" className="cyan-2" trim="both">
        {description}
      </Text>
    </Flex>
  );
}
