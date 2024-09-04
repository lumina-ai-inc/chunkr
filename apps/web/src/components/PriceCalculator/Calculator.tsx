import { Flex, Text, Slider } from "@radix-ui/themes";
import { useState } from "react";

export default function Calculator() {
  const [fastPages, setFastPages] = useState(0);
  const [highQualityPages, setHighQualityPages] = useState(0);
  // const [llmSegments, setLlmSegments] = useState(0);

  const formatNumber = (num: number) => {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  };

  const calculateTotalCost = () => {
    const fastCost = fastPages * 0.002;
    const highQualityCost = highQualityPages * 0.01;
    // const llmCost = llmSegments * 0.012;
    return (fastCost + highQualityCost).toFixed(2);
  };

  return (
    <Flex
      direction="row"
      width="100%"
      gap="8"
      style={{
        border: "3px solid var(--cyan-5)",
        borderRadius: "8px",
        backgroundColor: "hsl(191, 73%, 5%)",
        boxShadow: "0px 0px 20px 0px rgba(0, 0, 0, 1)",
      }}
    >
      <Flex direction="column" width="55%" p="8">
        <Text weight="medium" size="6" className="cyan-4" trim="start">
          API Calculator
        </Text>

        <Flex direction="column" mt="6" mb="5" gap="2">
          <Text size="9" weight="bold" className="cyan-2">
            ${calculateTotalCost()}
          </Text>
          <Text size="6" className="cyan-8">
            Estimated cost
          </Text>
          <Text
            size="2"
            weight="light"
            className="cyan-5"
            style={{ fontStyle: "italic" }}
          >
            *Free storage up to 1M pages
          </Text>
        </Flex>

        <Flex direction="column" gap="4">
          <Flex direction="column" width="100%">
            <Flex direction="column" gap="4" width="100%" mt="4">
              <Flex direction="column">
                <Text size="3" weight="bold" className="cyan-3">
                  FAST{" "}
                  <Text size="2" weight="light" className="cyan-5" mt="1">
                    (1000 pages free)
                  </Text>
                </Text>
                <Text size="2" weight="light" className="cyan-5" mt="1">
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

          <Flex direction="column" width="100%">
            <Flex direction="column" gap="4" width="100%" mt="4">
              <Flex direction="column">
                <Text size="3" weight="bold" className="cyan-3">
                  HIGH QUALITY{" "}
                  <Text size="2" weight="light" className="cyan-5" mt="1">
                    (500 pages free)
                  </Text>
                </Text>
                <Text size="2" weight="light" className="cyan-5" mt="1">
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

          {/* <Flex direction="column" width="100%">
            <Flex direction="column" gap="4" width="100%" mt="4">
              <Flex direction="column">
                <Text size="3" weight="bold" className="cyan-3">
                  LLM ADD-ONS{" "}
                  <Text size="2" weight="light" className="cyan-5" mt="1">
                    (100 segments free)
                  </Text>
                </Text>
                <Text size="2" weight="light" className="cyan-5" mt="1">
                  {formatNumber(llmSegments)} Segments
                </Text>
              </Flex>
              <Slider
                min={0}
                max={1000000}
                step={100}
                value={[llmSegments]}
                onValueChange={(value) => setLlmSegments(value[0])}
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
          </Flex> */}
        </Flex>
      </Flex>
      <Flex
        direction="column"
        width="45%"
        py="8"
        align="center"
        gap="9"
        style={{
          paddingRight: "40px",
          paddingLeft: "16px",
          paddingTop: "106px",
        }}
      >
        <ExplanationSection
          title="Fast"
          price="0.001 / page"
          description="Quick processing for standard documents. Ideal for bulk operations and time-sensitive tasks."
        />
        <ExplanationSection
          title="High Quality"
          price="0.01 / page"
          description="Enhanced processing for complex documents. Perfect for when accuracy and detail are crucial."
        />
        {/* <ExplanationSection
          title="LLM Add-ons"
          price="0.012 / segment"
          description="Advanced AI-powered analysis and insights. Unlock deeper understanding of your documents."
        /> */}
      </Flex>
    </Flex>
  );
}

function ExplanationSection({
  title,
  price,
  description,
}: {
  title: string;
  price: string;
  description: string;
}) {
  return (
    <Flex direction="column" gap="4">
      <Flex
        direction="row"
        justify="between"
        width="fit-content"
        px="12px"
        py="4px"
        style={{
          borderRadius: "4px",
          border: "2px solid var(--cyan-11)",
        }}
      >
        <Text size="3" weight="medium" className="cyan-4">
          {title}
        </Text>
      </Flex>
      <Text size="8" weight="bold" className="cyan-1">
        ${price}
      </Text>
      <Text
        size="5"
        weight="light"
        className="cyan-4"
        trim="both"
        mt="1"
        style={{ fontStyle: "italic" }}
      >
        {description}
      </Text>
    </Flex>
  );
}
