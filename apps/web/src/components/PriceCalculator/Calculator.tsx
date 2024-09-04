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
          icon={
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
            >
              <rect width="16" height="16" fill="white" fill-opacity="0.01" />
              <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M9.27642 0.0430443C9.50247 0.139774 9.63311 0.378514 9.59268 0.621051L8.62952 6.40001H13.3333C13.5352 6.40001 13.72 6.51414 13.8103 6.69484C13.9007 6.87552 13.8812 7.09173 13.7599 7.25334L7.35994 15.7867C7.21242 15.9834 6.94953 16.0537 6.72347 15.957C6.49742 15.8603 6.36678 15.6215 6.40721 15.379L7.37037 9.6H2.66663C2.46463 9.6 2.27995 9.48587 2.18961 9.30518C2.09927 9.1245 2.11876 8.90828 2.23996 8.74667L8.63994 0.213374C8.78747 0.0166672 9.05036 -0.0536855 9.27642 0.0430443ZM3.7333 8.53334H7.99994C8.15672 8.53334 8.30555 8.60231 8.40689 8.72194C8.50822 8.84156 8.55179 8.99971 8.52602 9.15435L7.81893 13.3969L12.2666 7.46668H7.99994C7.84318 7.46668 7.69434 7.3977 7.593 7.27808C7.49167 7.15845 7.44809 7.00031 7.47387 6.84566L8.18097 2.60311L3.7333 8.53334Z"
                fill="#3DB9CF"
              />
            </svg>
          }
        />
        <ExplanationSection
          title="High Quality"
          price="0.01 / page"
          description="Enhanced processing for complex documents. Perfect for when accuracy and detail are crucial."
          icon={
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 16 16"
              fill="none"
            >
              <rect width="16" height="16" fill="white" fill-opacity="0.01" />
              <path
                fill-rule="evenodd"
                clip-rule="evenodd"
                d="M0.935547 8.00221C0.935547 4.0994 4.0994 0.935547 8.00222 0.935547C11.905 0.935547 15.0689 4.0994 15.0689 8.00221C15.0689 11.905 11.905 15.0689 8.00222 15.0689C4.0994 15.0689 0.935547 11.905 0.935547 8.00221ZM1.97225 7.4667C2.22784 4.55082 4.55082 2.22784 7.4667 1.97225V4.80003C7.4667 5.09458 7.70547 5.33337 8.00003 5.33337C8.29458 5.33337 8.53337 5.09458 8.53337 4.80003V1.97186C11.4513 2.22553 13.7764 4.54935 14.0322 7.4667H11.2C10.9055 7.4667 10.6667 7.70547 10.6667 8.00003C10.6667 8.29458 10.9055 8.53337 11.2 8.53337H14.0325C13.7788 11.4527 11.4527 13.7788 8.53337 14.0325V11.2C8.53337 10.9055 8.29458 10.6667 8.00003 10.6667C7.70547 10.6667 7.4667 10.9055 7.4667 11.2V14.0322C4.54935 13.7764 2.22553 11.4513 1.97186 8.53337H4.80003C5.09458 8.53337 5.33337 8.29458 5.33337 8.00003C5.33337 7.70547 5.09458 7.4667 4.80003 7.4667H1.97225Z"
                fill="#3DB9CF"
              />
            </svg>
          }
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
  icon,
}: {
  title: string;
  price: string;
  description: string;
  icon: React.ReactNode;
}) {
  return (
    <Flex direction="column" gap="4">
      <Flex
        direction="row"
        justify="between"
        align="center"
        width="fit-content"
        gap="2"
        px="10px"
        py="4px"
        style={{
          borderRadius: "4px",
          border: "2px solid var(--cyan-8)",
        }}
      >
        {icon}
        <Text size="3" weight="medium" className="cyan-8">
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
