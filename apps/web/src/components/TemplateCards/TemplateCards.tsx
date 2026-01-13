import { Flex, Text, Card } from "@radix-ui/themes";
import "./TemplateCards.css";

interface TemplateCard {
  id: string;
  title: string;
  description: string;
  prompt: string;
}

interface TemplateCardsProps {
  onSelect: (prompt: string) => void;
}

export default function TemplateCards({ onSelect }: TemplateCardsProps) {
  const templates: TemplateCard[] = [
    {
      id: "analyze-property",
      title: "Analyze a Deal in 2 Minutes",
      description: "Drop a rent roll, P&L or cashflow statement to extract key facts NOI, DSCR, and cap rate instantly.",
      prompt: "What can you tell me from a rent roll?",
    },
    {
      id: "calculate-dscr",
      title: "Calculate DSCR & Cap Spread",
      description: "Get accurate financial numbers and rates to see if a deal pencils before you go deeper",
      prompt: "How do you calculate DSCR and cap spread? Show me an example",
    },
    {
      id: "generate-memo",
      title: "Create an LP Package",
      description: "Turn your deal into an investor-ready memo with financials, risks, and terms.",
      prompt: "What goes into an investor memo? Show me a sample structure.",
    },
    {
      id: "estimate-capital",
      title: "Estimate Capital Access",
      description: "See how much capital your rental equity can support — without refinancing.",
      prompt: "How do you estimate equity capital access from rental properties?",
    },
    {
      id: "model-renovation",
      title: "Model a Renovation",
      description: "See how unit upgrades affect rent, cash flow, and property value.",
      prompt: "How do I know if a renovation is worth it? What should I consider?",
    
    },
    {
      id: "generate-proforma",
      title: "Generate Pro Forma",
      description: "Create forward-looking financial projections with projected income, expenses, and returns for your property deal.",
      prompt: "Can you help me generate a pro forma for my property? What should it include?",
    },
  ];

  return (
    <Flex
      direction="column"
      style={{
        padding: "80px 24px",
        width: "100%",
        backgroundColor: "#fff",
      }}
    >
      <Text
        size="5"
        weight="medium"
        style={{
          marginBottom: "32px",
          color: "#111",
          textAlign: "center",
        }}
      >
        Where do you want to start?
      </Text>

      <Flex
        gap="24px"
        wrap="wrap"
        justify="center"
        style={{ maxWidth: "1000px", margin: "0 auto", width: "100%" }}
      >
        {templates.map((template) => (
          <Card
            key={template.id}
            style={{
              flex: "1 1 calc(50% - 12px)",
              minWidth: "400px",
              maxWidth: "480px",
              padding: "32px",
              cursor: "pointer",
              border: "1px solid #e0e0e0",
              borderRadius: "12px",
              backgroundColor: "#fff",
              transition: "all 0.2s",
            }}
            className="template-card"
            onClick={() => onSelect(template.prompt)}
          >
            <Flex direction="column" gap="12px">
              <Text
                size="5"
                weight="bold"
                style={{ color: "#111", marginBottom: "4px" }}
              >
                {template.title}
              </Text>
              <Text size="3" style={{ color: "#666", lineHeight: "1.6" }}>
                {template.description}
              </Text>
            </Flex>
          </Card>
        ))}
      </Flex>
    </Flex>
  );
}

