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
      description: "Drop rent rolls, P&Ls, or cash flow statements. We extract NOI, DSCR, and cap rates in seconds - no spreadsheet work required.",
      prompt: "What can you tell me from a rent roll?",
    },
    {
      id: "generate-proforma",
      title: "Build a Pro Forma",
      description: "Create detailed cash flow projections showing income, expenses, debt service, and investor returns over 5-10 years.",
      prompt: "Can you help me generate a pro forma for my property? What should it include?",
    },
    {
      id: "calculate-dscr",
      title: "Run Scenarios & Stress Tests",
      description: "Model different assumptions - vacancy rates, interest rates, rent growth - to see if the deal still works when things change.",
      prompt: "How do you calculate DSCR and cap spread? Show me an example",
    },
    {
      id: "generate-memo",
      title: "Draft Investor Memo",
      description: "Turn your analysis into professional investment packages with financials, risk highlights, and deal terms.",
      prompt: "What goes into an investor memo? Show me a sample structure.",
    },
    {
      id: "model-renovation",
      title: "Organize Your Partner List",
      description: "Import contacts, tag by investor type (accredited, institutional), and track who you've shared deals with.",
      prompt: "What are some of formats to send email updates to my list of investors?",
    
    },
    {
      id: "estimate-capital",
      title: "Share an Update with Investors",
      description: "Send personalized deal updates, emails and get engagement.",
      prompt: "How do you estimate home-equity capital access from my rental properties?",
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

