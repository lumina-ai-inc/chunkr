import { Flex, Button, Text } from "@radix-ui/themes";
import "./QuickActionButtons.css";

interface QuickActionButtonsProps {
  onAction: (prompt: string) => void;
}

export default function QuickActionButtons({ onAction }: QuickActionButtonsProps) {
  const actions = [
    {
      id: "analyze",
      icon: "📊",
      label: "Analyze Property",
      prompt: "I would like to analyze a property. What documents do I need?",
    },
    {
      id: "calculate",
      icon: "📈",
      label: "Generate Pro Forma",
      prompt: "I need to build Pro Forma for a property. Show me an example.",
    },
    {
      id: "generate",
      icon: "📄",
      label: "Create Investor Memo",
      prompt: "How do you create an investor ready memo? Give me a sample structure.",
    },
    {
      id: "estimate",
      icon: "💰",
      label: "Estimate Equity",
      prompt: "How much property equity can I access without refinancing or selling?",
    },
  ];

  return (
    <Flex
      gap="12px"
      wrap="wrap"
      justify="center"
      style={{
        marginTop: "16px",
        padding: "0 24px",
      }}
    >
      {actions.map((action) => (
        <Button
          key={action.id}
          variant="outline"
          size="2"
          onClick={() => onAction(action.prompt)}
          style={{
            cursor: "pointer",
            backgroundColor: "transparent",
            color: "#111",
            border: "2px solid #111",
            borderRadius: "12px",
            padding: "20px 16px",
            fontSize: "14px",
            fontWeight: 400,
            transition: "all 0.2s",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
          className="quick-action-button"
        >
          <Text size="3" style={{ color: "#111", filter: "grayscale(100%)" }}>
            {action.icon}
          </Text>
          <Text size="2" style={{ color: "#fff" }}>
            {action.label}
          </Text>
        </Button>
      ))}
    </Flex>
  );
}
