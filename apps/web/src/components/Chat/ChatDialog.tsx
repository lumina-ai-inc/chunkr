import { useState, useRef, useEffect } from "react";
import { Flex, Text, Button, TextField } from "@radix-ui/themes";
import { Dialog, DialogTitle, DialogContent } from "@mui/material";
import "./ChatDialog.css";

interface ChatDialogProps {
  open: boolean;
  onClose: () => void;
  initialPrompt?: string;
  onLoginRequest?: () => void;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function ChatDialog({ open, onClose, initialPrompt, onLoginRequest: _onLoginRequest }: ChatDialogProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hi! I'm Orin. I help founders understand their capital options. How can I help you today?",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      inputRef.current?.focus();
      // If initial prompt is provided, send it automatically
      if (initialPrompt) {
        setTimeout(() => {
          handleSendMessage(initialPrompt);
        }, 500);
      }
    }
  }, [open, initialPrompt]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = (content?: string) => {
    const messageContent = content || inputValue.trim();
    if (!messageContent) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: messageContent,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");

    // Simulate assistant response
    setTimeout(() => {
      let assistantResponse = "";
      
      const lowerContent = messageContent.toLowerCase();
      
      if (lowerContent.includes("startup") || lowerContent.includes("runway")) {
        assistantResponse = "Great! For startup runway, we typically look at rental properties, equity, and cashflow. What assets do you already have?";
      } else if (lowerContent.includes("property") || lowerContent.includes("rental")) {
        assistantResponse = "Excellent! Property-based capital is one of the most reliable paths. How many properties do you have, and what's their current equity?";
      } else if (lowerContent.includes("extend") || lowerContent.includes("fund")) {
        assistantResponse = "I can help you explore options to extend your runway without VC. Do you have rental properties, startup equity, or other assets?";
      } else {
        assistantResponse = "I understand. To give you the best options, can you tell me more about what you're trying to fund and what assets you have?";
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: assistantResponse,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        style: {
          borderRadius: "12px",
          maxHeight: "80vh",
          display: "flex",
          flexDirection: "column",
        },
      }}
    >
      <DialogTitle style={{ padding: "20px 24px", borderBottom: "1px solid #e0e0e0" }}>
        <Flex align="center" gap="12px">
          <Text size="5" weight="bold" style={{ color: "#111" }}>
            Talk to Orin
          </Text>
        </Flex>
      </DialogTitle>
      <DialogContent style={{ padding: 0, display: "flex", flexDirection: "column", flex: 1, overflow: "hidden" }}>
        <Flex
          direction="column"
          style={{
            height: "500px",
            display: "flex",
            flexDirection: "column",
          }}
        >
          {/* Messages */}
          <Flex
            direction="column"
            gap="16px"
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "24px",
            }}
          >
            {messages.map((message) => (
              <Flex
                key={message.id}
                direction="column"
                style={{
                  alignSelf: message.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "80%",
                }}
              >
                <Flex
                  style={{
                    padding: "12px 16px",
                    borderRadius: "12px",
                    backgroundColor: message.role === "user" ? "#545454" : "#f0f0f0",
                    color: message.role === "user" ? "#fff" : "#111",
                  }}
                >
                  <Text size="3" style={{ lineHeight: "1.5" }}>
                    {message.content}
                  </Text>
                </Flex>
              </Flex>
            ))}
            <div ref={messagesEndRef} />
          </Flex>

          {/* Input */}
          <Flex
            gap="8px"
            style={{
              padding: "16px 24px",
              borderTop: "1px solid #e0e0e0",
            }}
          >
            <TextField.Root
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              style={{
                flex: 1,
                fontSize: "14px",
              }}
            />
            <Button
              onClick={() => handleSendMessage()}
              disabled={!inputValue.trim()}
              style={{
                backgroundColor: "#545454",
                color: "#fff",
                cursor: inputValue.trim() ? "pointer" : "not-allowed",
                padding: "8px 16px",
              }}
            >
              Send
            </Button>
          </Flex>
        </Flex>
      </DialogContent>
    </Dialog>
  );
}

