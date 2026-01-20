import { useState, useRef, useEffect } from "react";
import { Flex, Text, Button } from "@radix-ui/themes";
import { ChatMessage } from "../../services/chatApi";
import "./MiddleChatPane.css";

interface MiddleChatPaneProps {
  dealId: string | null;
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  onFileUpload?: (file: File) => void;
  onPreviewUpdate?: (type: string, data: any) => void;
}

interface ExpandableActionProps {
  status: "pending" | "running" | "completed" | "failed";
  label: string;
  icon?: string;
  children?: React.ReactNode;
}

function ExpandableAction({ status, label, icon, children }: ExpandableActionProps) {
  const [isExpanded, setIsExpanded] = useState(status === "completed");

  const getStatusIcon = () => {
    if (icon) return icon;
    switch (status) {
      case "completed":
        return "✓";
      case "running":
        return "⟳";
      case "failed":
        return "✗";
      default:
        return "○";
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case "completed":
        return "#4CAF50";
      case "running":
        return "#FF9800";
      case "failed":
        return "#f44336";
      default:
        return "#999";
    }
  };

  return (
    <Flex direction="column" gap="8px">
      <Flex
        align="center"
        gap="8px"
        p="8px 12px"
        style={{
          backgroundColor: "#f8f9fa",
          borderRadius: "6px",
          cursor: children ? "pointer" : "default",
        }}
        onClick={() => children && setIsExpanded(!isExpanded)}
      >
        <Text
          size="3"
          weight="bold"
          style={{ color: getStatusColor() }}
        >
          {getStatusIcon()}
        </Text>
        <Text size="3" weight="medium" style={{ flex: 1 }}>
          {label}
        </Text>
        {children && (
          <Text size="2" style={{ color: "#999" }}>
            {isExpanded ? "▲" : "▼"}
          </Text>
        )}
      </Flex>
      {isExpanded && children && (
        <div style={{ marginLeft: "24px", padding: "12px", backgroundColor: "#fff", borderRadius: "8px", border: "1px solid #e0e0e0" }}>
          {children}
        </div>
      )}
    </Flex>
  );
}

export default function MiddleChatPane({
  dealId,
  messages,
  onSendMessage,
  onFileUpload,
}: MiddleChatPaneProps) {
  const [inputValue, setInputValue] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom of messages container (not the whole page)
  useEffect(() => {
    if (messagesContainerRef.current && messagesEndRef.current) {
      messagesContainerRef.current.scrollTo({
        top: messagesContainerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const handleSendMessage = () => {
    const message = inputValue.trim();
    if (!message || !dealId) return;

    onSendMessage(message);
    setInputValue("");
    // Keep focus on input after sending
    setTimeout(() => {
      inputRef.current?.focus();
    }, 100);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0 && onFileUpload) {
      onFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0 && onFileUpload) {
      onFileUpload(files[0]);
    }
  };

  const getPlaceholder = () => {
    if (!dealId) {
      return "Create a new deal or select an existing one to start chatting...";
    }
    return "Message Orin...";
  };

  const renderMessage = (message: ChatMessage) => {
    const isUser = message.role === "user";

    return (
      <Flex
        key={message.id}
        direction="column"
        style={{
          alignSelf: isUser ? "flex-end" : "flex-start",
          maxWidth: "80%",
          width: "fit-content",
          position: "relative",
        }}
      >
        <div
          style={{
            padding: "12px 16px",
            backgroundColor: isUser ? "#111" : "#f5f5f5",
            color: isUser ? "#fff" : "#111",
            borderRadius: "12px",
            border: isUser ? "1px solid #111" : "1px solid #e0e0e0",
            boxShadow: "none",
            overflow: "visible",
            position: "relative",
          }}
          className="chat-message-bubble"
        >
          <Text 
            size="3" 
            style={{ 
              lineHeight: "1.5", 
              whiteSpace: "pre-wrap",
              color: isUser ? "#fff" : "#111",
            }}
          >
            {message.content}
          </Text>
        </div>

        {/* Expandable Actions */}
        {message.actions && message.actions.length > 0 && (
          <Flex direction="column" gap="8px" mt="12px">
            {message.actions.map((action) => (
              <ExpandableAction
                key={action.id}
                status={action.status}
                label={action.label}
              >
                {action.data && (
                  <Text size="2" style={{ whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(action.data, null, 2)}
                  </Text>
                )}
              </ExpandableAction>
            ))}
          </Flex>
        )}
      </Flex>
    );
  };

  return (
    <Flex
      direction="column"
      style={{
        flex: 1,
        minWidth: "320px", // Reduced from 400px to compensate for wider right pane
        height: "100vh",
        borderRight: "1px solid #e0e0e0",
        backgroundColor: "#fff",
      }}
    >
      {/* Header */}
      <Flex
        align="center"
        justify="between"
        p="16px 24px"
        style={{
          borderBottom: "1px solid #e0e0e0",
        }}
      >
        <Text size="4" weight="medium">
          {dealId ? "Deal Chat" : "New Deal"}
        </Text>
      </Flex>

      {/* Messages */}
      <div
        ref={messagesContainerRef}
        className="messages-container"
        style={{
          flex: 1,
          padding: "24px",
          overflowY: "auto",
          overflowX: "visible",
          scrollBehavior: "smooth",
        }}
      >
        <Flex direction="column" gap="16px" style={{ paddingBottom: "8px" }}>
          {messages.length === 0 && dealId === null ? (
            <Flex
              direction="column"
              align="center"
              justify="center"
              style={{ height: "100%", padding: "40px", textAlign: "center" }}
            >
              <Text size="5" weight="medium" style={{ marginBottom: "12px" }}>
                Start a New Deal
              </Text>
              <Text size="3" style={{ color: "#666", maxWidth: "400px" }}>
                Create a new deal to analyze properties, extract facts, calculate underwriting metrics, and generate investor packages.
              </Text>
            </Flex>
          ) : (
            messages.map((message) => renderMessage(message))
          )}
          <div ref={messagesEndRef} />
        </Flex>
      </div>

      {/* Input Area - Outer container with border, buttons below text field */}
      <Flex
        p="16px 24px"
        style={{
          borderTop: "1px solid #e0e0e0",
        }}
      >
        <Flex
          direction="column"
          style={{
            border: isDragging ? "2px dashed #111" : "1px solid #e0e0e0",
            borderRadius: "12px",
            backgroundColor: isDragging ? "#f5f5f5" : "#fff",
            padding: "16px",
            transition: "all 0.2s",
            width: "100%",
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {isDragging ? (
            <Text
              size="3"
              style={{
                color: "#111",
                textAlign: "center",
                width: "100%",
                padding: "10px",
              }}
            >
              Drop your document here
            </Text>
          ) : (
            <>
              {/* Text Input - No borders, full width */}
              <input
                ref={inputRef}
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={getPlaceholder()}
                disabled={!dealId}
                style={{
                  width: "100%",
                  fontSize: "16px",
                  border: "none",
                  backgroundColor: "transparent",
                  boxShadow: "none",
                  padding: "0",
                  marginBottom: "12px",
                  outline: "none",
                  color: "#111",
                  cursor: dealId ? "text" : "not-allowed",
                }}
                className="chat-input-no-border"
              />

              {/* Buttons Row - Below text field, inside container */}
              <Flex
                justify="between"
                align="center"
                style={{
                  width: "100%",
                }}
              >
                {/* + Button at leftmost */}
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  variant="outline"
                  size="2"
                  disabled={!dealId}
                  style={{
                    cursor: dealId ? "pointer" : "not-allowed",
                    padding: "0",
                    minWidth: "32px",
                    width: "32px",
                    height: "32px",
                    backgroundColor: "transparent",
                    color: "#111",
                    border: "1px solid #111",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    opacity: dealId ? 1 : 0.5,
                  }}
                  className="add-button"
                >
                  <Text size="4" weight="bold" style={{ lineHeight: "1" }}>
                    +
                  </Text>
                </Button>

                {/* Arrow Button at rightmost */}
                <Button
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || !dealId}
                  variant="outline"
                  size="2"
                  style={{
                    cursor: inputValue.trim() && dealId ? "pointer" : "not-allowed",
                    padding: "0",
                    minWidth: "32px",
                    width: "32px",
                    height: "32px",
                    backgroundColor: inputValue.trim() && dealId ? "#111" : "transparent",
                    color: inputValue.trim() && dealId ? "#fff" : "#ccc",
                    border: "1px solid #111",
                    borderRadius: "50%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                  className="send-button"
                >
                  <Text size="3" weight="bold" style={{ lineHeight: "1" }}>
                    ↑
                  </Text>
                </Button>
              </Flex>
            </>
          )}

          <input
            ref={fileInputRef}
            type="file"
            style={{ display: "none" }}
            onChange={handleFileSelect}
            accept=".pdf,.doc,.docx,.xlsx,.xls,.jpg,.jpeg,.png"
          />
        </Flex>
      </Flex>
    </Flex>
  );
}
