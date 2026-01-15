import { useState, useRef, useEffect } from "react";
import { Flex, Text, Button } from "@radix-ui/themes";
import "./ChatInterface.css";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSend: (message: string) => void;
  onFileUpload?: (file: File) => void;
  placeholder?: string;
  isLoading?: boolean;
}

export default function ChatInterface({
  messages,
  onSend,
  onFileUpload,
  placeholder = "Give Orin a task to work on...",
  isLoading: _isLoading = false,
}: ChatInterfaceProps) {
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
    if (!message) return;

    onSend(message);
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

  return (
    <Flex
      direction="column"
      style={{
        width: "100%",
        maxWidth: "1000px",
        margin: "0 auto",
      }}
    >
      {/* Messages */}
      {messages.length > 0 && (
        <div
          ref={messagesContainerRef}
          className="messages-container"
          style={{
            marginBottom: "24px",
            maxHeight: "400px",
            overflowY: "auto",
            overflowX: "visible",
            padding: "0 24px",
            width: "100%",
            scrollBehavior: "smooth",
          }}
        >
          <Flex
            direction="column"
            gap="16px"
            style={{
              paddingBottom: "8px",
            }}
          >
            {messages.map((message) => (
              <Flex
                key={message.id}
                direction="column"
                style={{
                  alignSelf: message.role === "user" ? "flex-end" : "flex-start",
                  maxWidth: "80%",
                  width: "fit-content",
                  position: "relative",
                }}
              >
                <div
                  style={{
                    padding: "12px 16px",
                    backgroundColor:
                      message.role === "user" ? "#111" : "#f5f5f5",
                    color: message.role === "user" ? "#fff" : "#111",
                    borderRadius: "12px",
                    border: message.role === "user" ? "1px solid #111" : "none",
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
                      color: message.role === "user" ? "#fff" : "#111",
                    }}
                  >
                    {message.content}
                  </Text>
                </div>
              </Flex>
            ))}
            <div ref={messagesEndRef} />
          </Flex>
        </div>
      )}

      {/* Input Area - Outer container with border, buttons below text field */}
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
              placeholder={placeholder}
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
                style={{
                  cursor: "pointer",
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
                disabled={!inputValue.trim()}
                variant="outline"
                size="2"
                style={{
                  cursor: inputValue.trim() ? "pointer" : "not-allowed",
                  padding: "0",
                  minWidth: "32px",
                  width: "32px",
                  height: "32px",
                  backgroundColor: inputValue.trim() ? "#111" : "transparent",
                  color: inputValue.trim() ? "#fff" : "#ccc",
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
  );
}
