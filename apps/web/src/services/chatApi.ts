import axiosInstance from "./axios.config";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: string;
  actions?: ChatAction[];
  attachments?: FileAttachment[];
}

export interface ChatAction {
  id: string;
  label: string;
  type: "expandable" | "button";
  status: "pending" | "running" | "completed" | "failed";
  data?: any;
}

export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  url?: string;
}

export interface ChatContext {
  dealId?: string;
  dealData?: any;
  facts?: any[];
  documents?: any[];
  previousMessages: ChatMessage[];
}

// Public mode: Preview responses only (no login required)
export const sendPublicMessage = async (
  message: string,
  _file?: File // Reserved for future file upload feature
): Promise<ChatMessage> => {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 800));

  let content = "";
  const lowerMessage = message.toLowerCase();

  // Generate contextual responses
  if (lowerMessage.includes("analyze") || lowerMessage.includes("property")) {
    content =
      "Great! I can help you analyze a property. To get started, I'll need:\n\n• Rent roll (PDF or spreadsheet)\n• Trailing 12-month P&L\n• Mortgage statement (optional)\n\nYou can upload documents by dragging them into the chat. Want to proceed?";
  } else if (lowerMessage.includes("dscr") || lowerMessage.includes("cap")) {
    content =
      "I can calculate DSCR and cap spread for your property. I'll need:\n\n• Annual gross rent\n• Operating expenses\n• Debt service\n• Property value\n\nUpload documents or share these details to continue.";
  } else if (lowerMessage.includes("memo") || lowerMessage.includes("investor")) {
    content =
      "I'll help you create an investor-ready memo with:\n\n✓ Executive summary\n✓ Property overview\n✓ Financial analysis\n✓ Risk factors\n✓ Terms\n\nThis feature requires an account. Would you like to sign up?";
  } else if (lowerMessage.includes("capital") || lowerMessage.includes("fund")) {
    content =
      "To estimate your capital access, I'll need property details, equity position, and cash flow info. Upload documents or describe your properties.";
  } else {
    content =
      "I can help with property analysis, underwriting, and investor packages. What would you like to explore?";
  }

  return {
    id: Date.now().toString(),
    role: "assistant",
    content,
    timestamp: new Date().toISOString(),
  };
};

// Handle file upload in public mode (preview only)
export const handlePublicFileUpload = async (
  _file: File // Reserved for future file upload feature
): Promise<ChatMessage> => {
  // Simulate processing
  await new Promise((resolve) => setTimeout(resolve, 600));

  return {
    id: Date.now().toString(),
    role: "assistant",
    content:
      "Great! I can see you've uploaded a document. To fully process this file and extract key facts, you'll need to create a free account. This allows me to:\n\n✓ Extract property data using OCR\n✓ Calculate underwriting metrics\n✓ Generate investor-ready packages\n✓ Save your analysis\n\nReady to continue?",
    timestamp: new Date().toISOString(),
  };
};

// Private mode: Full functionality with backend integration
export const sendPrivateMessage = async (
  dealId: string,
  message: string,
  context: ChatContext
): Promise<ChatMessage> => {
  try {
    const response = await axiosInstance.post(
      `/api/v1/deals/${dealId}/chat`,
      {
        message,
        context: {
          previousMessages: context.previousMessages,
          facts: context.facts,
          documents: context.documents,
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error("Error sending private message:", error);
    
    // Fallback to mock response if backend not ready
    return {
      id: Date.now().toString(),
      role: "assistant",
      content:
        "I received your message. The backend chat service is currently being set up. For now, I can help you navigate the dashboard and access your deal data.",
      timestamp: new Date().toISOString(),
    };
  }
};

// Get chat history for a deal
export const getChatHistory = async (
  dealId: string
): Promise<ChatMessage[]> => {
  try {
    const response = await axiosInstance.get(
      `/api/v1/deals/${dealId}/chat/history`
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching chat history:", error);
    return [];
  }
};

// Save chat message to backend (for analytics)
export const saveChatMessage = async (
  dealId: string,
  message: ChatMessage
): Promise<void> => {
  try {
    await axiosInstance.post(`/api/v1/deals/${dealId}/chat/save`, message);
  } catch (error) {
    console.error("Error saving chat message:", error);
  }
};

// Check if user should see login prompt
export const shouldShowLoginPrompt = (
  messageCount: number,
  hasUploadedFile: boolean,
  isAuthenticated: boolean
): boolean => {
  if (isAuthenticated) return false;
  
  // Show after 2 messages or after file upload
  return messageCount >= 2 || hasUploadedFile;
};

