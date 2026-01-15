import axiosInstance from "./axios.config";

export interface Conversation {
  conversation_id: string;
  user_id: string;
  deal_id?: string;
  title?: string;
  context?: any;
  created_at: string;
  updated_at: string;
}

export interface Message {
  message_id: string;
  conversation_id: string;
  role: string;
  content: string;
  metadata?: any;
  embedding_id?: string;
  created_at: string;
}

export interface SendMessageResponse {
  response: string;
  intent: string;
  action?: {
    action_type: string;
    parameters: any;
  };
  context_used: number;
}

export interface PublicMessageResponse {
  response: string;
  session_id: string;
  intent: string;
}

// Create a new conversation
export const createConversation = async (dealId?: string, title?: string): Promise<Conversation> => {
  const response = await axiosInstance.post("/api/v1/conversations", {
    deal_id: dealId,
    title,
  });
  return response.data;
};

// List all conversations for the user
export const listConversations = async (): Promise<Conversation[]> => {
  const response = await axiosInstance.get("/api/v1/conversations");
  return response.data;
};

// Get a specific conversation with messages
export const getConversation = async (conversationId: string): Promise<{ conversation: Conversation; messages: Message[] }> => {
  const response = await axiosInstance.get(`/api/v1/conversations/${conversationId}`);
  return response.data;
};

// Send a message in a conversation
export const sendMessage = async (conversationId: string, content: string): Promise<SendMessageResponse> => {
  const response = await axiosInstance.post(`/api/v1/conversations/${conversationId}/messages`, {
    content,
  });
  return response.data;
};

// Delete a conversation
export const deleteConversation = async (conversationId: string): Promise<void> => {
  await axiosInstance.delete(`/api/v1/conversations/${conversationId}`);
};

// Send a public message (no authentication required)
export const sendPublicMessage = async (content: string, sessionId?: string): Promise<PublicMessageResponse> => {
  const response = await axiosInstance.post("/api/v1/conversations/public/message", {
    content,
    session_id: sessionId,
  });
  return response.data;
};

