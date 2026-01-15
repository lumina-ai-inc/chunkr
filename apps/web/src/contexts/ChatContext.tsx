import { createContext, useContext, useState, ReactNode, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "react-query";
import { ChatMessage, ChatContext as ChatContextType } from "../services/chatApi";
import { generateChatResponse } from "../services/openaiChat";
import {
  getDeal,
  getDealFacts,
  getDealDocuments,
  uploadDealDocuments,
  createDeal,
} from "../services/dealApi";
import toast from "react-hot-toast";

interface ChatProviderContextType {
  // Chat sessions per deal
  chatSessions: Map<string, ChatMessage[]>;
  currentDealId: string | null;

  // Actions
  sendMessage: (dealId: string, message: string) => Promise<void>;
  uploadFile: (dealId: string, file: File) => Promise<void>;
  selectDeal: (dealId: string) => void;
  createNewDeal: () => Promise<string>;

  // Preview state
  previewType: "empty" | "document" | "analysis" | "memo" | "facts" | "underwriting";
  previewData: any;
  updatePreview: (type: string, data: any) => void;

  // Loading states
  isSendingMessage: boolean;
  isUploadingFile: boolean;
}

const ChatContext = createContext<ChatProviderContextType | undefined>(undefined);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [chatSessions, setChatSessions] = useState<Map<string, ChatMessage[]>>(new Map());
  const [currentDealId, setCurrentDealId] = useState<string | null>(null);
  const [previewType, setPreviewType] = useState<"empty" | "document" | "analysis" | "memo" | "facts" | "underwriting">("empty");
  const [previewData, setPreviewData] = useState<any>(null);
  const [isSendingMessage, setIsSendingMessage] = useState(false);
  const [isUploadingFile, setIsUploadingFile] = useState(false);

  const queryClient = useQueryClient();

  // Get deal data for context
  const { data: deal } = useQuery(
    ["deal", currentDealId],
    () => (currentDealId ? getDeal(currentDealId) : null),
    { enabled: !!currentDealId }
  );

  const { data: facts } = useQuery(
    ["facts", currentDealId],
    () => (currentDealId ? getDealFacts(currentDealId) : null),
    { enabled: !!currentDealId }
  );

  const { data: documents } = useQuery(
    ["documents", currentDealId],
    () => (currentDealId ? getDealDocuments(currentDealId) : null),
    { enabled: !!currentDealId }
  );

  // Create deal mutation
  const createDealMutation = useMutation(
    (dealData: { deal_name: string }) => createDeal(dealData.deal_name),
    {
      onSuccess: (__data) => {
        queryClient.invalidateQueries("deals");
        toast.success("New deal created!");
      },
      onError: (error) => {
        console.error("Error creating deal:", error);
        toast.error("Failed to create deal");
      },
    }
  );

  // Upload file mutation
  const uploadFileMutation = useMutation(
    ({ dealId, file }: { dealId: string; file: File }) => {
      // #region agent log
      fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ChatContext.tsx:83',message:'uploadFileMutation starting',data:{dealId,fileName:file.name,fileSize:file.size},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H1-H2'})}).catch(()=>{});
      // #endregion
      return uploadDealDocuments(dealId, [file], 'rental_document') as any;
    },
    {
      onSuccess: (_, variables) => {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ChatContext.tsx:87',message:'uploadFileMutation success',data:{dealId:variables.dealId},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H2'})}).catch(()=>{});
        // #endregion
        queryClient.invalidateQueries(["documents", variables.dealId]);
        queryClient.invalidateQueries(["deal", variables.dealId]);
        toast.success("Document uploaded successfully!");
      },
      onError: (error) => {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'ChatContext.tsx:92',message:'uploadFileMutation error',data:{error:error instanceof Error?error.message:String(error)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H2-H3'})}).catch(()=>{});
        // #endregion
        console.error("Error uploading file:", error);
        toast.error("Failed to upload document");
      },
    }
  );

  const sendMessage = useCallback(
    async (dealId: string, message: string) => {
      if (!message.trim()) return;

      setIsSendingMessage(true);

      try {
        const userMessage: ChatMessage = {
          id: `user-${Date.now()}`,
          role: "user",
          content: message,
          timestamp: new Date().toISOString(),
        };

        // Add user message to chat
        setChatSessions((prev) => {
          const updated = new Map(prev);
          const messages = updated.get(dealId) || [];
          updated.set(dealId, [...messages, userMessage]);
          return updated;
        });

        // Build context
        const context: ChatContextType = {
          dealId,
          dealData: deal,
          facts: facts || undefined,
          documents: documents || undefined,
          previousMessages: chatSessions.get(dealId) || [],
        };

        // Get AI response
        const response = await generateChatResponse(message, context);

        // Add assistant response to chat
        setChatSessions((prev) => {
          const updated = new Map(prev);
          const messages = updated.get(dealId) || [];
          updated.set(dealId, [...messages.slice(0, -1), userMessage, response]);
          return updated;
        });

        // Update preview based on response
        if (response.actions?.some((a) => a.status === "completed")) {
          updatePreview("analysis", { dealId });
        }
      } catch (error) {
        console.error("Error sending message:", error);
        toast.error("Failed to send message");
      } finally {
        setIsSendingMessage(false);
      }
    },
    [deal, facts, documents, chatSessions]
  );

  const uploadFile = useCallback(
    async (dealId: string, file: File) => {
      setIsUploadingFile(true);

      try {
        await uploadFileMutation.mutateAsync({ dealId, file });

        // Add system message to chat
        const systemMessage: ChatMessage = {
          id: `system-${Date.now()}`,
          role: "assistant",
          content: `Document "${file.name}" uploaded successfully! I'm processing it now and will extract the key facts.`,
          timestamp: new Date().toISOString(),
        };

        setChatSessions((prev) => {
          const updated = new Map(prev);
          const messages = updated.get(dealId) || [];
          updated.set(dealId, [...messages, systemMessage]);
          return updated;
        });

        updatePreview("document", { dealId, fileName: file.name });
      } catch (error) {
        console.error("Error uploading file:", error);
      } finally {
        setIsUploadingFile(false);
      }
    },
    [uploadFileMutation]
  );

  const selectDeal = useCallback((dealId: string) => {
    setCurrentDealId(dealId);
    setPreviewType("analysis");
  }, []);

  const createNewDeal = useCallback(async () => {
    const dealName = `New Deal ${new Date().toLocaleDateString()}`;
    const result = await createDealMutation.mutateAsync({ deal_name: dealName });
    setCurrentDealId(result.deal_id);
    setChatSessions((prev) => {
      const updated = new Map(prev);
      updated.set(result.deal_id, [
        {
          id: "welcome",
          role: "assistant",
          content: `Welcome to your new deal! I'm here to help you analyze properties and prepare investor packages. You can:

• Upload documents (rent rolls, P&Ls, mortgage statements)
• Ask questions about property analysis
• Get underwriting metrics calculated
• Generate investor memos

What would you like to do first?`,
          timestamp: new Date().toISOString(),
        },
      ]);
      return updated;
    });
    setPreviewType("empty");
    return result.deal_id;
  }, [createDealMutation]);

  const updatePreview = useCallback((type: string, data: any) => {
    setPreviewType(type as any);
    setPreviewData(data);
  }, []);

  const value: ChatProviderContextType = {
    chatSessions,
    currentDealId,
    sendMessage,
    uploadFile,
    selectDeal,
    createNewDeal,
    previewType,
    previewData,
    updatePreview,
    isSendingMessage,
    isUploadingFile,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChatContext() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
}

