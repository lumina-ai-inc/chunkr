import { ChatMessage, ChatAction, ChatContext } from "./chatApi";
import { DealResponse, FactResponse, DocumentResponse } from "./dealApi";

interface OpenAIMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface OpenAIResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
}

// Build system prompt based on deal context
function buildSystemPrompt(context: ChatContext): string {
  const {
    dealId,
    dealData,
    facts = [],
    documents = [],
  } = context;

  const dealInfo = dealData as DealResponse | undefined;
  const factsList = facts as FactResponse[];
  const docsList = documents as DocumentResponse[];

  return `You are Orin, an AI-assisted underwriting and deal preparation assistant for real estate operators.

Your role:
- Help users analyze rental properties and commercial real estate
- Extract key facts from documents (rent rolls, P&Ls, mortgage statements)
- Calculate underwriting metrics (NOI, DSCR, cash flow)
- Guide users through deal preparation
- Generate investor-ready packages

Current context:
${dealId ? `Deal: ${dealInfo?.deal_name || "Unnamed Deal"}` : "No active deal"}
${dealInfo ? `Status: ${dealInfo.status}` : ""}
${factsList.length > 0 ? `Extracted facts: ${factsList.length}` : "No facts extracted yet"}
${docsList.length > 0 ? `Documents: ${docsList.map((d) => d.file_name).join(", ")}` : "No documents uploaded yet"}

Guidelines:
- Be conversational and helpful
- Ask clarifying questions when needed
- Present analysis results in structured format
- Suggest next steps proactively
- Use expandable action format for multi-step processes
- Format actions as:
  ✓ Completed Action
    Details here
  
  ○ Pending Action
  
  ⟳ Running Action

When discussing metrics:
- DSCR (Debt Service Coverage Ratio): NOI / Annual Debt Service (healthy > 1.25)
- Cap Rate: NOI / Property Value (typical 5-8% for multifamily)
- Cap Spread: Cap Rate - Risk-Free Rate (treasury yield)
- NOI: Gross Rent - Operating Expenses`;
}

// Parse AI response and extract actions
function parseAIResponse(content: string): { content: string; actions: ChatAction[] } {
  const actions: ChatAction[] = [];
  let cleanContent = content;

  // Extract action patterns
  const completedPattern = /✓\s+(.+?)(?:\n|$)/g;
  const pendingPattern = /○\s+(.+?)(?:\n|$)/g;
  const runningPattern = /⟳\s+(.+?)(?:\n|$)/g;

  let match;

  // Extract completed actions
  while ((match = completedPattern.exec(content)) !== null) {
    actions.push({
      id: `action-${Date.now()}-${actions.length}`,
      label: match[1].trim(),
      type: "expandable",
      status: "completed",
    });
  }

  // Extract pending actions
  while ((match = pendingPattern.exec(content)) !== null) {
    actions.push({
      id: `action-${Date.now()}-${actions.length}`,
      label: match[1].trim(),
      type: "expandable",
      status: "pending",
    });
  }

  // Extract running actions
  while ((match = runningPattern.exec(content)) !== null) {
    actions.push({
      id: `action-${Date.now()}-${actions.length}`,
      label: match[1].trim(),
      type: "expandable",
      status: "running",
    });
  }

  return { content: cleanContent, actions };
}

// Generate chat response using OpenAI API
export const generateChatResponse = async (
  userMessage: string,
  context: ChatContext // Chat context for future RAG integration
): Promise<ChatMessage> => {
  const apiKey = import.meta.env.VITE_OPENAI_API_KEY;
  const model = import.meta.env.VITE_OPENAI_MODEL || "gpt-4";

  // If no API key, return mock response
  if (!apiKey) {
    console.warn("OpenAI API key not configured, using mock responses");
    return generateMockResponse(userMessage, context);
  }

  try {
    const systemPrompt = buildSystemPrompt(context);

    const messages: OpenAIMessage[] = [
      { role: "system", content: systemPrompt },
      ...context.previousMessages.map((m) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: userMessage },
    ];

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.7,
        max_tokens: 1000,
      }),
    });

    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.statusText}`);
    }

    const data: OpenAIResponse = await response.json();
    const aiContent = data.choices[0].message.content;

    const { content, actions } = parseAIResponse(aiContent);

    return {
      id: data.id,
      role: "assistant",
      content,
      timestamp: new Date().toISOString(),
      actions,
    };
  } catch (error) {
    console.error("Error calling OpenAI API:", error);
    return generateMockResponse(userMessage, context);
  }
};

// Fallback mock response when OpenAI is not available
function generateMockResponse(
  userMessage: string,
  _context: ChatContext // Chat context for future RAG integration
): ChatMessage {
  const lowerMessage = userMessage.toLowerCase();

  let content = "";
  let actions: ChatAction[] = [];

  if (lowerMessage.includes("upload") || lowerMessage.includes("document")) {
    content = `Great! You can upload documents by dragging them into the chat or clicking the 📎 button. I support:

• Rent rolls (PDF, Excel)
• P&L statements
• Mortgage statements
• Tax documents

Once uploaded, I'll automatically extract key facts and prepare them for underwriting.`;
  } else if (lowerMessage.includes("analyze") || lowerMessage.includes("underwriting")) {
    content = `I'll help you analyze this property. Here's what I'll do:

✓ Extract property data
  Found: Multi-family property with verified documents

○ Calculate DSCR and metrics
○ Identify risk factors
○ Generate recommendations

Would you like me to proceed with the analysis?`;

    actions = [
      {
        id: "action-extract",
        label: "Extract property data",
        type: "expandable",
        status: "completed",
        data: { found: "Multi-family property with verified documents" },
      },
      {
        id: "action-calc",
        label: "Calculate DSCR and metrics",
        type: "expandable",
        status: "pending",
      },
    ];
  } else if (lowerMessage.includes("memo") || lowerMessage.includes("investor")) {
    content = `I can generate an investor-ready memo. It will include:

• Executive summary
• Property overview
• Financial analysis (NOI, DSCR, yields)
• Risk factors and mitigations
• Proposed investment terms

Would you like me to create this package?`;
  } else {
    content = `I'm here to help with your deal. I can:

• Process uploaded documents
• Extract and verify facts
• Calculate underwriting metrics
• Generate investor memos
• Answer questions about your property

What would you like to do next?`;
  }

  return {
    id: Date.now().toString(),
    role: "assistant",
    content,
    timestamp: new Date().toISOString(),
    actions,
  };
}

