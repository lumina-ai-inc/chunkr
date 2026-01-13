import axiosInstance from "./axios.config";
// Type imports reserved for future use
// import type { Deal, DealDocument, ExtractedFact } from "../models/deal.model";
import { createMockDeal, MOCK_FACTS, MOCK_DOCUMENTS, MOCK_DEALS, isMockDeal, saveMockData } from "./mockDealData";

// TODO: Set to false once backend is fully operational
const USE_MOCK_DATA = true;  // REVERTED: Back to mock mode - backend deal APIs not available
// #region agent log
fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:7',message:'dealApi module loaded',data:{USE_MOCK_DATA},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H6'})}).catch(()=>{});
// #endregion

export interface CreateDealRequest {
  deal_name: string;
}

export interface DealResponse {
  deal_id: string;
  user_id: string;
  deal_name: string;
  status: string;
  created_at: string;
  updated_at: string;
  metadata: any;
  document_count?: number;
  fact_count?: number;
}

export interface DocumentResponse {
  document_id: string;
  deal_id: string;
  file_name: string;
  document_type: string;
  status: string; // "pending" | "processing" | "completed" | "failed"
  storage_location?: string;
  page_count?: number;
  ocr_output?: any;
  created_at: string;
  updated_at: string;
  extracted_at?: string;
}

export interface FactResponse {
  fact_id: string;
  document_id: string;
  deal_id: string;
  fact_type: string;
  label: string;
  value: string;
  unit?: string;
  source_citation: {
    document: string;
    page: number;
    line?: string;
    bbox?: {
      left: number;
      top: number;
      width: number;
      height: number;
    };
  };
  status: string;
  confidence_score?: number;
  approved_at?: string;
  approved_by?: string;
  locked: boolean;
  created_at: string;
}

// Create a new deal
export const createDeal = async (dealName: string): Promise<DealResponse> => {
  if (USE_MOCK_DATA) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const newDeal = createMockDeal(dealName);
    console.log("Created mock deal:", newDeal);
    return newDeal;
  }
  
  const response = await axiosInstance.post("/api/v1/deals", {
    deal_name: dealName,
  });
  return response.data;
};

// Get all deals for the current user
export const getDeals = async (): Promise<DealResponse[]> => {
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:65',message:'getDeals called',data:{USE_MOCK_DATA},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H6-H7-H8'})}).catch(()=>{});
  // #endregion
  if (USE_MOCK_DATA) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:71',message:'Returning mock deals',data:{dealsCount:MOCK_DEALS.length,dealIds:MOCK_DEALS.map(d=>d.deal_id)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H6'})}).catch(()=>{});
    // #endregion
    return MOCK_DEALS;
  }
  
  const response = await axiosInstance.get("/api/v1/deals");
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:78',message:'Returning real deals from API',data:{dealsCount:response.data.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H7-H8'})}).catch(()=>{});
  // #endregion
  return response.data;
};

// Get a specific deal by ID
export const getDeal = async (dealId: string): Promise<DealResponse> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 200));
    const deal = MOCK_DEALS.find((d) => d.deal_id === dealId);
    if (!deal) throw new Error("Deal not found");
    return deal;
  }
  
  const response = await axiosInstance.get(`/api/v1/deals/${dealId}`);
  return response.data;
};

// Upload documents to a deal
export const uploadDealDocuments = async (
  dealId: string,
  files: File[],
  documentType: string
): Promise<DocumentResponse[]> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:118',message:'Mock mode - creating real processing tasks',data:{dealId,fileCount:files.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run2',hypothesisId:'H11'})}).catch(()=>{});
    // #endregion
    
    // Import uploadFile to create real processing tasks
    const { uploadFile } = await import('./uploadFileApi');
    const { OcrStrategy, SegmentationStrategy, Pipeline, ErrorHandling } = await import('../models/taskConfig.model');
    
    const mockDocuments: DocumentResponse[] = [];
    
    // For each file, create a REAL processing task
    for (let index = 0; index < files.length; index++) {
      const file = files[index];
      try {
        // Encode file to base64
        const reader = new FileReader();
        const b64 = await new Promise<string>((resolve, reject) => {
          reader.onload = () => resolve(reader.result as string);
          reader.onerror = reject;
          reader.readAsDataURL(file);
        });
        
        // Create real OCR task
        const payload: any = {
          file: b64,
          file_name: file.name,
          ocr_strategy: OcrStrategy.All,
          segmentation_strategy: SegmentationStrategy.Page,
          high_resolution: true,
          pipeline: Pipeline.Orin,
          error_handling: ErrorHandling.Fail,
        };
        
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:145',message:'Calling real uploadFile for mock document',data:{fileName:file.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run2',hypothesisId:'H11'})}).catch(()=>{});
        // #endregion
        
        const taskResult = await uploadFile(payload);
        
        // Create mock document with real task ID
        const mockDoc: DocumentResponse = {
          document_id: taskResult.task_id,  // Use real task ID
          deal_id: dealId,
          file_name: file.name,
          document_type: documentType,
          status: "processing",
          storage_location: taskResult.task_url || `/mock-documents/${file.name}`,
          page_count: 1,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          extracted_at: undefined,
        };
        
        mockDocuments.push(mockDoc);
        MOCK_DOCUMENTS.push(mockDoc);
        
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:166',message:'Real task created for mock document',data:{taskId:taskResult.task_id,fileName:file.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run2',hypothesisId:'H11'})}).catch(()=>{});
        // #endregion
      } catch (error) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:170',message:'Failed to create real task for mock document',data:{fileName:file.name,error:error instanceof Error?error.message:String(error)},timestamp:Date.now(),sessionId:'debug-session',runId:'run2',hypothesisId:'H11'})}).catch(()=>{});
        // #endregion
        console.error(`Failed to process ${file.name}:`, error);
      }
    }
    
    const deal = MOCK_DEALS.find(d => d.deal_id === dealId);
    if (deal) {
      deal.document_count = (deal.document_count || 0) + mockDocuments.length;
      deal.status = "processing_documents";
      deal.updated_at = new Date().toISOString();
    }
    
    saveMockData();  // Persist to localStorage
    console.log("Created mock documents with real tasks:", mockDocuments);
    return mockDocuments;
  }
  
  const formData = new FormData();
  files.forEach((file) => formData.append("files", file));
  formData.append("document_type", documentType);

  const response = await axiosInstance.post(
    `/api/v1/deals/${dealId}/documents`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );
  
  return response.data;
};

// Get documents for a deal
export const getDealDocuments = async (
  dealId: string
): Promise<DocumentResponse[]> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    return MOCK_DOCUMENTS.filter((d) => d.deal_id === dealId);
  }
  
  const response = await axiosInstance.get(`/api/v1/deals/${dealId}/documents`);
  return response.data;
};

// Poll document status for real-time updates
export const pollDocumentStatus = async (
  dealId: string,
  documentId: string,
  onStatusUpdate: (status: string, ocrOutput?: any) => void,
  interval = 2000, // Poll every 2 seconds
  timeout = 120000 // Timeout after 2 minutes
): Promise<DocumentResponse> => {
  const startTime = Date.now();
  return new Promise((resolve, reject) => {
    const intervalId = setInterval(async () => {
      if (Date.now() - startTime > timeout) {
        clearInterval(intervalId);
        reject(new Error("Document processing timed out"));
        return;
      }

      try {
        const documents = await getDealDocuments(dealId);
        const document = documents.find((doc) => doc.document_id === documentId);

        if (document) {
          onStatusUpdate(document.status, document.ocr_output);
          if (document.status === "completed" || document.status === "failed") {
            clearInterval(intervalId);
            resolve(document);
          }
        }
      } catch (error) {
        console.error("Polling error:", error);
        // Continue polling even if there's a temporary error
      }
    }, interval);
  });
};

// Get facts for a deal
export const getDealFacts = async (dealId: string): Promise<FactResponse[]> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 200));
    return MOCK_FACTS.filter((f) => f.deal_id === dealId);
  }
  
  const response = await axiosInstance.get(`/api/v1/deals/${dealId}/facts`);
  return response.data;
};

// Approve a single fact
export const approveFact = async (
  dealId: string,
  factId: string
): Promise<FactResponse> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const fact = MOCK_FACTS.find((f) => f.fact_id === factId);
    if (!fact) throw new Error("Fact not found");
    fact.status = "approved";
    fact.locked = true;
    return fact;
  }
  
  const response = await axiosInstance.patch(
    `/api/v1/deals/${dealId}/facts/${factId}/approve`
  );
  return response.data;
};

// Approve multiple facts
export const approveFacts = async (
  dealId: string,
  factIds: string[]
): Promise<FactResponse[]> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const approvedFacts: FactResponse[] = [];
    factIds.forEach(factId => {
      const fact = MOCK_FACTS.find((f) => f.fact_id === factId);
      if (fact) {
        fact.status = "approved";
        fact.locked = true;
        approvedFacts.push(fact);
      }
    });
    return approvedFacts;
  }
  
  const response = await axiosInstance.post(
    `/api/v1/deals/${dealId}/facts/approve-batch`,
    { fact_ids: factIds }
  );
  return response.data;
};

// Reset facts
export const resetFacts = async (
  dealId: string,
  factIds?: string[]
): Promise<void> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 200));
    MOCK_FACTS.forEach(fact => {
      if (fact.deal_id === dealId) {
        fact.status = "pending_approval";
        fact.locked = false;
      }
    });
    return;
  }
  
  await axiosInstance.post(
    `/api/v1/deals/${dealId}/facts/reset`,
    { fact_ids: factIds || [] }
  );
};

// Update a fact
export const updateFact = async (
  dealId: string,
  factId: string,
  updates: Partial<FactResponse>
): Promise<FactResponse> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 200));
    const fact = MOCK_FACTS.find((f) => f.fact_id === factId);
    if (!fact) throw new Error("Fact not found");
    Object.assign(fact, updates);
    return fact;
  }
  
  const response = await axiosInstance.patch(
    `/api/v1/deals/${dealId}/facts/${factId}`,
    updates
  );
  return response.data;
};

// Update deal status
export const updateDealStatus = async (
  dealId: string,
  status: string
): Promise<void> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    const deal = MOCK_DEALS.find((d) => d.deal_id === dealId);
    if (deal) {
      deal.status = status;
      deal.updated_at = new Date().toISOString();
    }
    return;
  }

  await axiosInstance.patch(`/api/v1/deals/${dealId}`, { status });
};
