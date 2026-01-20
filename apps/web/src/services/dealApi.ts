import axiosInstance from "./axios.config";
// Type imports reserved for future use
// import type { Deal, DealDocument, ExtractedFact } from "../models/deal.model";
import { createMockDeal, MOCK_FACTS, MOCK_DOCUMENTS, MOCK_DEALS, isMockDeal, isPreexistingMockDeal, saveMockData } from "./mockDealData";

// Deal management uses localStorage since backend deal APIs are temporarily disabled
// Backend deal APIs need Diesel to tokio-postgres conversion
const USE_MOCK_DATA = true;  // Using mock data until backend deal APIs are converted
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
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:70',message:'createDeal called',data:{dealName,USE_MOCK_DATA},timestamp:Date.now(),sessionId:'debug-session',runId:'run5',hypothesisId:'H19'})}).catch(()=>{});
  // #endregion
  
  if (USE_MOCK_DATA) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    const newDeal = createMockDeal(dealName);
    
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:78',message:'Created mock deal',data:{dealId:newDeal.deal_id,dealName:newDeal.deal_name},timestamp:Date.now(),sessionId:'debug-session',runId:'run5',hypothesisId:'H19'})}).catch(()=>{});
    // #endregion
    
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
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:65',message:'getDeals called',data:{USE_MOCK_DATA},timestamp:Date.now(),sessionId:'debug-session',runId:'run5',hypothesisId:'H18'})}).catch(()=>{});
  // #endregion
  if (USE_MOCK_DATA) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    // Return ALL deals (both pre-existing mock deals AND newly created deals)
    const allDeals = MOCK_DEALS;
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:71',message:'Returning all deals',data:{dealsCount:allDeals.length,dealIds:allDeals.map(d=>d.deal_id),mockDeals:allDeals.filter(d=>isMockDeal(d.deal_id)).map(d=>d.deal_id)},timestamp:Date.now(),sessionId:'debug-session',runId:'run5',hypothesisId:'H18'})}).catch(()=>{});
    // #endregion
    return allDeals;
  }
  
  const response = await axiosInstance.get("/api/v1/deals");
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
  // ALL mock deals use the hybrid approach: create mock documents + real OCR tasks
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:118',message:'Mock deal - hybrid mode (mock docs + real tasks)',data:{dealId,fileCount:files.length,isPreexisting:isPreexistingMockDeal(dealId)},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
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
          pipeline: Pipeline.Chunkr,  // FIXED: Use Chunkr pipeline (backend compatible)
          error_handling: ErrorHandling.Fail,
        };
        
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:156',message:'Calling real uploadFile for mock deal',data:{fileName:file.name,dealId,isPreexisting:isPreexistingMockDeal(dealId)},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
        // #endregion
        
        const taskResult = await uploadFile(payload);
        
        // Create mock document with real task ID
        const mockDoc: DocumentResponse = {
          document_id: taskResult.task_id,  // Use real task ID
      deal_id: dealId,
      file_name: file.name,
      document_type: documentType,
      status: "processing",
          // Use pdf_url (presigned URL) if available, otherwise fallback to task_url or mock path
          storage_location: taskResult.output?.pdf_url || taskResult.task_url || `/mock-documents/${file.name}`,
      page_count: 1,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      extracted_at: undefined,
        };
        
        mockDocuments.push(mockDoc);
        MOCK_DOCUMENTS.push(mockDoc);
        
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:180',message:'Real task created for mock deal',data:{taskId:taskResult.task_id,fileName:file.name,dealId},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
        // #endregion
      } catch (error) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:184',message:'Failed to create task for mock deal',data:{fileName:file.name,dealId,error:error instanceof Error?error.message:String(error)},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
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
  
  // For all other deals (non-mock deals from real backend), use REAL backend multipart upload
  // This path is not currently used since backend doesn't have deal management endpoints yet
  // #region agent log
  fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:207',message:'Using REAL backend multipart upload',data:{dealId,fileCount:files.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
  // #endregion
  
  try {
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
    
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:225',message:'REAL backend multipart upload success',data:{dealId,documentsCount:response.data.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
    // #endregion
    
    return response.data;
  } catch (error) {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:231',message:'REAL backend multipart upload FAILED',data:{dealId,error:error instanceof Error?error.message:String(error)},timestamp:Date.now(),sessionId:'debug-session',runId:'run6',hypothesisId:'H20'})}).catch(()=>{});
    // #endregion
    throw error;
  }
};

// Get documents for a deal
export const getDealDocuments = async (
  dealId: string
): Promise<DocumentResponse[]> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const mockDocs = MOCK_DOCUMENTS.filter((d) => d.deal_id === dealId);
    
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:227',message:'getDealDocuments for mock deal',data:{dealId,isPreexisting:isPreexistingMockDeal(dealId),docCount:mockDocs.length,hasDocumentIds:mockDocs.some(d=>!!d.document_id)},timestamp:Date.now(),sessionId:'debug-session',runId:'run4',hypothesisId:'H17'})}).catch(()=>{});
    // #endregion
    
    // For mock documents with real task IDs, fetch the real task status
    // Note: document_id IS the task_id for documents created with real tasks
    const updatedDocs = await Promise.all(
      mockDocs.map(async (doc) => {
        if (doc.document_id && doc.status === "processing") {
          try {
            // #region agent log
            fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:238',message:'Polling real task for mock document',data:{taskId:doc.document_id,fileName:doc.file_name},timestamp:Date.now(),sessionId:'debug-session',runId:'run3',hypothesisId:'H15'})}).catch(()=>{});
            // #endregion
            
            const taskResponse = await axiosInstance.get(`/api/v1/task/${doc.document_id}`);
            const task = taskResponse.data;
            
            // #region agent log NEW: More detailed task info
            fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:270',message:'Task status received - DETAILED',data:{taskId:doc.document_id,fileName:doc.file_name,taskStatus:task.status,hasPdfUrl:!!task.output?.pdf_url,pdfUrl:task.output?.pdf_url?.substring(0,100),outputKeys:task.output?Object.keys(task.output):[],taskKeys:Object.keys(task)},timestamp:Date.now(),sessionId:'debug-session',runId:'view-source-debug',hypothesisId:'H21'})}).catch(()=>{});
            // #endregion
            
            // Update mock document with real task data
            const updatedDoc = {
              ...doc,
              status: task.status.toLowerCase(),
              ocr_output: task.output,
              page_count: task.page_count,
              message: task.message,
              // Update storage_location with presigned URL if available
              storage_location: task.output?.pdf_url || doc.storage_location,
            };
            
            // #region agent log NEW: Log the updated document
            fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'dealApi.ts:285',message:'Updated document',data:{docId:doc.document_id,fileName:doc.file_name,status:updatedDoc.status,hasStorageLocation:!!updatedDoc.storage_location,storageLocationPrefix:updatedDoc.storage_location?.substring(0,100)},timestamp:Date.now(),sessionId:'debug-session',runId:'view-source-debug',hypothesisId:'H22'})}).catch(()=>{});
            // #endregion
            
            // Update in MOCK_DOCUMENTS array for persistence
            const docIndex = MOCK_DOCUMENTS.findIndex(d => d.document_id === doc.document_id);
            if (docIndex !== -1) {
              MOCK_DOCUMENTS[docIndex] = updatedDoc;
              saveMockData();  // Save both deals and documents to localStorage
            }
            
            return updatedDoc;
          } catch (error) {
            console.error(`Error fetching task ${doc.document_id}:`, error);
            return doc;
          }
        }
        return doc;
      })
    );
    
    return updatedDocs;
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

// Delete a deal
export const deleteDeal = async (dealId: string): Promise<void> => {
  if (USE_MOCK_DATA && isMockDeal(dealId)) {
    // Remove from MOCK_DEALS array
    const index = MOCK_DEALS.findIndex((d) => d.deal_id === dealId);
    if (index !== -1) {
      MOCK_DEALS.splice(index, 1);
      // Also remove associated documents and facts
      const { MOCK_DOCUMENTS, MOCK_FACTS, saveMockData } = await import('./mockDealData');
      
      // Remove all documents for this deal
      let docIndex = MOCK_DOCUMENTS.findIndex((d) => d.deal_id === dealId);
      while (docIndex !== -1) {
        MOCK_DOCUMENTS.splice(docIndex, 1);
        docIndex = MOCK_DOCUMENTS.findIndex((d) => d.deal_id === dealId);
      }
      
      // Remove all facts for this deal
      let factIndex = MOCK_FACTS.findIndex((f) => f.deal_id === dealId);
      while (factIndex !== -1) {
        MOCK_FACTS.splice(factIndex, 1);
        factIndex = MOCK_FACTS.findIndex((f) => f.deal_id === dealId);
      }
      
      saveMockData();
    }
    return;
  }

  await axiosInstance.delete(`/api/v1/deals/${dealId}`);
};
