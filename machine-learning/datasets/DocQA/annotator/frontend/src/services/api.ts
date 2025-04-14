import axios, { AxiosRequestConfig } from 'axios';
import { Annotation, QASuggestion } from '../types';

// Use the proxy path if configured, otherwise use the full backend URL
const API_PREFIX = '/api'; // Or 'http://localhost:8001' if not using proxy

// --- Edit 1: Type for optional headers ---
type ApiHeaders = Record<string, string>;

const apiClient = axios.create({
    baseURL: API_PREFIX,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add an interceptor to handle errors globally, especially 401/403
apiClient.interceptors.response.use(
    response => response,
    error => {
        if (error.response && (error.response.status === 401 || error.response.status === 403)) {
            console.error("Authentication error detected by Axios interceptor:", error.response.data);
            // Optionally trigger a logout event or redirect here if needed globally
            // Example: window.dispatchEvent(new Event('auth-error'));
            // For now, App.tsx handles logout in its catch blocks.
        }
        // Re-throw the error so specific catch blocks in App.tsx can handle it
        // Include response data if available for better debugging in App.tsx
        const errorData = error.response?.data || { detail: error.message };
        const enhancedError = new Error(errorData.detail || error.message);
        // Attach status for easier checking in App.tsx
        (enhancedError as any).status = error.response?.status;
        return Promise.reject(enhancedError);
    }
);

export const getDatasets = async (headers?: ApiHeaders): Promise<string[]> => {
    // Merge provided headers with defaults
    const config: AxiosRequestConfig = { headers };
    const response = await apiClient.get<string[]>('/datasets', config);
    return response.data;
};

export const createDataset = async (datasetId: string, headers?: ApiHeaders): Promise<any> => {
    const config: AxiosRequestConfig = { headers };
    const response = await apiClient.post(`/datasets/${datasetId}`, {}, config); // Send empty body if needed
    return response.data;
};

export const getPdfs = async (datasetId: string, onlyUnannotated: boolean = false, headers?: ApiHeaders): Promise<string[]> => {
    const config: AxiosRequestConfig = {
        params: { only_unannotated: onlyUnannotated },
        headers, // Pass headers here
    };
    const response = await apiClient.get<string[]>(`/datasets/${datasetId}/pdfs`, config);
    return response.data;
};

// Renamed and modified to fetch the PDF blob using authenticated client
export const getPdfBlob = async (datasetId: string, pdfFilename: string, headers?: ApiHeaders): Promise<Blob> => {
    // Fetch the blob using apiClient to ensure auth headers are included
    const config: AxiosRequestConfig = { headers, responseType: 'blob' };
    try {
        const response = await apiClient.get(`/datasets/${datasetId}/pdfs/${pdfFilename}`, config);
        return response.data;
    } catch (error) {
        console.error(`Failed to fetch PDF blob for ${pdfFilename}:`, error);
        // Re-throw the error so the caller can handle it (e.g., show an error message)
        // The interceptor should have already enhanced the error object
        throw error;
    }
};

export const uploadFile = async (datasetId: string, file: File, headers?: ApiHeaders): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    // Merge auth headers with multipart header
    const config: AxiosRequestConfig = {
        headers: {
            ...headers, // Include auth header
            'Content-Type': 'multipart/form-data', // Keep multipart header
        },
    };
    const response = await apiClient.post(`/datasets/${datasetId}/pdfs`, formData, config);
    return response.data;
};


export const getAnnotations = async (datasetId: string, docId?: string, headers?: ApiHeaders): Promise<Annotation[]> => {
    const config: AxiosRequestConfig = {
        params: docId ? { doc_id: docId } : {},
        headers,
    };
    const response = await apiClient.get<Annotation[]>(`/datasets/${datasetId}/annotations`, config);
    return response.data;
};

export const saveAnnotation = async (datasetId: string, annotationData: Omit<Annotation, 'annotation_id'>, headers?: ApiHeaders): Promise<Annotation> => {
    // No need for try/catch here if using interceptor, unless specific logic is needed
    console.log('Saving annotation:', { datasetId, annotationData });
    const config: AxiosRequestConfig = {
        headers: {
            ...headers, // Include auth header
            'Content-Type': 'application/json', // Ensure correct content type
        }
    };
    const response = await apiClient.post<Annotation>(
        `/datasets/${datasetId}/annotations`,
        annotationData,
        config
    );
    console.log('Annotation saved successfully:', response.data);
    return response.data;
};

export const generateQA = async (
    datasetId: string,
    pdfFilename: string,
    pages: number[],
    numQuestions: number = 3,
    headers?: ApiHeaders // Add headers parameter
): Promise<QASuggestion[]> => {
    const config: AxiosRequestConfig = { headers };
    const response = await apiClient.post<{ suggestions: QASuggestion[] }>(
        `/datasets/${datasetId}/pdfs/${pdfFilename}/generate-qa`,
        {
            pages,
            num_questions: numQuestions
        },
        config // Pass config with headers
    );
    return response.data.suggestions || [];
};

export const updateAnnotation = async (
    datasetId: string,
    annotation: Annotation,
    headers?: ApiHeaders // Add headers parameter
): Promise<Annotation> => {
    const config: AxiosRequestConfig = { headers };
    // Assuming PUT endpoint exists: /datasets/{dataset_id}/annotations/{annotation_id}
    const response = await apiClient.put(
        `/datasets/${datasetId}/annotations/${annotation.annotation_id || ''}`,
        annotation,
        config // Pass config with headers
    );
    return response.data;
};

export const deleteAnnotation = async (datasetId: string, annotationId: string, headers?: ApiHeaders): Promise<void> => {
    const config: AxiosRequestConfig = { headers };
    await apiClient.delete(`/datasets/${datasetId}/annotations/${annotationId}`, config);
};

export const deletePdf = async (datasetId: string, pdfFilename: string, headers?: ApiHeaders): Promise<void> => {
    const config: AxiosRequestConfig = { headers };
    await apiClient.delete(`/datasets/${datasetId}/pdfs/${pdfFilename}`, config);
};

// --- End Edit 2 ---

// Add a dummy saveQA function if App.tsx calls it (remove if not needed)
export const saveQA = async (
    datasetId: string,
    pdfFilename: string,
    pages: number[],
    question: string,
    answer: string,
    headers?: ApiHeaders // Add headers if this needs protection
): Promise<void> => {
    console.warn("saveQA function called but not fully implemented in api.ts");
    // Implement actual backend call if required
    // const config: AxiosRequestConfig = { headers };
    // await apiClient.post(`/path/to/save/qa`, { data }, config);
    return Promise.resolve();
}

// --- No changes needed for fetchAnnotations if it's not exported/used ---
async function fetchAnnotations() { // Keep as internal helper if needed, or remove
    console.warn("Internal fetchAnnotations helper called - ensure it's necessary or remove.");
}
