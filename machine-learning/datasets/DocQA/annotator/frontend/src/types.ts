export interface Annotation {
    annotation_id: string;
    doc_id: string; // filename without extension
    question: string;
    answers: string[];
    page_ids: string[]; // e.g., ["doc_p0", "doc_p1"]
    answer_page_idx?: number[]; // Optional: indices within page_ids
}

export interface QASuggestion {
    question: string;
    answer: string;
}

// Add other types as needed, e.g., for PDF info