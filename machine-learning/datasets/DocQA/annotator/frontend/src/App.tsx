import { useState, useEffect, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';
import { FiUpload, FiSave, FiEdit, FiCheck, FiLogOut, FiSun, FiMoon } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import DOMPurify from 'dompurify';

import * as api from './services/api';
import { Annotation, QASuggestion } from './types';
// import './App.css'; // Already imported in main.tsx

// PDF.js worker configuration
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

// Add a type for storing per-PDF state
interface PdfState {
  activeTab: 'manual' | 'llm' | 'existing';
  selectedPagesForLLM: number[];
  pageInputText: string;
  numQuestions: number;
}

// Add dataset stats type
interface DatasetStats {
  totalDocuments: number;
  totalQuestions: number;
  averageQuestionsPerDoc: number;
  loading: boolean;
}

// Add a new component for rendering formatted text
const FormattedText = ({ text }: { text: string }) => {
  // Function to safely render HTML
  const createMarkup = (html: string) => {
    return { __html: DOMPurify.sanitize(html) };
  };

  return (
    <ReactMarkdown
      remarkPlugins={[remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        // Allow HTML to be rendered safely
        p: ({ children }) => {
          if (typeof children[0] === 'string' && children[0].includes('<')) {
            return <div dangerouslySetInnerHTML={createMarkup(children[0] as string)} />;
          }
          return <p>{children}</p>;
        },
      }}
    >
      {text}
    </ReactMarkdown>
  );
};

function App() {
  // --- State ---
  // Add theme state
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    // Check localStorage for saved theme or default to 'light'
    const savedTheme = localStorage.getItem('theme');
    return (savedTheme === 'dark' ? 'dark' : 'light');
  });
  const [apiKey, setApiKey] = useState<string>('');
  const [isValidApiKey, setIsValidApiKey] = useState<boolean>(false);
  const [authError, setAuthError] = useState<string | null>(null);

  const [datasets, setDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [pdfs, setPdfs] = useState<string[]>([]);
  const [selectedPdf, setSelectedPdf] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1); // PDF.js pages are 1-based

  const [annotations, setAnnotations] = useState<Annotation[]>([]);
  const [qaSuggestions, setQaSuggestions] = useState<QASuggestion[]>([]);
  const [selectedPagesForLLM, setSelectedPagesForLLM] = useState<number[]>([]); // 0-based for API

  const [isLoadingDatasets, setIsLoadingDatasets] = useState<boolean>(false);
  const [isLoadingPdfs, setIsLoadingPdfs] = useState<boolean>(false);
  const [isLoadingAnnotations, setIsLoadingAnnotations] = useState<boolean>(false);
  const [isLoadingPdfDocument, setIsLoadingPdfDocument] = useState<boolean>(false);
  const [isGeneratingQA, setIsGeneratingQA] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);

  const [error, setError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [annotationError, setAnnotationError] = useState<string | null>(null);

  const [newDatasetName, setNewDatasetName] = useState<string>('');
  const [showOnlyUnannotated, setShowOnlyUnannotated] = useState<boolean>(false);

  const [scale, setScale] = useState<number>(1.0); // Add scale state

  // Add this new state for the input text
  const [pageInputText, setPageInputText] = useState<string>('');

  const [llmError, setLlmError] = useState<string | null>(null);

  const [activeTab, setActiveTab] = useState<'manual' | 'llm' | 'existing'>('manual');

  // Add state for editable suggestions
  const [editingSuggestion, setEditingSuggestion] = useState<number | null>(null);
  const [editedQuestion, setEditedQuestion] = useState<string>('');
  const [editedAnswer, setEditedAnswer] = useState<string>('');

  // Add state for tracking saved suggestions
  const [savedSuggestions, setSavedSuggestions] = useState<Set<number>>(new Set());

  // Add a state for the number of questions
  const [numQuestions, setNumQuestions] = useState<number>(3);

  // Add these new state variables at the top with other state declarations
  const [searchText, setSearchText] = useState<string>('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [currentSearchIndex, setCurrentSearchIndex] = useState<number>(0);
  const [showSearch, setShowSearch] = useState<boolean>(false);

  // Add this state to store per-PDF settings
  const [pdfStates, setPdfStates] = useState<Record<string, PdfState>>({});

  // Add dataset stats state
  const [datasetStats, setDatasetStats] = useState<DatasetStats>({
    totalDocuments: 0,
    totalQuestions: 0,
    averageQuestionsPerDoc: 0,
    loading: false
  });

  // First, let's fix the page index issue by adding a helper function
  const displayPageNumber = (pageIndex: number): number => {
    // Add 1 to convert from 0-based index to 1-based page number for display
    return pageIndex + 1;
  };

  // Then let's add state for editing page numbers
  const [editingPages, setEditingPages] = useState<boolean>(false);
  const [editedPages, setEditedPages] = useState<{ [key: number]: number[] }>({});

  // Add state for editing pages
  const [editingPagesForSuggestion, setEditingPagesForSuggestion] = useState<number | null>(null);
  const [tempEditedPages, setTempEditedPages] = useState<number[]>([]);

  // Extend our state to handle editing pages in existing annotations
  const [editingExistingPages, setEditingExistingPages] = useState<number | null>(null);
  const [tempExistingPages, setTempExistingPages] = useState<number[]>([]);

  // Add state for success message
  const [saveSuccess, setSaveSuccess] = useState<boolean>(false);
  const [successMessage, setSuccessMessage] = useState<string>('');

  // Add these state variables for editing existing annotations
  const [editingExistingQuestion, setEditingExistingQuestion] = useState<number | null>(null);
  const [editingExistingAnswer, setEditingExistingAnswer] = useState<number | null>(null);
  const [tempExistingQuestion, setTempExistingQuestion] = useState<string>('');
  const [tempExistingAnswer, setTempExistingAnswer] = useState<string>('');

  // Add this state declaration with the other useState declarations
  const [savedQuestions, setSavedQuestions] = useState<Set<number>>(new Set());

  // Add these state variables in your App component
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState<{
    type: 'pdf' | 'annotation';
    id: string;
    name: string;
  } | null>(null);

  // --- Theme Toggle Effect ---
  useEffect(() => {
    // Apply the theme class to the body element
    document.body.classList.remove('light', 'dark'); // Remove any existing theme class
    document.body.classList.add(theme); // Add the current theme class
    localStorage.setItem('theme', theme); // Save preference to localStorage
  }, [theme]); // Re-run whenever theme changes

  // --- Theme Toggle Function ---
  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  // --- Helper Function to Get Headers ---
  const getAuthHeaders = useCallback((): ApiHeaders => {
    // Read directly from state during the function call for freshness
    const currentApiKey = localStorage.getItem('apiKey'); // Prefer localStorage for persistence across calls
    if (!currentApiKey) {
      console.error("Attempted to make API call without API key.");
      logout(); // Log out if key disappears unexpectedly
      // Throwing here is correct, the caller should handle it
      throw new Error("API Key not found.");
    }
    return { 'Authorization': `Bearer ${currentApiKey}` };
  }, []); // No dependencies needed, logout handles state change

  // --- Edit 3: Logout Function ---
  const logout = () => {
    localStorage.removeItem('apiKey');
    setApiKey('');
    setIsValidApiKey(false);
    setAuthError(null);
    // Reset application state
    setDatasets([]);
    setSelectedDataset('');
    setPdfs([]);
    setSelectedPdf(null);
    setPdfUrl(null);
    setAnnotations([]);
    setQaSuggestions([]);
    setError(null);
    // ... reset other relevant states ...
  };

  // --- Edit 4: API Key Validation Function ---
  const validateApiKey = async (keyToValidate: string) => {
    if (!keyToValidate) {
      setAuthError("API Key cannot be empty.");
      return;
    }
    setAuthError(null); // Clear previous errors
    console.log(`Validating API Key: Bearer ${keyToValidate}`); // Log the key format being sent
    try {
      // --- Use the keyToValidate directly for the validation call ---
      const validationHeaders = { 'Authorization': `Bearer ${keyToValidate}` };
      await api.getDatasets(validationHeaders); // Pass the constructed headers directly

      // If the call succeeds (doesn't throw 401/403), the key is valid
      console.log("API Key validation successful.");
      localStorage.setItem('apiKey', keyToValidate); // NOW save to localStorage
      setApiKey(keyToValidate); // Update state as well
      setIsValidApiKey(true);
      setAuthError(null);
    } catch (err: any) {
      console.error("API Key validation failed:", err);
      localStorage.removeItem('apiKey');
      setIsValidApiKey(false);
      // Provide more specific feedback if possible
      if (err.status === 401 || err.status === 403) { // Check status from enhanced error
        setAuthError("Invalid API Key provided.");
      } else if (err.message?.includes('Failed to fetch') || err.message?.includes('NetworkError')) {
        setAuthError("Failed to connect to the backend. Is it running and accessible?");
      } else {
        setAuthError(`Validation failed: ${err.message}. Check console.`);
      }
    }
  };

  // --- Edit 5: Check localStorage and Validate on Mount ---
  useEffect(() => {
    const savedApiKey = localStorage.getItem('apiKey');
    if (savedApiKey) {
      console.log("Found saved API key, attempting validation...");
      setApiKey(savedApiKey); // Set key state immediately for potential use
      // Call validateApiKey with the saved key directly
      validateApiKey(savedApiKey);
    } else {
      console.log("No saved API key found.");
    }
    // Intentionally run validateApiKey within this effect based on savedApiKey existence.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only once on mount

  // --- Effects ---

  // Fetch datasets on initial load *after* key is validated
  useEffect(() => {
    if (!isValidApiKey) return; // Don't fetch if key is not valid

    const fetchDatasets = async () => {
      setIsLoadingDatasets(true);
      setError(null);
      try {
        // IMPORTANT: Assumes api.getDatasets is modified to accept headers
        const fetchedDatasets = await api.getDatasets(getAuthHeaders());
        setDatasets(fetchedDatasets);
        if (fetchedDatasets.length > 0 && !selectedDataset) {
          setSelectedDataset(fetchedDatasets[0]); // Select the first dataset by default
        }
      } catch (err: any) { // Catch potential auth errors here too
        console.error("Error fetching datasets:", err);
        setError("Failed to load datasets.");
        if (err.message?.includes('401')) logout(); // Logout on auth error
      } finally {
        setIsLoadingDatasets(false);
      }
    };
    fetchDatasets();
  }, [isValidApiKey, getAuthHeaders]); // Re-run if validity changes

  // Fetch PDFs when selectedDataset changes or showOnlyUnannotated changes
  useEffect(() => {
    // Ensure key is valid and dataset is selected
    if (!isValidApiKey || !selectedDataset) return;

    const fetchPdfs = async () => {
      setIsLoadingPdfs(true);
      setError(null);
      setSelectedPdf(null); // Reset selected PDF when dataset changes
      setPdfUrl(null);
      setNumPages(0);
      setCurrentPage(1);
      setAnnotations([]);
      setQaSuggestions([]);
      try {
        // IMPORTANT: Assumes api.getPdfs is modified to accept headers
        const fetchedPdfs = await api.getPdfs(selectedDataset, showOnlyUnannotated, getAuthHeaders());
        setPdfs(fetchedPdfs);
      } catch (err: any) {
        console.error(`Error fetching PDFs for ${selectedDataset}:`, err);
        setError(`Failed to load PDFs for dataset '${selectedDataset}'.`);
        if (err.message?.includes('401')) logout(); // Logout on auth error
      } finally {
        setIsLoadingPdfs(false);
      }
    };
    fetchPdfs();
  }, [selectedDataset, showOnlyUnannotated, isValidApiKey, getAuthHeaders]);

  // Fetch PDF URL and annotations when selectedPdf changes
  useEffect(() => {
    let currentPdfUrl: string | null = null; // Keep track of the created object URL

    if (!isValidApiKey || !selectedDataset || !selectedPdf) {
      setPdfUrl(null);
      setNumPages(0);
      setCurrentPage(1);
      setAnnotations([]);
      setQaSuggestions([]);
      setSelectedPagesForLLM([]);
      setPageInputText(''); // Clear the input text
      return; // Exit early
    }

    const loadPdfData = async () => {
      setIsLoadingAnnotations(true);
      setIsLoadingPdfDocument(true); // Indicate PDF blob loading
      setError(null);
      setAnnotationError(null);
      setPdfUrl(null); // Reset URL while fetching new one
      setNumPages(0);
      setCurrentPage(1);

      try {
        // 1. Fetch the PDF data as a blob using the authenticated API call
        const blob = await api.getPdfBlob(selectedDataset, selectedPdf, getAuthHeaders());

        // 2. Create an object URL from the blob
        currentPdfUrl = URL.createObjectURL(blob);
        setPdfUrl(currentPdfUrl); // Set the object URL for react-pdf

        // Fetch annotations for the selected document
        const docId = selectedPdf.replace(/\.[^/.]+$/, "");
        const fetchedAnnotations = await api.getAnnotations(selectedDataset, docId, getAuthHeaders());
        setAnnotations(fetchedAnnotations);

        // Note: setIsLoadingPdfDocument will be set to false in onDocumentLoadSuccess
      } catch (err: any) { // Catch potential errors from getPdfBlob or getAnnotations
        console.error(`Error loading data for ${selectedPdf}:`, err);
        const errorDetail = err.message || 'Unknown error';
        const statusHint = err.status ? ` (Status: ${err.status})` : '';
        setError(`Failed to load PDF or annotations for '${selectedPdf}'. ${errorDetail}${statusHint}.`);
        setPdfUrl(null); // Clear URL on error
        setIsLoadingPdfDocument(false); // Ensure loading state is turned off on error
        setIsLoadingAnnotations(false);
        if (err.status === 401 || err.message?.includes('401')) {
          logout(); // Logout on auth error
        }
      } finally {
        // Only set isLoadingAnnotations to false if no error occurred during annotation fetch
        // (It's already handled in the catch block if an error happens)
        if (!error) { // Check if the 'error' state was set
          setIsLoadingAnnotations(false);
        }
        // setIsLoadingPdfDocument is handled by react-pdf's onLoadSuccess/onError callbacks
      }
    };

    loadPdfData();

    // 3. Cleanup function: Revoke the object URL when the component unmounts
    //    or when selectedPdf changes (triggering the effect again)
    return () => {
      if (currentPdfUrl) {
        URL.revokeObjectURL(currentPdfUrl);
        console.log("Revoked object URL:", currentPdfUrl); // For debugging
      }
    };
    // Added `error` state to dependency array to potentially clear loading indicators if error occurs
  }, [selectedDataset, selectedPdf, isValidApiKey, getAuthHeaders, error]); // Added 'error' to deps

  // Add this function to handle searching in the PDF
  const handleSearch = async () => {
    if (!searchText.trim() || !pdfUrl) {
      setSearchResults([]);
      return;
    }

    try {
      // Using the PDF.js findController to search - this requires accessing the document viewer
      // Since we can't directly access the findController in react-pdf, let's implement a simple search
      // This is a simplified approach - PDF.js has more robust search capabilities

      // Toggle loading state
      const prevResults = searchResults;
      setSearchResults([{ loading: true }]);

      // This would typically be a more complex integration with PDF.js's find API
      // For now, we'll simulate search results
      setTimeout(() => {
        // Mock results for demonstration
        // In a real implementation, you'd connect with PDF.js's find API
        const mockResults = Array.from({ length: Math.floor(Math.random() * 10) + 1 }, (_, i) => ({
          pageIndex: Math.min(numPages - 1, Math.floor(Math.random() * numPages)),
          matchText: searchText,
          position: { top: 100 + (i * 50), left: 100 }
        }));

        if (mockResults.length > 0) {
          setSearchResults(mockResults);
          // Jump to the first result's page
          goToPage(mockResults[0].pageIndex + 1);
          setCurrentSearchIndex(0);
        } else {
          setSearchResults([{ noResults: true }]);
        }
      }, 500); // Simulate search delay
    } catch (error) {
      console.error("Search error:", error);
      setSearchResults([{ error: true }]);
    }
  };

  // Function to navigate to next/previous search result
  const navigateSearchResults = (direction: 'next' | 'prev') => {
    if (searchResults.length <= 1) return; // No results or only status indicators

    let newIndex;
    if (direction === 'next') {
      newIndex = (currentSearchIndex + 1) % searchResults.length;
    } else {
      newIndex = (currentSearchIndex - 1 + searchResults.length) % searchResults.length;
    }

    setCurrentSearchIndex(newIndex);
    goToPage(searchResults[newIndex].pageIndex + 1);
  };

  // Toggle search bar visibility
  const toggleSearch = () => {
    setShowSearch(!showSearch);
    if (!showSearch) {
      // Reset search state when opening
      setSearchText('');
      setSearchResults([]);
    }
  };

  // Add keyboard shortcut for search
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Handle Ctrl+F or Command+F (Mac)
      if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
        e.preventDefault(); // Prevent browser's default search
        toggleSearch();
      }

      // Handle Escape key to close search
      if (e.key === 'Escape' && showSearch) {
        setShowSearch(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showSearch]);

  // --- Callbacks ---

  const handleDatasetChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedDataset(event.target.value);
    // PDF list will update via useEffect
  };

  const handlePdfSelect = (pdfName: string) => {
    setSelectedPdf(pdfName);
    setNumPages(0);
    setSelectedPagesForLLM([]);
    setQaSuggestions([]);
    setSavedQuestions(new Set()); // Reset saved questions when switching PDFs
    setPageInputText('');
    setLlmError(null);

    // PDF URL and annotations will update via useEffect
  };

  const handleCreateDataset = async () => {
    if (!newDatasetName.trim()) {
      setError("Please enter a valid dataset name.");
      return;
    }
    if (datasets.includes(newDatasetName.trim())) {
      setError(`Dataset '${newDatasetName.trim()}' already exists.`);
      return;
    }
    if (!isValidApiKey) return;
    setError(null);
    setIsLoadingDatasets(true);
    try {
      // IMPORTANT: Assumes api.createDataset is modified to accept headers
      await api.createDataset(newDatasetName.trim(), getAuthHeaders());
      // Refresh dataset list
      // IMPORTANT: Assumes api.getDatasets is modified to accept headers
      const fetchedDatasets = await api.getDatasets(getAuthHeaders());
      setDatasets(fetchedDatasets);
      setSelectedDataset(newDatasetName.trim());
      setNewDatasetName('');
    } catch (err: any) {
      console.error("Error creating dataset:", err);
      setError(`Failed to create dataset '${newDatasetName.trim()}'.`);
      if (err.message?.includes('401')) logout();
    } finally {
      setIsLoadingDatasets(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files || event.target.files.length === 0) {
      return;
    }
    if (!selectedDataset) {
      setUploadError("Please select a dataset first.");
      return;
    }
    if (!isValidApiKey) return;

    const file = event.target.files[0];
    setIsUploading(true);
    setUploadError(null);

    try {
      // IMPORTANT: Assumes api.uploadFile is modified to accept headers
      await api.uploadFile(selectedDataset, file, getAuthHeaders());
      // Refresh PDF list after successful upload
      // IMPORTANT: Assumes api.getPdfs is modified to accept headers
      const fetchedPdfs = await api.getPdfs(selectedDataset, showOnlyUnannotated, getAuthHeaders());
      setPdfs(fetchedPdfs);
      // Optionally select the newly uploaded PDF
      // setSelectedPdf(file.name);
    } catch (err: any) {
      console.error("Error uploading file:", err);
      setUploadError(`Failed to upload file '${file.name}'.`);
      if (err.message?.includes('401')) logout();
    } finally {
      setIsUploading(false);
      // Clear the file input value so the same file can be uploaded again if needed
      event.target.value = '';
    }
  };


  // Callback for react-pdf when document loads successfully
  const onDocumentLoadSuccess = ({ numPages: nextNumPages }: { numPages: number }) => {
    setNumPages(nextNumPages);
    setCurrentPage(1); // Reset to first page on new document load
    setIsLoadingPdfDocument(false); // PDF loaded
    setError(null); // Clear previous errors
    setSelectedPagesForLLM([]); // Reset LLM page selection
  };

  // Callback for react-pdf when document fails to load
  const onDocumentLoadError = (error: Error) => {
    console.error('Failed to load PDF:', error);
    setError(`Failed to load PDF document: ${error.message}. Check backend logs and file path.`);
    setPdfUrl(null); // Clear URL on error
    setNumPages(0);
    setIsLoadingPdfDocument(false);
  };

  // Page navigation handlers
  const goToPrevPage = () => setCurrentPage(prev => Math.max(1, prev - 1));
  const goToNextPage = () => setCurrentPage(prev => Math.min(numPages, prev + 1));
  const goToPage = (page: number) => {
    const targetPage = Math.max(1, Math.min(numPages, page));
    setCurrentPage(targetPage);
  }

  // --- TODO: Implement Annotation/QA Handlers ---
  const handleSaveAnnotation = async (suggestion: QASuggestion, index: number) => {
    if (!isValidApiKey || !selectedDataset || !selectedPdf) return;

    const docId = selectedPdf.replace(/\.[^/.]+$/, "");
    const annotationPayload = { // Define payload structure clearly
      question: suggestion.question,
      answers: [suggestion.answer], // Assuming answer is a single string
      doc_id: docId,
      page_ids: suggestion.source_pages.map(p => `${docId}_p${p}`),
      data_split: "test", // Or derive as needed
      dataset_id: selectedDataset // Include if required by backend
    };

    try {
      setAnnotationError(null);
      // IMPORTANT: Assumes api.saveAnnotation is modified to accept headers
      await api.saveAnnotation(selectedDataset, annotationPayload, getAuthHeaders());

      // Refresh annotations
      // IMPORTANT: Assumes api.getAnnotations is modified to accept headers
      const updatedAnnotations = await api.getAnnotations(selectedDataset, docId, getAuthHeaders());
      setAnnotations(updatedAnnotations);

      // Mark as saved
      setSavedSuggestions(prev => new Set([...prev, index]));
      setSavedQuestions(prev => new Set([...prev, index]));

      refreshStats(); // Refresh stats after saving
    } catch (err: any) {
      console.error("Error saving annotation:", err);
      setAnnotationError(err.response?.data?.detail || "Failed to save annotation");
      if (err.message?.includes('401')) logout();
    }
  };

  const handleSaveQuestion = async (index: number) => {
    if (!selectedDataset || !selectedPdf) return;

    try {
      await api.saveQA(
        selectedDataset,
        selectedPdf,
        qaSuggestions[index].pages,
        qaSuggestions[index].question,
        qaSuggestions[index].answer
      );

      // Update saved questions set
      setSavedQuestions(prev => new Set([...prev, index]));

      // Refresh annotations
      fetchAnnotations();
    } catch (err) {
      console.error("Error saving QA:", err);
    }
  };

  const handleGenerateQA = async () => {
    if (!isValidApiKey || !selectedDataset || !selectedPdf) {
      setLlmError("Please select a PDF first");
      return;
    }

    // Clear both suggestions and saved state
    setQaSuggestions([]);
    setSavedQuestions(new Set());
    setSavedSuggestions(new Set()); // Also reset savedSuggestions state

    let pagesToUse = selectedPagesForLLM;
    if (pagesToUse.length === 0) {
      pagesToUse = Array.from({ length: numPages }, (_, i) => i);
      setLlmError(`No pages specified. Using all ${pagesToUse.length} pages.`);

      const MAX_SAFE_PAGES = 20;
      if (pagesToUse.length > MAX_SAFE_PAGES) {
        setLlmError(`Warning: Using a large number of pages (${pagesToUse.length}). This may take longer to process.`);
      }
    } else {
      setLlmError(null);
    }

    const questionsToGenerate = numQuestions || 3;
    setIsGeneratingQA(true);

    try {
      // IMPORTANT: Assumes api.generateQA is modified to accept headers
      const suggestions = await api.generateQA(
        selectedDataset,
        selectedPdf,
        pagesToUse,
        questionsToGenerate,
        getAuthHeaders() // Pass headers
      );

      if (suggestions && suggestions.length > 0) {
        setQaSuggestions(suggestions);
        setLlmError(null);
      } else {
        setLlmError("No suggestions generated. Try different pages.");
      }
    } catch (err: any) {
      console.error("Error generating QA:", err);
      setLlmError(err.response?.data?.detail || "Failed to generate questions. Please try again.");
      setQaSuggestions([]);
      if (err.message?.includes('401')) logout();
    } finally {
      setIsGeneratingQA(false);
    }
  };

  const handleAddSuggestion = async ( /* suggestion data */) => { /* ... */ };
  const handlePageSelectForLLM = (pageIndex: number) => { // pageIndex is 0-based
    setSelectedPagesForLLM(prev =>
      prev.includes(pageIndex)
        ? prev.filter(p => p !== pageIndex)
        : [...prev, pageIndex]
    );
  };

  function parsePageInput(input: string, numPages: number): number[] {
    const result = new Set<number>();
    const parts = input.split(',').map(part => part.trim());

    parts.forEach(part => {
      if (part.includes('-')) {
        const [startStr, endStr] = part.split('-').map(num => num.trim());
        const start = parseInt(startStr);
        const end = parseInt(endStr);

        if (!isNaN(start) && !isNaN(end)) {
          // Convert 1-based input to 0-based index
          for (let i = start - 1; i <= end - 1 && i < numPages; i++) {
            if (i >= 0) {  // Ensure we don't add negative indices
              result.add(i);
            }
          }
        }
      } else {
        const page = parseInt(part);
        if (!isNaN(page)) {
          // Convert 1-based input to 0-based index
          const pageIndex = page - 1;
          if (pageIndex >= 0 && pageIndex < numPages) {
            result.add(pageIndex);
          }
        }
      }
    });

    return Array.from(result).sort((a, b) => a - b);
  }

  // Add zoom handlers
  const handleZoomIn = () => setScale(prev => Math.min(prev + 0.2, 3.0));
  const handleZoomOut = () => setScale(prev => Math.max(prev - 0.2, 0.5));
  const handleResetZoom = () => setScale(1.0);

  const startEditing = (index: number, suggestion: QASuggestion) => {
    setEditingSuggestion(index);
    setEditedQuestion(suggestion.question);
    setEditedAnswer(suggestion.answer);

    // We're not editing pages yet, just the question and answer
    setEditingPagesForSuggestion(null);
  };

  const saveEdits = (index: number) => {
    if (editingSuggestion !== index) return;

    // Update the suggestion with edited content
    setQaSuggestions(prev =>
      prev.map((s, i) =>
        i === index
          ? { ...s, question: editedQuestion, answer: editedAnswer }
          : s
      )
    );

    // Clear editing state
    setEditingSuggestion(null);
    setEditedQuestion('');
    setEditedAnswer('');
  };

  const cancelEditing = () => {
    setEditingSuggestion(null);
    setEditedQuestion('');
    setEditedAnswer('');
    setEditingPagesForSuggestion(null);
    setTempEditedPages([]);
  };

  // Update handleTabChange to save state
  const handleTabChange = (tab: 'manual' | 'llm' | 'existing') => {
    setActiveTab(tab);

    // Save tab state for this PDF
    if (selectedPdf) {
      setPdfStates(prev => ({
        ...prev,
        [selectedPdf]: {
          ...prev[selectedPdf],
          activeTab: tab
        }
      }));
    }
  };

  // Add a function to save other states
  const savePdfState = () => {
    if (!selectedPdf) return;

    setPdfStates(prev => ({
      ...prev,
      [selectedPdf]: {
        activeTab,
        selectedPagesForLLM,
        pageInputText,
        numQuestions
      }
    }));
  };

  // Effect to restore state when PDF changes
  useEffect(() => {
    if (!selectedPdf) return;

    const savedState = pdfStates[selectedPdf];
    if (savedState) {
      // Restore saved state for this PDF
      setActiveTab(savedState.activeTab);
      setSelectedPagesForLLM(savedState.selectedPagesForLLM);
      setPageInputText(savedState.pageInputText);
      setNumQuestions(savedState.numQuestions);
    } else {
      // Reset to defaults for new PDFs
      setActiveTab('manual');
      setSelectedPagesForLLM([]);
      setPageInputText('');
      setNumQuestions(3);
    }
  }, [selectedPdf]);

  // Modify existing state-changing functions to save state
  const handlePageInputChange = (input: string) => {
    setPageInputText(input);
    const pages = parsePageInput(input, numPages);
    setSelectedPagesForLLM(pages);

    // Save state after change
    if (selectedPdf) {
      setPdfStates(prev => ({
        ...prev,
        [selectedPdf]: {
          ...prev[selectedPdf] || {},
          pageInputText: input,
          selectedPagesForLLM: pages
        }
      }));
    }
  };

  const handleNumQuestionsChange = (value: number) => {
    const newValue = Math.max(1, Math.min(10, value || 3));
    setNumQuestions(newValue);

    // Save state after change
    if (selectedPdf) {
      setPdfStates(prev => ({
        ...prev,
        [selectedPdf]: {
          ...prev[selectedPdf] || {},
          numQuestions: newValue
        }
      }));
    }
  };

  // Function to fetch dataset statistics
  const fetchDatasetStats = useCallback(async (datasetId: string) => {
    if (!datasetId) return;

    setDatasetStats(prev => ({ ...prev, loading: true }));

    try {
      // Get PDFs count
      const allPdfs = await api.getPdfs(datasetId, false, getAuthHeaders()); // Seems okay if PDF list works

      // Get all annotations FOR THE ENTIRE DATASET
      const allAnnotations = await api.getAnnotations(datasetId, undefined, getAuthHeaders()); // Passing undefined docId

      // Count unique documents with annotations
      const docsWithAnnotations = new Set();
      allAnnotations.forEach(ann => {
        if (ann.doc_id) docsWithAnnotations.add(ann.doc_id);
      });

      // Calculate statistics
      const totalDocs = allPdfs.length;
      const totalQuestions = allAnnotations.length; // If allAnnotations is empty, this will be 0
      const avgQuestionsPerDoc = docsWithAnnotations.size > 0 // If docsWithAnnotations is empty, this will be 0
        ? totalQuestions / docsWithAnnotations.size
        : 0;

      setDatasetStats({
        totalDocuments: totalDocs,
        totalQuestions: totalQuestions,
        averageQuestionsPerDoc: avgQuestionsPerDoc,
        loading: false
      });
    } catch (error) {
      console.error('Error fetching dataset stats:', error);
      setDatasetStats({
        totalDocuments: 0,
        totalQuestions: 0,
        averageQuestionsPerDoc: 0,
        loading: false
      });
      // Check if it was an auth error
      if ((error as any).status === 401 || (error as any).message?.includes('401')) {
        logout();
      }
    }
  }, [getAuthHeaders]); // Dependency on getAuthHeaders is correct

  // Update stats when dataset changes
  useEffect(() => {
    if (selectedDataset) {
      fetchDatasetStats(selectedDataset);
    }
  }, [selectedDataset, fetchDatasetStats]); // Dependencies seem correct

  // Refresh stats after adding annotations
  const refreshStats = () => {
    if (selectedDataset) {
      fetchDatasetStats(selectedDataset);
    }
  };

  // Add these handler functions
  const handleDeleteAnnotation = async (annotationId: string) => {
    if (!isValidApiKey || !selectedDataset) return;

    try {
      // IMPORTANT: Assumes api.deleteAnnotation is modified to accept headers
      await api.deleteAnnotation(selectedDataset, annotationId, getAuthHeaders());
      // Refresh annotations
      if (selectedPdf) {
        const docId = selectedPdf.replace(/\.[^/.]+$/, "");
        // IMPORTANT: Assumes api.getAnnotations is modified to accept headers
        const updatedAnnotations = await api.getAnnotations(selectedDataset, docId, getAuthHeaders());
        setAnnotations(updatedAnnotations);
      }
      setShowDeleteConfirmation(null);
      refreshStats(); // Refresh stats after deleting
    } catch (err: any) {
      console.error("Error deleting annotation:", err);
      setAnnotationError("Failed to delete annotation");
      if (err.message?.includes('401')) logout();
    }
  };

  const handleDeletePdf = async (pdfFilename: string) => {
    if (!isValidApiKey || !selectedDataset) return;

    try {
      // IMPORTANT: Assumes api.deletePdf is modified to accept headers
      await api.deletePdf(selectedDataset, pdfFilename, getAuthHeaders());
      // Refresh PDF list
      // IMPORTANT: Assumes api.getPdfs is modified to accept headers
      const updatedPdfs = await api.getPdfs(selectedDataset, showOnlyUnannotated, getAuthHeaders());
      setPdfs(updatedPdfs);
      if (selectedPdf === pdfFilename) {
        setSelectedPdf(null);
        setPdfUrl(null);
        setAnnotations([]);
      }
      setShowDeleteConfirmation(null);
      refreshStats(); // Refresh stats after deleting
    } catch (err: any) {
      console.error("Error deleting PDF:", err);
      setError("Failed to delete PDF");
      if (err.message?.includes('401')) logout();
    }
  };

  // Add the confirmation modal component
  const DeleteConfirmationModal = () => {
    if (!showDeleteConfirmation) return null;

    return (
      <>
        <div className="modal-backdrop" onClick={() => setShowDeleteConfirmation(null)} />
        <div className="confirmation-modal">
          <h3>Confirm Delete</h3>
          <p>Are you sure you want to delete this {showDeleteConfirmation.type}?</p>
          <p><strong>{showDeleteConfirmation.name}</strong></p>
          <div className="modal-actions">
            <button onClick={() => setShowDeleteConfirmation(null)}>Cancel</button>
            <button
              className="delete-button"
              onClick={() => {
                if (showDeleteConfirmation.type === 'pdf') {
                  handleDeletePdf(showDeleteConfirmation.id);
                } else {
                  handleDeleteAnnotation(showDeleteConfirmation.id);
                }
              }}
            >
              Delete
            </button>
          </div>
        </div>
      </>
    );
  };

  // --- Rendering ---
  if (!isValidApiKey) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <h2>Enter API Key</h2>
          <p>Please provide the API key to access the annotator.</p>
          <input
            type="password" // Use password type for sensitive keys
            placeholder="API Key"
            value={apiKey}
            onChange={(e) => {
              setApiKey(e.target.value);
              // Optionally clear error when user starts typing new key
              if (authError) setAuthError(null);
            }}
            className="auth-input"
          />
          {authError && <p className="error-message auth-error">{authError}</p>}
          <button
            onClick={() => validateApiKey(apiKey)} // Pass current input value
            className="auth-button"
            disabled={!apiKey.trim()} // Disable button if input is empty
          >
            Validate & Enter
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>Datasets</h2>
          {/* Theme Toggle Button */}
          <div className="sidebar-controls">
            <button onClick={toggleTheme} className="theme-toggle-button" title={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}>
              {theme === 'light' ? <FiMoon size={18} /> : <FiSun size={18} />}
            </button>
            <button onClick={logout} className="logout-button" title="Logout">
              <FiLogOut size={18} /> Logout
            </button>
          </div>
        </div>
        {isLoadingDatasets && <p>Loading datasets...</p>}
        {error && <p className="error-message">{error}</p>}
        {!isLoadingDatasets && datasets.length > 0 && (
          <select value={selectedDataset} onChange={handleDatasetChange} disabled={isLoadingPdfs}>
            {datasets.map(ds => (
              <option key={ds} value={ds}>{ds}</option>
            ))}
          </select>
        )}
        {!isLoadingDatasets && datasets.length === 0 && !error && (
          <p>No datasets found. Create one below.</p>
        )}

        {/* Dataset Statistics */}
        {selectedDataset && (
          <div className="dataset-stats">
            <h3>Dataset Statistics</h3>
            {datasetStats.loading ? (
              <div className="stats-loading">Loading stats...</div>
            ) : (
              <div className="stats-grid">
                <div className="stat-item">
                  <div className="stat-value">{datasetStats.totalDocuments}</div>
                  <div className="stat-label">Documents</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{datasetStats.totalQuestions}</div>
                  <div className="stat-label">Questions</div>
                </div>
                <div className="stat-item">
                  <div className="stat-value">{datasetStats.averageQuestionsPerDoc.toFixed(1)}</div>
                  <div className="stat-label">Avg Q/Doc</div>
                </div>
                <button
                  className="refresh-stats-button"
                  onClick={refreshStats}
                  title="Refresh Statistics"
                >
                  ↻
                </button>
              </div>
            )}
          </div>
        )}

        <hr className="sidebar-divider" />

        <h2>PDF Documents</h2>
        {/* Improved File Upload Drop Area */}
        <div
          className={`upload-container ${isUploading ? 'uploading' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            if (!selectedDataset || isUploading) return;

            const files = e.dataTransfer.files;
            if (files.length > 0) {
              const fileInput = document.getElementById('file-upload') as HTMLInputElement;
              fileInput.files = files;
              handleFileUpload({ target: { files: files } } as any);
            }
          }}
        >
          <div className="upload-inner">
            <div className="upload-icon-container">
              <div className="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M12 16V4M12 4L7 9M12 4L17 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M3 19H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
              </div>
            </div>
            <div className="upload-text">
              <p className="upload-title">Drag & drop PDF or image files here</p>
              <p className="upload-subtitle">or</p>
              <label htmlFor="file-upload" className={`upload-button ${isUploading ? 'disabled' : ''}`}>
                {isUploading ? 'Uploading...' : 'Browse Files'}
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".pdf,.jpg,.jpeg,.png"
                onChange={handleFileUpload}
                disabled={isUploading || !selectedDataset}
                style={{ display: 'none' }}
              />
            </div>
          </div>
          {uploadError && <p className="error-message">{uploadError}</p>}
        </div>

        {selectedDataset && (
          <>
            <div className="filter-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={showOnlyUnannotated}
                  onChange={(e) => setShowOnlyUnannotated(e.target.checked)}
                  disabled={isLoadingPdfs}
                />
                Show only unannotated
              </label>
            </div>
            {isLoadingPdfs && <p>Loading PDFs...</p>}
            {!isLoadingPdfs && pdfs.length === 0 && <p>No PDFs found in this dataset.</p>}
            {!isLoadingPdfs && pdfs.length > 0 && (
              <div className="pdf-list">
                {pdfs.map(pdf => (
                  <div
                    key={pdf}
                    className={`pdf-list-item ${selectedPdf === pdf ? 'active' : ''}`}
                    onClick={() => handlePdfSelect(pdf)}
                  >
                    <span style={{ flex: 1 }}>
                      {pdf}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        {/* Create Dataset Section */}
        <div className="create-dataset">
          <input
            type="text"
            placeholder="New dataset name"
            value={newDatasetName}
            onChange={(e) => setNewDatasetName(e.target.value)}
            disabled={isLoadingDatasets}
          />
          <button
            onClick={handleCreateDataset}
            disabled={isLoadingDatasets || !newDatasetName.trim()}
          >
            {isLoadingDatasets ? 'Creating...' : 'Create Dataset'}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* PDF Viewer Area */}
        <section className="pdf-viewer-wrapper">
          <div className="pdf-header">
            <h2 className="pdf-viewer-title">PDF Viewer {selectedPdf ? `- ${selectedPdf}` : ''}</h2>
            <div className="pdf-header-controls">
              {/* Add search toggle button */}
              {pdfUrl && (
                <>
                  <button
                    onClick={toggleSearch}
                    className="search-toggle-button"
                    title="Search in PDF (Ctrl+F)"
                  >
                    🔍
                  </button>
                  <button
                    className="delete-pdf-button"
                    onClick={() => {
                      setShowDeleteConfirmation({
                        type: 'pdf',
                        id: selectedPdf!,
                        name: selectedPdf!
                      });
                    }}
                    title="Delete PDF"
                  >
                    🗑️
                  </button>
                </>
              )}
            </div>
          </div>

          {/* Search Bar */}
          {showSearch && pdfUrl && (
            <div className="search-container">
              <div className="search-input-wrapper">
                <input
                  type="text"
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  placeholder="Search in document..."
                  className="search-input"
                  autoFocus
                />
                <button
                  onClick={handleSearch}
                  className="search-button"
                  disabled={!searchText.trim()}
                >
                  Search
                </button>
                <button
                  onClick={() => setShowSearch(false)}
                  className="search-close-button"
                >
                  ✕
                </button>
              </div>

              {searchResults.length > 0 && (
                <div className="search-results-nav">
                  {searchResults[0]?.loading ? (
                    <span>Searching...</span>
                  ) : searchResults[0]?.error ? (
                    <span>Error searching document</span>
                  ) : searchResults[0]?.noResults ? (
                    <span>No results found</span>
                  ) : (
                    <>
                      <span>{currentSearchIndex + 1} of {searchResults.length} results</span>
                      <div className="search-nav-buttons">
                        <button
                          onClick={() => navigateSearchResults('prev')}
                          disabled={searchResults.length <= 1}
                        >
                          ▲
                        </button>
                        <button
                          onClick={() => navigateSearchResults('next')}
                          disabled={searchResults.length <= 1}
                        >
                          ▼
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {isLoadingPdfDocument && <p>Loading PDF document...</p>}
          {!selectedPdf && !isLoadingPdfDocument && <p>Select a PDF from the list to view.</p>}
          {pdfUrl && (
            <>
              <div className="zoom-controls">
                <button onClick={handleZoomOut} title="Zoom Out">-</button>
                <button onClick={handleResetZoom} title="Reset Zoom">{Math.round(scale * 100)}%</button>
                <button onClick={handleZoomIn} title="Zoom In">+</button>
              </div>
              <div className="pdf-document-container">
                <Document
                  file={pdfUrl}
                  onLoadSuccess={onDocumentLoadSuccess}
                  onLoadError={onDocumentLoadError}
                  loading={<p>Loading PDF preview...</p>}
                  error={<p>Error loading PDF preview.</p>}
                >
                  <Page
                    pageNumber={currentPage}
                    width={700 * scale}
                    renderTextLayer={true}
                    renderAnnotationLayer={true}
                  />
                </Document>
              </div>
            </>
          )}
          {numPages > 0 && (
            <div className="pdf-navigation">
              <button onClick={goToPrevPage} disabled={currentPage <= 1}>Previous</button>
              <span>Page {currentPage} of {numPages}</span>
              <button onClick={goToNextPage} disabled={currentPage >= numPages}>Next</button>
            </div>
          )}
          {error && !pdfUrl && <p className="error-message">{error}</p>}
        </section>

        {/* Annotation Area */}
        <section className="annotation-wrapper">
          <h2>Annotations</h2>
          {!selectedPdf && <p>Select a PDF to manage annotations.</p>}
          {selectedPdf && (
            <>
              {/* Fix the tabs to make sure they're fully clickable */}
              <div className="annotation-tabs">
                <button
                  className={`tab-button ${activeTab === 'manual' ? 'active' : ''}`}
                  onClick={() => handleTabChange('manual')}
                >
                  Manual Entry
                </button>
                <button
                  className={`tab-button ${activeTab === 'llm' ? 'active' : ''}`}
                  onClick={() => handleTabChange('llm')}
                >
                  LLM Suggestions
                </button>
                <button
                  className={`tab-button ${activeTab === 'existing' ? 'active' : ''}`}
                  onClick={() => handleTabChange('existing')}
                >
                  Existing Q&A
                </button>
              </div>

              <div className="annotation-content">
                {/* Manual Annotation Tab */}
                <div className={`tab-panel ${activeTab === 'manual' ? 'active' : ''}`}>
                  <h3>Add New Q&A</h3>

                  {/* Success message */}
                  {saveSuccess && (
                    <div style={{
                      backgroundColor: '#f6ffed',
                      border: '1px solid #b7eb8f',
                      color: '#52c41a',
                      padding: '10px 15px',
                      borderRadius: '4px',
                      marginBottom: '15px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between'
                    }}>
                      <span>
                        <strong>✓ Saved!</strong> {successMessage}
                      </span>
                    </div>
                  )}

                  {/* Error message */}
                  {annotationError && (
                    <div style={{
                      backgroundColor: '#fff2f0',
                      border: '1px solid #ffccc7',
                      color: '#f5222d',
                      padding: '10px 15px',
                      borderRadius: '4px',
                      marginBottom: '15px'
                    }}>
                      <strong>Error:</strong> {annotationError}
                    </div>
                  )}

                  {/* Rest of form elements */}
                  <textarea
                    placeholder="Enter question..."
                    rows={3}
                    style={{ width: '100%', marginBottom: '10px', padding: '8px' }}
                    value={editedQuestion}
                    onChange={(e) => setEditedQuestion(e.target.value)}
                    disabled={isGeneratingQA}
                  ></textarea>

                  <textarea
                    placeholder="Enter answer..."
                    rows={3}
                    style={{ width: '100%', marginBottom: '15px', padding: '8px' }}
                    value={editedAnswer}
                    onChange={(e) => setEditedAnswer(e.target.value)}
                    disabled={isGeneratingQA}
                  ></textarea>

                  {/* Improved page selection UI */}
                  <div style={{ marginBottom: '15px', padding: '10px', border: '1px solid #f0f0f0', borderRadius: '4px' }}>
                    <p><strong>Select relevant pages:</strong></p>

                    {/* Quick add pages input */}
                    <div style={{ marginBottom: '10px', display: 'flex', gap: '10px' }}>
                      <input
                        type="text"
                        value={pageInputText}
                        onChange={(e) => handlePageInputChange(e.target.value)}
                        placeholder="Enter page numbers (e.g., 1,2,3 or 1-3)"
                        style={{ flex: 1, padding: '8px' }}
                        disabled={isGeneratingQA}
                      />
                      <button
                        onClick={() => {
                          const pages = parsePageInput(pageInputText, numPages);
                          console.log("Parsed pages:", pages);
                          setSelectedPagesForLLM(pages);
                        }}
                        disabled={isGeneratingQA || !pageInputText.trim()}
                        style={{ padding: '8px 12px' }}
                      >
                        Add Pages
                      </button>
                    </div>

                    {/* Pages grid */}
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', maxHeight: '200px', overflowY: 'auto' }}>
                      {numPages > 0 ? (
                        Array.from({ length: numPages }, (_, i) => (
                          <div
                            key={i}
                            style={{ padding: '5px', border: '1px solid #ddd', borderRadius: '4px', cursor: 'pointer', margin: '3px' }}
                            onClick={() => handlePageSelectForLLM(i)}
                          >
                            <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                              <input
                                type="checkbox"
                                checked={selectedPagesForLLM.includes(i)}
                                onChange={() => {/* Event handled by div onClick */ }}
                                disabled={isGeneratingQA}
                                style={{ marginRight: '5px' }}
                              />
                              Page {i + 1}
                            </label>
                          </div>
                        ))
                      ) : (
                        <p>No pages available. Please select a PDF first.</p>
                      )}
                    </div>

                    {/* Selected pages summary */}
                    {selectedPagesForLLM.length > 0 && (
                      <div style={{ marginTop: '10px', fontSize: '0.9em', color: '#1890ff' }}>
                        Selected {selectedPagesForLLM.length} page(s): {selectedPagesForLLM.map(p => p + 1).join(', ')}
                      </div>
                    )}
                  </div>

                  {/* Save button with success indicator */}
                  <button
                    onClick={() => handleSaveAnnotation({ question: editedQuestion, answer: editedAnswer, source_pages: selectedPagesForLLM }, 0)}
                    disabled={isGeneratingQA || !editedQuestion.trim() || !editedAnswer.trim() || selectedPagesForLLM.length === 0}
                    style={{
                      padding: '10px 15px',
                      backgroundColor: saveSuccess ? '#52c41a' : '#1890ff',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: isGeneratingQA || !editedQuestion.trim() || !editedAnswer.trim() || selectedPagesForLLM.length === 0 ? 'not-allowed' : 'pointer',
                      opacity: isGeneratingQA || !editedQuestion.trim() || !editedAnswer.trim() || selectedPagesForLLM.length === 0 ? 0.7 : 1,
                      transition: 'background-color 0.3s'
                    }}
                  >
                    {isGeneratingQA ? 'Generating...' : (saveSuccess ? '✓ Saved!' : 'Save Annotation')}
                  </button>
                </div>

                {/* LLM Suggestions Tab */}
                <div className={`tab-panel ${activeTab === 'llm' ? 'active' : ''}`}>
                  <h3>Generate Q&A with LLM</h3>

                  <div className="llm-options">
                    <div className="llm-pages-input">
                      <label htmlFor="pages-input">Pages (e.g., 1,2,3 or 1-10):</label>
                      <input
                        id="pages-input"
                        type="text"
                        value={pageInputText}
                        onChange={(e) => handlePageInputChange(e.target.value)}
                        placeholder="Leave empty to use all pages"
                        disabled={isGeneratingQA}
                      />
                      <div className="input-help">
                        {selectedPagesForLLM.length > 0
                          ? `Selected ${selectedPagesForLLM.length} pages`
                          : "No pages selected, will use all pages"}
                      </div>
                    </div>

                    <div className="llm-questions-input">
                      <label htmlFor="questions-input">Number of questions:</label>
                      <input
                        id="questions-input"
                        type="number"
                        min="1"
                        max="10"
                        value={numQuestions}
                        onChange={(e) => handleNumQuestionsChange(parseInt(e.target.value))}
                        placeholder="3"
                        disabled={isGeneratingQA}
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleGenerateQA}
                    disabled={isGeneratingQA || !selectedPdf}
                    className={`generate-button ${isGeneratingQA ? 'loading' : ''}`}
                  >
                    {isGeneratingQA ? 'Generating...' : 'Generate Suggestions'}
                  </button>

                  {llmError && (
                    <div className="error-message">
                      {llmError}
                    </div>
                  )}

                  {isGeneratingQA && (
                    <div className="loading-container">
                      <div className="loading-spinner"></div>
                      <p>Generating questions...</p>
                    </div>
                  )}

                  {!isGeneratingQA && qaSuggestions.length > 0 && (
                    <div className="qa-suggestions">
                      <h4>Generated Questions:</h4>
                      {qaSuggestions.map((suggestion, index) => (
                        <div key={index} className={`suggestion-item ${savedSuggestions.has(index) ? 'saved' : ''}`}>
                          <div className="suggestion-header">
                            <h5>Suggestion {index + 1}</h5>
                            <div className="suggestion-actions">
                              {editingPagesForSuggestion === index ? (
                                // Page editing mode
                                <div className="page-edit-mode">
                                  <div className="current-pages">
                                    {tempEditedPages.map(page => (
                                      <span key={page} className="editable-page">
                                        {page}
                                        <button
                                          className="remove-page"
                                          onClick={() => removePageFromEditing(page)}
                                        >×</button>
                                      </span>
                                    ))}
                                    <input
                                      type="number"
                                      className="add-page-input"
                                      min="0"
                                      max={numPages - 1}
                                      placeholder="Add page"
                                      onKeyDown={(e) => {
                                        if (e.key === 'Enter') {
                                          const input = e.target as HTMLInputElement;
                                          const pageNum = parseInt(input.value);
                                          if (!isNaN(pageNum) && pageNum >= 0 && pageNum < numPages) {
                                            addPageToEditing(pageNum);
                                            input.value = '';
                                          }
                                        }
                                      }}
                                    />
                                  </div>
                                  <div className="edit-page-actions">
                                    <button
                                      className="save-pages-btn"
                                      onClick={() => saveEditedPages(index)}
                                    >
                                      Save
                                    </button>
                                    <button
                                      className="cancel-pages-btn"
                                      onClick={cancelEditingPages}
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                // Normal display mode for pages
                                <div className="page-tags">
                                  {suggestion.source_pages.map(page => (
                                    <button
                                      key={page}
                                      className="page-tag"
                                      onClick={() => goToPage(page + 1)}
                                      title={`Go to page ${page + 1}`}
                                    >
                                      {page + 1}
                                    </button>
                                  ))}
                                  {!savedSuggestions.has(index) && (
                                    <button
                                      className="edit-pages-btn"
                                      onClick={() => startEditingPages(index, suggestion.source_pages)}
                                      title="Edit page numbers"
                                    >
                                      ✎
                                    </button>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                          <div className="suggestion-content">
                            {editingSuggestion === index ? (
                              // Editing mode for question and answer
                              <>
                                <div className="edit-field">
                                  <label>Question:</label>
                                  <textarea
                                    className="edit-question"
                                    value={editedQuestion}
                                    onChange={(e) => setEditedQuestion(e.target.value)}
                                    placeholder="Question"
                                    rows={2}
                                  />
                                </div>
                                <div className="edit-field">
                                  <label>Answer:</label>
                                  <textarea
                                    className="edit-answer"
                                    value={editedAnswer}
                                    onChange={(e) => setEditedAnswer(e.target.value)}
                                    placeholder="Answer"
                                    rows={3}
                                  />
                                </div>
                                <div className="edit-actions">
                                  <button onClick={() => saveEdits(index)} className="save-edits-btn">
                                    Save Changes
                                  </button>
                                  <button onClick={cancelEditing} className="cancel-button">
                                    Cancel
                                  </button>
                                </div>
                              </>
                            ) : (
                              // Display mode for question and answer
                              <>
                                <div className="question-answer-container">
                                  <p className="question">
                                    <strong>Q:</strong> <FormattedText text={suggestion.question} />
                                  </p>
                                  <p className="answer">
                                    <strong>A:</strong> <FormattedText text={suggestion.answer} />
                                  </p>

                                  {!savedSuggestions.has(index) && (
                                    <button
                                      onClick={() => startEditing(index, suggestion)}
                                      className="edit-qa-btn"
                                      title="Edit question and answer"
                                    >
                                      ✎ Edit Q&A
                                    </button>
                                  )}
                                </div>

                                <button
                                  onClick={() => handleSaveAnnotation(suggestion, index)}
                                  className={`save-suggestion-btn ${savedSuggestions.has(index) ? 'saved' : ''}`}
                                  disabled={savedSuggestions.has(index)}
                                >
                                  {savedSuggestions.has(index) ? (
                                    <>✓ Saved to Dataset</>
                                  ) : (
                                    <>💾 Save to Dataset</>
                                  )}
                                </button>
                              </>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Existing Annotations Tab */}
                <div className={`tab-panel ${activeTab === 'existing' ? 'active' : ''}`}>
                  <h3>Existing Annotations for {selectedPdf}</h3>
                  {isLoadingAnnotations && <p>Loading annotations...</p>}
                  {!isLoadingAnnotations && annotations.length === 0 && <p>No annotations yet for this document.</p>}
                  {annotationError && (
                    <div className="error-message" style={{
                      color: '#f5222d',
                      backgroundColor: '#fff2f0',
                      padding: '10px',
                      marginBottom: '15px',
                      borderRadius: '4px',
                      border: '1px solid #ffccc7'
                    }}>
                      {annotationError}
                    </div>
                  )}
                  {!isLoadingAnnotations && annotations.length > 0 && (
                    <ul className="existing-annotations-list">
                      {annotations.map((annotation, index) => {
                        const pageNumbers = annotation.page_ids
                          .map(id => {
                            const match = id.match(/_p(\d+)$/);
                            return match ? parseInt(match[1]) + 1 : null;  // Add 1 here
                          })
                          .filter(num => num !== null) as number[];

                        return (
                          <li key={annotation.annotation_id || index} className="annotation-item">
                            <div className="annotation-content">
                              {/* Question section with edit capability */}
                              <div className="annotation-question">
                                {editingExistingQuestion === index ? (
                                  <div className="edit-question-container">
                                    <label><strong>Question:</strong></label>
                                    <textarea
                                      value={tempExistingQuestion}
                                      onChange={(e) => setTempExistingQuestion(e.target.value)}
                                      className="edit-existing-text"
                                      rows={2}
                                    />
                                    <div className="edit-actions">
                                      <button
                                        onClick={() => saveEditedExistingQuestion(index, annotation)}
                                        className="save-edit-btn"
                                        disabled={!tempExistingQuestion.trim()}
                                      >
                                        Save
                                      </button>
                                      <button
                                        onClick={cancelEditingExistingQuestion}
                                        className="cancel-edit-btn"
                                      >
                                        Cancel
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="question-display">
                                    <strong>Q:</strong> <FormattedText text={annotation.question} />
                                    <button
                                      className="edit-btn"
                                      onClick={() => startEditingExistingQuestion(index, annotation.question)}
                                      title="Edit question"
                                    >
                                      ✎
                                    </button>
                                  </div>
                                )}
                              </div>

                              {/* Answer section with edit capability */}
                              <div className="annotation-answer">
                                {editingExistingAnswer === index ? (
                                  <div className="edit-answer-container">
                                    <label><strong>Answer:</strong></label>
                                    <textarea
                                      value={tempExistingAnswer}
                                      onChange={(e) => setTempExistingAnswer(e.target.value)}
                                      className="edit-existing-text"
                                      rows={3}
                                      placeholder="For multiple answers, separate with semicolons (;)"
                                    />
                                    <div className="edit-actions">
                                      <button
                                        onClick={() => saveEditedExistingAnswer(index, annotation)}
                                        className="save-edit-btn"
                                        disabled={!tempExistingAnswer.trim()}
                                      >
                                        Save
                                      </button>
                                      <button
                                        onClick={cancelEditingExistingAnswer}
                                        className="cancel-edit-btn"
                                      >
                                        Cancel
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="answer-display">
                                    <strong>A:</strong> {annotation.answers.map((answer, i) => (
                                      <FormattedText key={i} text={answer} />
                                    ))}
                                    <button
                                      className="edit-btn"
                                      onClick={() => startEditingExistingAnswer(index, annotation.answers)}
                                      title="Edit answer"
                                    >
                                      ✎
                                    </button>
                                  </div>
                                )}
                              </div>

                              {/* Pages section with editing capability */}
                              <div className="annotation-pages">
                                <strong>Pages:</strong>{' '}

                                {editingExistingPages === index ? (
                                  // Editing mode for existing annotation pages
                                  <div className="existing-pages-edit">
                                    <div className="current-pages">
                                      {tempExistingPages.map(pageNum => (
                                        <span key={pageNum} className="editable-page">
                                          {pageNum}
                                          <button
                                            className="remove-page"
                                            onClick={() => removePageFromExistingPages(pageNum)}
                                          >×</button>
                                        </span>
                                      ))}
                                      <input
                                        type="number"
                                        className="add-page-input"
                                        min="0"
                                        max={numPages - 1}
                                        placeholder="+"
                                        onKeyDown={(e) => {
                                          if (e.key === 'Enter') {
                                            const input = e.target as HTMLInputElement;
                                            const pageNum = parseInt(input.value);
                                            if (!isNaN(pageNum) && pageNum >= 0 && pageNum < numPages) {
                                              addPageToExistingPages(pageNum);
                                              input.value = '';
                                            }
                                          }
                                        }}
                                      />
                                    </div>
                                    <div className="edit-page-actions">
                                      <button
                                        className="save-pages-btn"
                                        onClick={() => saveEditedExistingPages(index, annotation)}
                                      >
                                        Save
                                      </button>
                                      <button
                                        className="cancel-pages-btn"
                                        onClick={cancelEditingExistingPages}
                                      >
                                        Cancel
                                      </button>
                                    </div>
                                  </div>
                                ) : (
                                  // Normal display mode
                                  <div className="page-display">
                                    {pageNumbers.length > 0 ? (
                                      <div className="page-buttons">
                                        {pageNumbers.map((pageNum, pageIndex) => (
                                          <button
                                            key={pageIndex}
                                            className="page-link-button"
                                            onClick={() => goToPage(pageNum + 1)}
                                            title={`Go to page ${pageNum + 1}`}
                                          >
                                            {pageNum + 1}
                                          </button>
                                        ))}

                                        <button
                                          className="edit-btn"
                                          onClick={() => startEditingExistingPages(index, annotation.page_ids)}
                                          title="Edit page numbers"
                                        >
                                          ✎
                                        </button>
                                      </div>
                                    ) : (
                                      <span>N/A</span>
                                    )}
                                  </div>
                                )}
                              </div>
                            </div>
                            {index < annotations.length - 1 && <hr className="annotation-separator" />}
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>
              </div>
            </>
          )}
        </section >
      </main >
      {showDeleteConfirmation && <DeleteConfirmationModal />}
    </div >
  );
}

export default App;
