import { Flex, Text } from "@radix-ui/themes";
import { useState, useCallback, useEffect } from "react";
import {
  UploadFormData,
  OcrStrategy,
  SegmentationStrategy,
  DEFAULT_UPLOAD_CONFIG,
  DEFAULT_SEGMENT_PROCESSING,
  DEFAULT_CHUNK_PROCESSING,
  Pipeline,
  ErrorHandling,
} from "../../models/taskConfig.model";
import "./UploadMain.css";
import Upload from "./Upload";
import {
  ToggleGroup,
  SegmentProcessingControls,
  ChunkProcessingControls,
  LlmProcessingControls,
} from "./ConfigControls";
import { TabControls } from "./TabControls";
import { uploadFile } from "../../services/uploadFileApi";
import { UploadForm } from "../../models/upload.model";
import { getEnvConfig, WhenEnabled } from "../../config/env.config";
import { env } from "../../config/env";
import { toast } from "react-hot-toast";

const DOCS_URL = env.docsUrl || "https://docs.chunkr.ai";

// threshold under which we encode on main thread (10 MB)
const SMALL_FILE_SIZE = 10 * 1024 * 1024;

// Base64 on main thread
function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = reject;
    reader.onload = () => {
      const dataUrl = reader.result as string;
      resolve(dataUrl.split(",", 2)[1]);
    };
    reader.readAsDataURL(file);
  });
}

// Offload Base64 to Web Worker for larger files
function fileToBase64InWorker(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(
      new URL("../../workers/base64.worker.ts", import.meta.url),
      { type: "module" }
    );
    worker.onerror = reject;
    worker.onmessage = (e) => {
      resolve(e.data as string);
      worker.terminate();
    };
    worker.postMessage(file);
  });
}

// Chooses main‐thread vs worker based on file size
async function encodeFile(file: File): Promise<string> {
  return file.size <= SMALL_FILE_SIZE
    ? fileToBase64(file)
    : fileToBase64InWorker(file);
}

interface UploadMainProps {
  isAuthenticated: boolean;
  onUploadSuccess?: () => void;
  onUploadStart?: () => void;
}

export default function UploadMain({
  isAuthenticated,
  onUploadSuccess,
  onUploadStart,
}: UploadMainProps) {
  const { features } = getEnvConfig();
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<UploadFormData>(DEFAULT_UPLOAD_CONFIG);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedSegmentType, setSelectedSegmentType] = useState<
    keyof typeof DEFAULT_SEGMENT_PROCESSING
  >("Text");
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const handleFileChange = (ufs: File[]) => {
    setFiles((prev) => [...prev, ...ufs]);
    setUploadError(null);
  };

  const handleFileRemove = (fileName: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== fileName));
    setUploadError(null);
  };

  const getEffectiveSegmentProcessing = (current: UploadFormData) => {
    if (current.segmentation_strategy === SegmentationStrategy.Page) {
      return {
        ...DEFAULT_SEGMENT_PROCESSING,
        Page:
          current.segment_processing?.Page || DEFAULT_SEGMENT_PROCESSING.Page,
      };
    }
    return current.segment_processing || DEFAULT_SEGMENT_PROCESSING;
  };

  // Compute available segment types
  const segmentTypes = config.segmentation_strategy === SegmentationStrategy.Page
    ? ["Page"]
    : Object.keys(config.segment_processing || DEFAULT_SEGMENT_PROCESSING).filter(k => k !== "Page");

  // Keep selectedSegmentType in sync with available types
  useEffect(() => {
    if (!selectedSegmentType || !segmentTypes.includes(selectedSegmentType)) {
      const defaultType = segmentTypes.length > 0 ? segmentTypes[0] : "Text";
      setSelectedSegmentType(defaultType as keyof typeof DEFAULT_SEGMENT_PROCESSING);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [segmentTypes.join(","), selectedSegmentType]);

  const handleSubmit = useCallback(async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    onUploadStart?.();

    // show a single loading toast
    const toastId = toast.loading("Uploading…");

    let successCount = 0;
    for (const file of files) {
      try {
        const b64 = await encodeFile(file);
        const payload: UploadForm = {
          file: b64,
          file_name: file.name,
          chunk_processing: config.chunk_processing,
          high_resolution: config.high_resolution,
          ocr_strategy: config.ocr_strategy,
          segment_processing: getEffectiveSegmentProcessing(config),
          segmentation_strategy: config.segmentation_strategy,
          pipeline: config.pipeline,
          llm_processing: config.llm_processing,
          error_handling: config.error_handling,
        };
        await uploadFile(payload);
        successCount++;
      } catch (err) {
        console.error(`Upload failed for ${file.name}:`, err);
      }
    }

    // replace loading toast with summary
    toast.dismiss(toastId);
    if (successCount === files.length) {
      toast.success(`Uploaded ${successCount}/${files.length} files`);
    } else if (successCount > 0) {
      toast.success(`Uploaded ${successCount}/${files.length} files`);
      toast.error(`Failed to upload ${files.length - successCount} file(s)`);
    } else {
      toast.error("Failed to upload any files");
    }

    // reset state and notify parent
    setFiles([]);
    setConfig(DEFAULT_UPLOAD_CONFIG);
    onUploadSuccess?.();
    setIsUploading(false);
  }, [files, config, onUploadStart, onUploadSuccess]);

  const getButtonText = () => {
    if (isUploading) {
      return "Processing…";
    }
    if (files.length === 1) {
      return `Process Document (${files.length})`;
    }
    if (files.length > 1) {
      return `Process Documents (${files.length})`;
    }
    // Default text when no files are selected (button is disabled)
    return "Process Document";
  };

  return (
    <div className="upload-form-container">
      <div>
        <section
          className={`config-section ${!isAuthenticated ? "disabled" : ""}`}
        >
          <TabControls tabLabels={["Verification Vault"]} title="New Deal">
            {/* Extract Tab */}
            <div>
              <div className="upload-section">
                <Upload
                  onFileUpload={handleFileChange}
                  onFileRemove={handleFileRemove}
                  files={files}
                  isAuthenticated={isAuthenticated}
                  isUploading={isUploading}
                />
                {uploadError && (
                  <Text size="2" style={{ color: "red", marginTop: "8px" }}>
                    {uploadError}
                  </Text>
                )}
              </div>
              {/* Main Settings - Always Visible */}
              {/* Row 1: Image Analysis and Layout Processing */}
              <div className="config-row">
                <ToggleGroup
                  docsUrl={`${DOCS_URL}/docs/features/ocr`}
                  label={
                    <Flex gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <circle
                          cx="12"
                          cy="12"
                          r="9.25"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <circle
                          cx="12"
                          cy="12"
                          r="5.25"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                      </svg>
                      <span>Image Analysis</span>
                    </Flex>
                  }
                  value={config.ocr_strategy || OcrStrategy.Auto}
                  onChange={(value) =>
                    setConfig({ ...config, ocr_strategy: value as OcrStrategy })
                  }
                  options={[
                    { label: "Auto Select", value: OcrStrategy.Auto },
                    { label: "All Pages", value: OcrStrategy.All },
                  ]}
                />

                {/* Layout Processing - Main Setting (Dropdown Only) */}
                <div className="config-card">
                  <div className="config-card-header">
                    <Flex direction="row" gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M21.25 12C21.25 17.1086 17.1086 21.25 12 21.25M2.75 12C2.75 6.89137 6.89137 2.75 12 2.75"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M17.25 12C17.25 9.10051 14.8995 6.75 12 6.75M12 17.25C9.10051 17.25 6.75 14.8995 6.75 12"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                      </svg>
                      <Text size="3" weight="bold" className="white">
                        Layout Processing
                      </Text>
                    </Flex>
                    <Flex
                      onClick={() =>
                        DOCS_URL &&
                        window.open(
                          `${DOCS_URL}/docs/features/segment-processing`,
                          "_blank"
                        )
                      }
                      direction="row"
                      gap="1"
                      align="center"
                      justify="end"
                      className="docs-text"
                    >
                      <Text size="1" weight="bold" style={{ color: "#545454" }}>
                        Help
                      </Text>
                      <svg
                        width="12px"
                        height="12px"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M9.8374 14.1625L14.1625 9.8374"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                      </svg>
                    </Flex>
                  </div>

                  {/* Functional dropdown for segment type selection */}
                  <div className="model-selector" style={{ marginTop: "16px" }}>
                    <select
                      className="model-selector-select"
                      value={selectedSegmentType}
                      onChange={(e) => setSelectedSegmentType(e.target.value as keyof typeof DEFAULT_SEGMENT_PROCESSING)}
                      style={{ 
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        border: "1px solid #545454",
                        backgroundColor: "white",
                        color: "#111",
                        fontSize: "14px",
                        fontWeight: "500",
                        cursor: "pointer"
                      }}
                    >
                      <option value="Text">Text</option>
                      <option value="Table">Table</option>
                      <option value="Formula">Formula</option>
                      <option value="Caption">Caption</option>
                      <option value="Footnote">Footnote</option>
                      <option value="ListItem">List Item</option>
                      <option value="Page">Page</option>
                      <option value="PageFooter">Page Footer</option>
                      <option value="PageHeader">Page Header</option>
                      <option value="Picture">Picture</option>
                      <option value="SectionHeader">Section Header</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Row 2: Tokenizer and Primary Model */}
              <div className="config-row" style={{ marginTop: "20px" }}>

                {/* Tokenizer - Main Setting (Dropdown Only) */}
                <div className="config-card">
                  <div className="config-card-header">
                    <Flex direction="row" gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <g clipPath="url(#clip0_305_31854)">
                          <path
                            d="M9.25 16C9.25 14.2051 7.79493 12.75 6 12.75C4.20507 12.75 2.75 14.2051 2.75 16C2.75 17.7949 4.20507 19.25 6 19.25C7.79493 19.25 9.25 17.7949 9.25 16Z"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M16.8699 4.75L8.85994 17.55L8.68994 17.82"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M14.75 16C14.75 17.7949 16.2051 19.25 18 19.25C19.7949 19.25 21.25 17.7949 21.25 16C21.25 14.2051 19.7949 12.75 18 12.75C16.2051 12.75 14.75 14.2051 14.75 16Z"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          <path
                            d="M15.3099 17.82L15.1399 17.55L7.12988 4.75"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </g>
                        <defs>
                          <clipPath id="clip0_305_31854">
                            <rect width="24" height="24" fill="white" />
                          </clipPath>
                        </defs>
                      </svg>
                      <Text size="3" weight="bold" className="white">
                        Chunking Tokenizer
                      </Text>
                    </Flex>
                    <Flex
                      onClick={() => DOCS_URL && window.open(`${DOCS_URL}/docs/features/chunking`, "_blank")}
                      direction="row"
                      gap="1"
                      align="center"
                      justify="end"
                      className="docs-text"
                    >
                      <Text size="1" weight="bold" style={{ color: "#545454" }}>
                        Help
                      </Text>
                      <svg
                        width="12px"
                        height="12px"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M9.8374 14.1625L14.1625 9.8374"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                      </svg>
                    </Flex>
                  </div>

                  {/* Functional tokenizer dropdown */}
                  <div className="model-selector" style={{ marginTop: "16px" }}>
                    <select
                      className="model-selector-select"
                      value={config.chunk_processing?.tokenizer?.Enum || "Word"}
                      onChange={(e) => setConfig({ 
                        ...config, 
                        chunk_processing: { 
                          ...config.chunk_processing, 
                          tokenizer: { Enum: e.target.value },
                          target_length: config.chunk_processing?.target_length || 512
                        }
                      })}
                      style={{ 
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        border: "1px solid #545454",
                        backgroundColor: "white",
                        color: "#111",
                        fontSize: "14px",
                        fontWeight: "500",
                        cursor: "pointer"
                      }}
                    >
                      <option value="Word">Word</option>
                      <option value="Cl100kBase">CL100K Base</option>
                      <option value="XlmRobertaBase">XLM-Roberta Base</option>
                      <option value="BertBaseUncased">BERT Base Uncased</option>
                    </select>
                  </div>
                </div>

                {/* Primary Model - Main Setting (Dropdown Only) */}
                <div className="config-card">
                  <div className="config-card-header">
                    <Flex direction="row" gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M19 3V7M17 5H21M19 17V21M17 19H21M10 5L8.53001 8.72721C8.3421 9.20367 8.24814 9.4419 8.10427 9.64278C7.97675 9.82084 7.82084 9.97675 7.64278 10.1043C7.4419 10.2481 7.20367 10.3421 6.72721 10.53L3 12L6.72721 13.47C7.20367 13.6579 7.4419 13.7519 7.64278 13.8957C7.82084 14.0233 7.97675 14.1792 8.10427 14.3572C8.24814 14.5581 8.3421 14.7963 8.53001 15.2728L10 19L11.47 15.2728C11.6579 14.7963 11.7519 14.5581 11.8957 14.3572C12.0233 14.1792 12.1792 14.0233 12.3572 13.8957C12.5581 13.7519 12.7963 13.6579 13.2728 13.47L17 12L13.2728 10.53C12.7963 10.3421 12.5581 10.2481 12.3572 10.1043C12.1792 9.97675 12.0233 9.82084 11.8957 9.64278C11.7519 9.4419 11.6579 9.20367 11.47 8.72721L10 5Z"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <Text size="3" weight="bold" className="white">
                        LLM Model
                      </Text>
                    </Flex>
                    <Flex
                      onClick={() => DOCS_URL && window.open(`${DOCS_URL}/docs/features/llm-processing`, "_blank")}
                      direction="row"
                      gap="1"
                      align="center"
                      justify="end"
                      className="docs-text"
                    >
                      <Text size="1" weight="bold" style={{ color: "#545454" }}>
                        Help
                      </Text>
                      <svg
                        width="12px"
                        height="12px"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M14.1625 18.4876L13.4417 19.2084C11.053 21.5971 7.18019 21.5971 4.79151 19.2084C2.40283 16.8198 2.40283 12.9469 4.79151 10.5583L5.51236 9.8374"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M9.8374 14.1625L14.1625 9.8374"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                        <path
                          d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                        />
                      </svg>
                    </Flex>
                  </div>

                  {/* Functional model dropdown */}
                  <div className="model-selector" style={{ marginTop: "16px" }}>
                    <select
                      className="model-selector-select"
                      value={config.llm_processing?.model_id || ""}
                      onChange={(e) => setConfig({ 
                        ...config, 
                        llm_processing: { 
                          ...config.llm_processing, 
                          model_id: e.target.value
                        }
                      })}
                      style={{ 
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        border: "1px solid #545454",
                        backgroundColor: "white",
                        color: "#111",
                        fontSize: "14px",
                        fontWeight: "500",
                        cursor: "pointer"
                      }}
                    >
                      <option value="">Default Model</option>
                      <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                      <option value="gpt-4">GPT-4</option>
                      <option value="claude-3-sonnet">Claude 3 Sonnet</option>
                      <option value="claude-3-haiku">Claude 3 Haiku</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Advanced Settings Toggle */}
              <div style={{ marginTop: "30px", textAlign: "center" }}>
                <Text 
                  size="2" 
                  weight="medium" 
                  style={{ 
                    color: "#06b6d4", 
                    cursor: "pointer", 
                    textDecoration: showAdvancedSettings ? "none" : "underline"
                  }}
                  onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                >
                  {showAdvancedSettings ? "Hide Advanced Settings" : "Advanced Settings"}
                  <span style={{ marginLeft: "8px" }}>
                    {showAdvancedSettings ? "▲" : "▼"}
                  </span>
                </Text>
              </div>

              {/* Advanced Settings - Collapsible */}
              {showAdvancedSettings && (
                <div style={{ marginTop: "20px" }}>
                  <div className="config-grid">
                    {features.pipeline && (
                      <ToggleGroup
                        docsUrl={`${DOCS_URL}/docs/features/pipeline`}
                        label={
                          <Flex gap="2" align="center">
                            <svg
                              width="20px"
                              height="20px"
                              viewBox="0 0 16 16"
                              xmlns="http://www.w3.org/2000/svg"
                              fill="none"
                            >
                              <path
                                fill="#545454"
                                fillRule="evenodd"
                                d="M2.75 2.5A1.75 1.75 0 001 4.25v1C1 6.216 1.784 7 2.75 7h1a1.75 1.75 0 001.732-1.5H6.5a.75.75 0 01.75.75v3.5A2.25 2.25 0 009.5 12h1.018c.121.848.85 1.5 1.732 1.5h1A1.75 1.75 0 0015 11.75v-1A1.75 1.75 0 0013.25 9h-1a1.75 1.75 0 00-1.732 1.5H9.5a.75.75 0 01-.75-.75v-3.5A2.25 2.25 0 006.5 4H5.482A1.75 1.75 0 003.75 2.5h-1zM2.5 4.25A.25.25 0 012.75 4h1a.25.25 0 01.25.25v1a.25.25 0 01-.25.25h-1a.25.25 0 01-.25-.25v-1zm9.75 6.25a.25.25 0 00-.25.25v1c0 .138.112.25.25.25h1a.25.25 0 00.25-.25v-1a.25.25 0 00-.25-.25h-1z"
                                clipRule="evenodd"
                              />
                            </svg>
                            <span>OCR Technology</span>
                          </Flex>
                        }
                        value={config.pipeline || Pipeline.Azure}
                        onChange={(value) =>
                          setConfig({
                            ...config,
                            pipeline: (features.pipeline
                              ? value === Pipeline.Chunkr
                                ? Pipeline.Chunkr
                                : Pipeline.Azure
                              : undefined) as WhenEnabled<"pipeline", Pipeline>,
                          })
                        }
                        options={[
                          { label: "Azure", value: Pipeline.Azure },
                          { label: "Chunkr", value: Pipeline.Chunkr },
                        ]}
                      />
                    )}
                    
                    <ToggleGroup
                      docsUrl={`${DOCS_URL}/api-references/task/create-task#body-error-handling`}
                      label={
                        <Flex gap="2" align="center">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path
                              d="M12 13.75V9.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                            <circle cx="12" cy="17" r="1" fill="#545454" />
                            <path
                              d="M4.39877 20.25C3.64805 20.25 3.16502 19.4536 3.51196 18.7879L11.1132 4.20171C11.4869 3.48456 12.5131 3.48456 12.8868 4.20171L20.488 18.7879C20.835 19.4536 20.352 20.25 19.6012 20.25H4.39877Z"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </svg>
                          <span>Error Handling</span>
                        </Flex>
                      }
                      value={config.error_handling || ErrorHandling.Fail}
                      onChange={(value) =>
                        setConfig({ ...config, error_handling: value as ErrorHandling })
                      }
                      options={[
                        { label: "Fail & Fix", value: ErrorHandling.Fail },
                        { label: "Skip & Continue", value: ErrorHandling.Continue },
                      ]}
                    />

                    <ToggleGroup
                      docsUrl={`${DOCS_URL}/api-references/task/create-task#body-high-resolution`}
                      label={
                        <Flex gap="2" align="center">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <circle
                              cx="12"
                              cy="5.5"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="5.5"
                              cy="5.5"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="18.5"
                              cy="5.5"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="12"
                              cy="18.5"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="5.5"
                              cy="18.5"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="18.5"
                              cy="18.5"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="12"
                              cy="12"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="5.5"
                              cy="12"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <circle
                              cx="18.5"
                              cy="12"
                              r="1.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                          </svg>
                          <span>High Resolution</span>
                        </Flex>
                      }
                      value={config.high_resolution ? "ON" : "OFF"}
                      onChange={(value) =>
                        setConfig({ ...config, high_resolution: value === "ON" })
                      }
                      options={[
                        { label: "On (Complex)", value: "ON" },
                        { label: "Off (Save Tokens)", value: "OFF" },
                      ]}
                    />

                    <ToggleGroup
                      docsUrl={`${DOCS_URL}/docs/features/layout-analysis/what`}
                      label={
                        <Flex gap="2" align="center">
                          <svg
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <g clipPath="url(#clip0_305_27919)">
                              <path
                                d="M7.75 20.25V8.75C7.75 8.2 8.2 7.75 8.75 7.75H20.25"
                                stroke="#545454"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              <path
                                d="M16.25 3.75V15.25C16.25 15.8 15.8 16.25 15.25 16.25H3.75"
                                stroke="#545454"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                            </g>
                            <defs>
                              <clipPath id="clip0_305_27919">
                                <rect width="24" height="24" fill="white" />
                              </clipPath>
                            </defs>
                          </svg>
                          <span>Segment Strategy</span>
                        </Flex>
                      }
                      value={
                        config.segmentation_strategy ||
                        SegmentationStrategy.LayoutAnalysis
                      }
                      onChange={(value) =>
                        setConfig({
                          ...config,
                          segmentation_strategy: value as SegmentationStrategy,
                        })
                      }
                      options={[
                        {
                          label: "By Layout",
                          value: SegmentationStrategy.LayoutAnalysis,
                        },
                        { label: "Per Page", value: SegmentationStrategy.Page },
                      ]}
                    />
                  </div>

                  {/* Full Layout Processing Controls */}
                  <div className="config-card" style={{ marginTop: "20px" }}>
                    <div className="config-card-header">
                      <Flex direction="row" gap="2" align="center">
                        <svg
                          width="24"
                          height="24"
                          viewBox="0 0 24 24"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M21.25 12C21.25 17.1086 17.1086 21.25 12 21.25M2.75 12C2.75 6.89137 6.89137 2.75 12 2.75"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                          />
                          <path
                            d="M17.25 12C17.25 9.10051 14.8995 6.75 12 6.75M12 17.25C9.10051 17.25 6.75 14.8995 6.75 12"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                          />
                        </svg>
                        <Text size="3" weight="bold" className="white">
                          Layout Processing (Full Controls)
                        </Text>
                      </Flex>
                    </div>

                    <SegmentProcessingControls
                      value={
                        config.segment_processing && 
                        Object.keys(config.segment_processing).length > 0 
                          ? config.segment_processing 
                          : DEFAULT_SEGMENT_PROCESSING
                      }
                      onChange={(value) =>
                        setConfig({
                          ...config,
                          segment_processing: value,
                        })
                      }
                      showOnlyPage={
                        config.segmentation_strategy === SegmentationStrategy.Page
                      }
                      selectedType={selectedSegmentType}
                      onTypeChange={setSelectedSegmentType}
                    />
                  </div>

                  {/* Full Chunk Processing Controls */}
                  <div style={{ marginTop: "20px" }}>
                    <ChunkProcessingControls
                      docsUrl={`${DOCS_URL}/docs/features/chunking`}
                      value={
                        config.chunk_processing && 
                        Object.keys(config.chunk_processing).length > 0
                          ? config.chunk_processing 
                          : DEFAULT_CHUNK_PROCESSING
                      }
                      onChange={(value) =>
                        setConfig({
                          ...config,
                          chunk_processing: value,
                        })
                      }
                      hideTokenizer={true}
                    />
                  </div>

                  {/* Full LLM Processing Controls */}
                  <div style={{ marginTop: "20px" }}>
                    <LlmProcessingControls
                      value={config.llm_processing!}
                      onChange={(llm) => setConfig({ ...config, llm_processing: llm })}
                      docsUrl={`${DOCS_URL}/docs/features/llm-processing`}
                    />
                  </div>
                </div>
              )}

            </div>

            {/* Finance Records Tab */}
            <div>
              {/* Row 1: Bank Transactions and Credit Data */}
              <div className="config-row">
                {/* Bank Transactions */}
                <div className="config-card">
                  <div className="config-card-header">
                    <Flex direction="row" gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M2 8V6C2 4.89543 2.89543 4 4 4H20C21.1046 4 22 4.89543 22 6V8M2 8V18C2 19.1046 2.89543 20 4 20H20C21.1046 20 22 19.1046 22 18V8M2 8H22"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M6 12H10"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <Text size="3" weight="bold" className="white">
                        Bank Connections
                      </Text>
                    </Flex>
                  </div>
                  <div className="model-selector" style={{ marginTop: "16px" }}>
                    <select
                      className="model-selector-select"
                      style={{ 
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        border: "1px solid #545454",
                        backgroundColor: "white",
                        color: "#111",
                        fontSize: "14px",
                        fontWeight: "500",
                        cursor: "pointer"
                      }}
                    >
                      <option value="">Select Provider</option>
                      <option value="plaid">Plaid</option>
                      <option value="max">Max</option>
                    </select>
                  </div>
                </div>

                {/* Credit Data */}
                <div className="config-card">
                  <div className="config-card-header">
                    <Flex direction="row" gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M3 10H21C21.5523 10 22 10.4477 22 11V19C22 20.1046 21.1046 21 20 21H4C2.89543 21 2 20.1046 2 19V11C2 10.4477 2.44772 10 3 10Z"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M7 15H13"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M7 7H17C18.1046 7 19 7.89543 19 9V10H5V9C5 7.89543 5.89543 7 7 7Z"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <Text size="3" weight="bold" className="white">
                        Credit Data
                      </Text>
                    </Flex>
                  </div>
                  <div className="model-selector" style={{ marginTop: "16px" }}>
                    <select
                      className="model-selector-select"
                      style={{ 
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        border: "1px solid #545454",
                        backgroundColor: "white",
                        color: "#111",
                        fontSize: "14px",
                        fontWeight: "500",
                        cursor: "pointer"
                      }}
                    >
                      <option value="">Select Bureau</option>
                      <option value="experian">Experian</option>
                      <option value="equifax">Equifax</option>
                      <option value="transunion">TransUnion</option>
                    </select>
                  </div>
                </div>
              </div>

              {/* Row 2: Income/Payroll and Investments */}
              <div className="config-row" style={{ marginTop: "20px" }}>
                {/* Income/Payroll */}
                <div className="config-card">
                  <div className="config-card-header">
                    <Flex direction="row" gap="2" align="center">
                      <svg
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                      >
                        <path
                          d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M12 16V22"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M9 19H15"
                          stroke="#545454"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                      <Text size="3" weight="bold" className="white">
                        Cashflow Analysis
                      </Text>
                    </Flex>
                  </div>
                  <div className="model-selector" style={{ marginTop: "16px" }}>
                    <select
                      className="model-selector-select"
                      style={{ 
                        width: "100%",
                        padding: "12px 16px",
                        borderRadius: "8px",
                        border: "1px solid #545454",
                        backgroundColor: "white",
                        color: "#111",
                        fontSize: "14px",
                        fontWeight: "500",
                        cursor: "pointer"
                      }}
                    >
                      <option value="">Select Provider</option>
                      <option value="pinwheel">Pinwheel</option>
                    </select>
                  </div>
                </div>
              </div>
            </div>


          </TabControls>
        </section>

        <section
          className={`submit-section ${!isAuthenticated ? "disabled" : ""}`}
        >
          <button
            className="submit-button"
            onClick={handleSubmit}
            disabled={files.length === 0 || !isAuthenticated || isUploading}
          >
            <Text size="3" weight="bold">
              {getButtonText()}
            </Text>
          </button>
        </section>
      </div>
    </div>
  );
}
