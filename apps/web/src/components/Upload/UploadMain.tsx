import { Flex, Text } from "@radix-ui/themes";
import { useState, useCallback } from "react";
import {
  UploadFormData,
  OcrStrategy,
  SegmentationStrategy,
  DEFAULT_UPLOAD_CONFIG,
  DEFAULT_SEGMENT_PROCESSING,
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
import { uploadFile } from "../../services/uploadFileApi";
import { UploadForm } from "../../models/upload.model";
import { getEnvConfig, WhenEnabled } from "../../config/env.config";
import { toast } from "react-hot-toast";

const DOCS_URL = import.meta.env.VITE_DOCS_URL || "https://docs.chunkr.ai";

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
    <div className="upload-main-container">
      <div className="upload-form-container">
        <div className="upload-scrollable-content">
          <section className="upload-section">
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
          </section>

          <section
            className={`config-section ${!isAuthenticated ? "disabled" : ""}`}
          >
            <div className="config-grid" style={{ marginTop: "32px" }}>
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
                        fill="#FFF"
                        fill-rule="evenodd"
                        d="M2.75 2.5A1.75 1.75 0 001 4.25v1C1 6.216 1.784 7 2.75 7h1a1.75 1.75 0 001.732-1.5H6.5a.75.75 0 01.75.75v3.5A2.25 2.25 0 009.5 12h1.018c.121.848.85 1.5 1.732 1.5h1A1.75 1.75 0 0015 11.75v-1A1.75 1.75 0 0013.25 9h-1a1.75 1.75 0 00-1.732 1.5H9.5a.75.75 0 01-.75-.75v-3.5A2.25 2.25 0 006.5 4H5.482A1.75 1.75 0 003.75 2.5h-1zM2.5 4.25A.25.25 0 012.75 4h1a.25.25 0 01.25.25v1a.25.25 0 01-.25.25h-1a.25.25 0 01-.25-.25v-1zm9.75 6.25a.25.25 0 00-.25.25v1c0 .138.112.25.25.25h1a.25.25 0 00.25-.25v-1a.25.25 0 00-.25-.25h-1z"
                        clip-rule="evenodd"
                      />
                    </svg>
                    <span>Pipeline</span>
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
                          stroke="#FFF"
                          strokeWidth="1.5"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                        <path
                          d="M16.25 3.75V15.25C16.25 15.8 15.8 16.25 15.25 16.25H3.75"
                          stroke="#FFF"
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
                    <span>Segmentation Strategy</span>
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
                    label: "Layout Analysis",
                    value: SegmentationStrategy.LayoutAnalysis,
                  },
                  { label: "Page", value: SegmentationStrategy.Page },
                ]}
              />
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
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="12"
                        cy="12"
                        r="5.25"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                      />
                    </svg>

                    <span>OCR Strategy</span>
                  </Flex>
                }
                value={config.ocr_strategy || OcrStrategy.Auto}
                onChange={(value) =>
                  setConfig({ ...config, ocr_strategy: value as OcrStrategy })
                }
                options={[
                  { label: "Auto", value: OcrStrategy.Auto },
                  { label: "All", value: OcrStrategy.All },
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
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="5.5"
                        cy="5.5"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="18.5"
                        cy="5.5"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="12"
                        cy="18.5"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="5.5"
                        cy="18.5"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="18.5"
                        cy="18.5"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="12"
                        cy="12"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="5.5"
                        cy="12"
                        r="1.75"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeMiterlimit="10"
                        strokeLinecap="round"
                      />
                      <circle
                        cx="18.5"
                        cy="12"
                        r="1.75"
                        stroke="#FFF"
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
                  { label: "ON", value: "ON" },
                  { label: "OFF", value: "OFF" },
                ]}
              />

              <ToggleGroup
                docsUrl={`${DOCS_URL}/api-references/task/create-task#body-error-handling`} // Updated docs URL
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
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <circle cx="12" cy="17" r="1" fill="#FFF" />
                      <path
                        d="M4.39877 20.25C3.64805 20.25 3.16502 19.4536 3.51196 18.7879L11.1132 4.20171C11.4869 3.48456 12.5131 3.48456 12.8868 4.20171L20.488 18.7879C20.835 19.4536 20.352 20.25 19.6012 20.25H4.39877Z"
                        stroke="#FFF"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                    <span>Error Handling</span>
                  </Flex>
                }
                // Use config.error_handling or default to ErrorHandling.Fail
                value={config.error_handling || ErrorHandling.Fail}
                onChange={(value) =>
                  // Update config.error_handling
                  setConfig({
                    ...config,
                    error_handling: value as ErrorHandling,
                  })
                }
                // Updated options for ErrorHandling enum
                options={[
                  { label: "Fail", value: ErrorHandling.Fail },
                  { label: "Continue", value: ErrorHandling.Continue },
                ]}
              />
            </div>
          </section>

          <div
            className={`config-section ${!isAuthenticated ? "disabled" : ""}`}
          >
            <SegmentProcessingControls
              value={config.segment_processing || {}}
              onChange={(value) =>
                setConfig({
                  ...config,
                  segment_processing: value,
                })
              }
              showOnlyPage={
                config.segmentation_strategy === SegmentationStrategy.Page
              }
              docsUrl={`${DOCS_URL}/docs/features/segment-processing`}
            />
          </div>

          <ChunkProcessingControls
            docsUrl={`${DOCS_URL}/docs/features/chunking`}
            value={config.chunk_processing || { target_length: 512 }}
            onChange={(value) =>
              setConfig({
                ...config,
                chunk_processing: value,
              })
            }
          />

          <div style={{ marginTop: "40px" }}>
            <LlmProcessingControls
              value={config.llm_processing!}
              onChange={(llm) => setConfig({ ...config, llm_processing: llm })}
              docsUrl={`${DOCS_URL}/docs/features/llm-processing`}
            />
          </div>
        </div>
      </div>

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
  );
}
