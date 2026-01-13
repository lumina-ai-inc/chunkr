import { useState, useCallback } from "react";
import { Flex, Text, Button, Card } from "@radix-ui/themes";
import { uploadFile } from "../../services/uploadFileApi";
import { UploadForm } from "../../models/upload.model";
import { OcrStrategy, SegmentationStrategy, Pipeline, ErrorHandling } from "../../models/taskConfig.model";
import { toast } from "react-hot-toast";
import "./DealUpload.css";

interface DealUploadProps {
  onUploadSuccess?: (taskId: string) => void;
  onUploadStart?: () => void;
}

const SMALL_FILE_SIZE = 10 * 1024 * 1024;

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

async function encodeFile(file: File): Promise<string> {
  return file.size <= SMALL_FILE_SIZE
    ? fileToBase64(file)
    : fileToBase64(file); // Simplified for now
}

export default function DealUpload({ onUploadSuccess, onUploadStart }: DealUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<Record<string, boolean>>({});

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    setFiles((prev) => [...prev, ...selectedFiles]);
  };

  const handleFileRemove = (fileName: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== fileName));
  };

  const getDocumentType = (fileName: string): string => {
    const lower = fileName.toLowerCase();
    if (lower.includes("rent") || lower.includes("roll")) return "Rent Roll";
    if (lower.includes("p&l") || lower.includes("profit") || lower.includes("loss")) return "P&L";
    if (lower.includes("mortgage") || lower.includes("statement")) return "Mortgage Statement";
    return "Other";
  };

  const handleSubmit = useCallback(async () => {
    // #region agent log
    fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DealUpload.tsx:56',message:'handleSubmit called',data:{fileCount:files.length},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H1'})}).catch(()=>{});
    // #endregion
    if (files.length === 0) {
      toast.error("Please select at least one file");
      return;
    }

    setIsUploading(true);
    onUploadStart?.();

    const toastId = toast.loading("Uploading documents...");

    let successCount = 0;
    for (const file of files) {
      try {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DealUpload.tsx:70',message:'Before encodeFile',data:{fileName:file.name,fileSize:file.size},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H1-H2'})}).catch(()=>{});
        // #endregion
        const b64 = await encodeFile(file);
        const payload: UploadForm = {
          file: b64,
          file_name: file.name,
          ocr_strategy: OcrStrategy.All,
          segmentation_strategy: SegmentationStrategy.Page,  // Changed from LayoutAnalysis to Page (no segmentation service needed)
          high_resolution: true,
          pipeline: Pipeline.Orin as any,
          error_handling: ErrorHandling.Fail,
        };
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DealUpload.tsx:81',message:'Before uploadFile API call',data:{fileName:file.name,payloadSize:b64.length,segmentationStrategy:payload.segmentation_strategy,pipeline:payload.pipeline},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H2'})}).catch(()=>{});
        // #endregion
        
        const result = await uploadFile(payload);
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DealUpload.tsx:82',message:'uploadFile success',data:{taskId:result.task_id,fileName:file.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H2'})}).catch(()=>{});
        // #endregion
        setUploadProgress((prev) => ({ ...prev, [file.name]: true }));
        successCount++;
        
        if (successCount === 1) {
          onUploadSuccess?.(result.task_id);
        }
      } catch (err) {
        // #region agent log
        fetch('http://127.0.0.1:7242/ingest/8ba094c0-f913-4a1d-9d69-0a38a5483749',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'DealUpload.tsx:89',message:'Upload failed with error',data:{fileName:file.name,error:err instanceof Error?err.message:String(err),errorType:err?.constructor?.name},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'H2-H3'})}).catch(()=>{});
        // #endregion
        console.error(`Upload failed for ${file.name}:`, err);
        toast.error(`Failed to upload ${file.name}`);
      }
    }

    toast.dismiss(toastId);
    if (successCount === files.length) {
      toast.success(`Successfully uploaded ${successCount} file${successCount > 1 ? "s" : ""}`);
      setFiles([]);
    } else if (successCount > 0) {
      toast.success(`Uploaded ${successCount}/${files.length} files`);
      toast.error(`Failed to upload ${files.length - successCount} file(s)`);
    } else {
      toast.error("Failed to upload any files");
    }

    setIsUploading(false);
  }, [files, onUploadStart, onUploadSuccess]);

  return (
    <Flex direction="column" gap="24px" style={{ padding: "24px", width: "100%" }}>
      <Flex direction="column" gap="8px">
        <Text size="6" weight="bold" style={{ color: "#111" }}>
          Step 1: New Deal
        </Text>
        <Text size="3" style={{ color: "#666" }}>
          Upload property documents to begin underwriting
        </Text>
      </Flex>

      <Flex direction="column" gap="16px">
        <Card
          style={{
            padding: "24px",
            border: "2px dashed #e0e0e0",
            borderRadius: "8px",
            backgroundColor: "#f8f9fa",
            cursor: "pointer",
            transition: "all 0.2s ease",
          }}
          onMouseEnter={(e: React.MouseEvent<HTMLDivElement>) => {
            e.currentTarget.style.borderColor = "#545454";
            e.currentTarget.style.backgroundColor = "#f0f0f0";
          }}
          onMouseLeave={(e: React.MouseEvent<HTMLDivElement>) => {
            e.currentTarget.style.borderColor = "#e0e0e0";
            e.currentTarget.style.backgroundColor = "#f8f9fa";
          }}
        >
          <label
            htmlFor="file-upload"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "12px",
              cursor: "pointer",
            }}
          >
            <Text size="4" style={{ color: "#666" }}>
              📁 Click to upload or drag and drop
            </Text>
            <Text size="2" style={{ color: "#999" }}>
              Rent roll (PDF or CSV) • Trailing 12-month P&L • Mortgage statement (optional)
            </Text>
            <input
              id="file-upload"
              type="file"
              multiple
              accept=".pdf,.csv"
              onChange={handleFileChange}
              style={{ display: "none" }}
              disabled={isUploading}
            />
          </label>
        </Card>

        {files.length > 0 && (
          <Flex direction="column" gap="8px">
            <Text size="3" weight="medium" style={{ color: "#111" }}>
              Selected Files ({files.length})
            </Text>
            {files.map((file) => (
              <Card
                key={file.name}
                style={{
                  padding: "12px 16px",
                  border: "1px solid #e0e0e0",
                  borderRadius: "6px",
                  backgroundColor: "#fff",
                }}
              >
                <Flex align="center" justify="between">
                  <Flex direction="column" gap="4px" style={{ flex: 1 }}>
                    <Text size="3" weight="medium" style={{ color: "#111" }}>
                      {file.name}
                    </Text>
                    <Text size="2" style={{ color: "#666" }}>
                      {getDocumentType(file.name)} • {(file.size / 1024).toFixed(1)} KB
                    </Text>
                  </Flex>
                  {!isUploading && (
                    <Button
                      size="2"
                      onClick={() => handleFileRemove(file.name)}
                      style={{
                        backgroundColor: "#f44336",
                        color: "#fff",
                        cursor: "pointer",
                      }}
                    >
                      Remove
                    </Button>
                  )}
                  {uploadProgress[file.name] && (
                    <Text size="2" style={{ color: "#4CAF50" }}>
                      ✓ Uploaded
                    </Text>
                  )}
                </Flex>
              </Card>
            ))}
          </Flex>
        )}

        <Button
          onClick={handleSubmit}
          disabled={files.length === 0 || isUploading}
          style={{
            backgroundColor: files.length > 0 && !isUploading ? "#545454" : "#ccc",
            color: "#fff",
            padding: "12px 24px",
            cursor: files.length > 0 && !isUploading ? "pointer" : "not-allowed",
            alignSelf: "flex-start",
          }}
        >
          {isUploading ? "Uploading..." : "Upload & Extract Facts"}
        </Button>
      </Flex>
    </Flex>
  );
}

