import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import {
  UploadFormData,
  OcrStrategy,
  SegmentationStrategy,
  DEFAULT_UPLOAD_CONFIG,
  DEFAULT_SEGMENT_PROCESSING,
} from "../../models/newTask.model";
import "./UploadMain.css";
import Upload from "./Upload";
import {
  ToggleGroup,
  NumberInput,
  SegmentProcessingControls,
  JsonSchemaControls,
} from "./ConfigControls";
import { uploadFile } from "../../services/uploadFileApi";
import { UploadForm } from "../../models/upload.model";

interface UploadMainProps {
  onSubmit: (config: UploadFormData) => void;
  isAuthenticated: boolean;
}

export default function UploadMain({ isAuthenticated }: UploadMainProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<Partial<UploadFormData>>(
    DEFAULT_UPLOAD_CONFIG
  );
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const handleFileChange = (uploadedFiles: File[]) => {
    setFiles((prev) => [...prev, ...uploadedFiles]);
    setUploadError(null);
  };

  const handleFileRemove = (fileName: string) => {
    setFiles((prev) => prev.filter((file) => file.name !== fileName));
    setUploadError(null);
  };

  const handleSubmit = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      for (const file of files) {
        const uploadPayload: UploadForm = {
          file,
          chunk_processing: config.chunk_processing,
          high_resolution: config.high_resolution,
          json_schema: config.json_schema,
          ocr_strategy: config.ocr_strategy,
          segment_processing: config.segment_processing,
          segmentation_strategy: config.segmentation_strategy,
        };

        console.log("Upload payload:", uploadPayload);

        const response = await uploadFile(uploadPayload);
        console.log("Upload successful:", response);
      }

      setFiles([]);
      setConfig(DEFAULT_UPLOAD_CONFIG);
    } catch (error) {
      console.error("Upload failed:", error);
      setUploadError(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="upload-form-container">
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

      <div>
        <section
          className={`config-section ${!isAuthenticated ? "disabled" : ""}`}
        >
          <div className="config-grid">
            <ToggleGroup
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
                      stroke-width="1.5"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="12"
                      cy="12"
                      r="5.25"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-linecap="round"
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

            <NumberInput
              label={
                <Flex gap="2" align="center">
                  <svg
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <g clip-path="url(#clip0_305_31854)">
                      <path
                        d="M9.25 16C9.25 14.2051 7.79493 12.75 6 12.75C4.20507 12.75 2.75 14.2051 2.75 16C2.75 17.7949 4.20507 19.25 6 19.25C7.79493 19.25 9.25 17.7949 9.25 16Z"
                        stroke="#FFF"
                        stroke-width="1.5"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                      <path
                        d="M16.8699 4.75L8.85994 17.55L8.68994 17.82"
                        stroke="#FFF"
                        stroke-width="1.5"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                      <path
                        d="M14.75 16C14.75 17.7949 16.2051 19.25 18 19.25C19.7949 19.25 21.25 17.7949 21.25 16C21.25 14.2051 19.7949 12.75 18 12.75C16.2051 12.75 14.75 14.2051 14.75 16Z"
                        stroke="#FFF"
                        stroke-width="1.5"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                      <path
                        d="M15.3099 17.82L15.1399 17.55L7.12988 4.75"
                        stroke="#FFF"
                        stroke-width="1.5"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                    </g>
                    <defs>
                      <clipPath id="clip0_305_31854">
                        <rect width="24" height="24" fill="white" />
                      </clipPath>
                    </defs>
                  </svg>
                  <span>Target Chunk Length</span>
                </Flex>
              }
              value={config.chunk_processing?.target_length || 512}
              onChange={(value) =>
                setConfig({
                  ...config,
                  chunk_processing: {
                    ...config.chunk_processing,
                    target_length: value,
                  },
                })
              }
              min={0}
            />

            <ToggleGroup
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
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="5.5"
                      cy="5.5"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="18.5"
                      cy="5.5"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="12"
                      cy="18.5"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="5.5"
                      cy="18.5"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="18.5"
                      cy="18.5"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="12"
                      cy="12"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="5.5"
                      cy="12"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
                    />
                    <circle
                      cx="18.5"
                      cy="12"
                      r="1.75"
                      stroke="#FFF"
                      stroke-width="1.5"
                      stroke-miterlimit="10"
                      stroke-linecap="round"
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
          </div>

          <div className="config-card" style={{ marginTop: "32px" }}>
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
                    stroke="#FFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  />
                  <path
                    d="M17.25 12C17.25 9.10051 14.8995 6.75 12 6.75M12 17.25C9.10051 17.25 6.75 14.8995 6.75 12"
                    stroke="#FFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  />
                </svg>
                <Text size="3" weight="bold" className="white">
                  Segment Processing
                </Text>
              </Flex>
              <Flex
                direction="row"
                gap="1"
                align="center"
                justify="end"
                className="docs-text"
              >
                <Text size="1" weight="medium" className="white ">
                  Docs
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
                    stroke="#FFFFFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  />
                  <path
                    d="M9.8374 14.1625L14.1625 9.8374"
                    stroke="#FFFFFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  />
                  <path
                    d="M9.8374 5.51236L10.5583 4.79151C12.9469 2.40283 16.8198 2.40283 19.2084 4.79151M18.4876 14.1625L19.2084 13.4417C20.4324 12.2177 21.0292 10.604 20.9988 9"
                    stroke="#FFFFFF"
                    stroke-width="1.5"
                    stroke-linecap="round"
                  />
                </svg>
              </Flex>
            </div>
            <SegmentProcessingControls
              value={config.segment_processing || DEFAULT_SEGMENT_PROCESSING}
              onChange={(value) =>
                setConfig({
                  ...config,
                  segment_processing: value,
                })
              }
            />
          </div>

          <Flex direction="column" mt="32px">
            <JsonSchemaControls
              value={config.json_schema}
              onChange={(newSchema) =>
                setConfig({
                  ...config,
                  json_schema: newSchema,
                })
              }
            />
          </Flex>
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
              {isUploading ? "Processing..." : "Process Document"}
            </Text>
          </button>
        </section>
      </div>
    </div>
  );
}
