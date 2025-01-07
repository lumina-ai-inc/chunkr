import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import {
  UploadFormData,
  OcrStrategy,
  SegmentationStrategy,
  DEFAULT_UPLOAD_CONFIG,
  DEFAULT_SEGMENT_PROCESSING,
} from "../../models/newTask.model";
import "./NewUpload.css";
import Upload from "./Upload";
import {
  ToggleGroup,
  NumberInput,
  SegmentProcessingControls,
  JsonSchemaControls,
} from "./ConfigControls";

interface NewUploadFormProps {
  onSubmit: (config: UploadFormData) => void;
  isAuthenticated: boolean;
}

export default function NewUploadForm({
  onSubmit,
  isAuthenticated,
}: NewUploadFormProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<Partial<UploadFormData>>(
    DEFAULT_UPLOAD_CONFIG
  );

  const handleFileChange = (uploadedFiles: File[]) => {
    setFiles((prev) => [...prev, ...uploadedFiles]);
  };

  const handleFileRemove = (fileName: string) => {
    setFiles((prev) => prev.filter((file) => file.name !== fileName));
  };

  const handleSubmit = () => {
    if (files.length === 0) return;
    onSubmit({ ...config, files } as UploadFormData);
  };

  return (
    <div className="upload-form-container">
      <section className="upload-section">
        <Upload
          onFileUpload={handleFileChange}
          onFileRemove={handleFileRemove}
          files={files}
          isAuthenticated={isAuthenticated}
        />
      </section>

      <div>
        <section
          className={`config-section ${!isAuthenticated ? "disabled" : ""}`}
        >
          <div className="config-grid">
            <ToggleGroup
              label="Segmentation Strategy"
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
              label="OCR Strategy"
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
              label="Target Chunk Length"
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
              label="High Resolution"
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

          <div className="config-card" style={{ marginTop: "24px" }}>
            <div className="config-card-header">
              <Text size="3" weight="bold" className="white">
                Segment Processing
              </Text>
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

          <Flex direction="column" mt="24px">
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
            disabled={files.length === 0 || !isAuthenticated}
          >
            <Text size="3" weight="bold">
              Process Document
            </Text>
          </button>
        </section>
      </div>
    </div>
  );
}
