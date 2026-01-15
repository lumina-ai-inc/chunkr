import { Flex, Text } from "@radix-ui/themes";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useAuth } from "react-oidc-context";
import "./Upload.css";
import BetterButton from "../BetterButton/BetterButton";

// Reserved for future use
// const DOCS_URL = import.meta.env.VITE_DOCS_URL;

interface UploadProps {
  onFileUpload: (files: File[]) => void;
  onFileRemove: (fileName: string) => void;
  files: File[];
  isAuthenticated: boolean;
  isUploading?: boolean;
}

export default function Upload({
  onFileUpload,
  onFileRemove,
  files,
  isAuthenticated,
  isUploading = false,
}: UploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      onFileUpload(acceptedFiles);
    },
    [onFileUpload]
  );

  const { getRootProps, getInputProps, open } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "application/msword": [".doc"],
      "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        [".pptx"],
      "application/vnd.ms-powerpoint": [".ppt"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
        ".xlsx",
      ],
      "application/vnd.ms-excel": [".xls"],
      "image/jpeg": [".jpg", ".jpeg"],
      "image/png": [".png"],
    },
    multiple: true,
    noClick: true,
  });

  const auth = useAuth();

  return (
    <>
      <Flex direction="row" width="100%" gap="4" mb="0">
        <Flex direction="column" gap="4">
          <Flex direction="row" justify="between" width="100%">
            {files.length > 0 && (
              <Flex
                direction="row"
                gap="2"
                style={{
                  padding: "4px 8px",
                  backgroundColor: "hsla(180, 100%, 100%)",
                  borderRadius: "4px",
                  width: "fit-content",
                }}
              >
                <Text size="2" weight="bold" style={{ opacity: 0.9 }}>
                  UPLOADED {files.length}{" "}
                  {files.length === 1 ? "FILE" : "FILES"}
                </Text>
              </Flex>
            )}
          </Flex>

        </Flex>
      </Flex>

      <Flex
        {...(isAuthenticated && !isUploading ? getRootProps() : {})}
        direction="row"
        width="100%"
        height="200px"
        align="center"
        justify="center"
        className={`upload-container ${!isAuthenticated ? "inactive" : ""} ${
          files.length > 0 ? "has-files" : ""
        } ${isUploading ? "uploading" : ""}`}
        style={{ cursor: isUploading ? "default" : "pointer" }}
        onClick={
          isUploading
            ? undefined
            : isAuthenticated
            ? open
            : () => auth.signinRedirect()
        }
      >
        <input {...(isUploading ? {} : getInputProps())} />
        <Flex
          direction="column"
          py="18px"
          px="32px"
          style={{ border: "1px dashed hsla(0, 0%, 100%, 0.2)" }}
        >
          <Text size="6" weight="medium" className="white">
            {isUploading
              ? "Processing Files..."
              : files.length > 0
              ? `${files.length} ${
                  files.length === 1 ? "File" : "Files"
                } Uploaded`
              : "AI Extract & Verify"}
          </Text>
          <Text
            size="4"
            className="white"
            weight="light"
            style={{ marginTop: "8px" }}
          >
            {files.length > 0
              ? `${files.length} ${
                  files.length === 1 ? "file" : "files"
                } selected`
              : "Upload lease agreements, mortgage statements, and rent rolls to auto fill your capital package"}
              <br />
          </Text>
          <Flex direction="column" gap="1" wrap="wrap" mt="8px">
            <Flex direction="row" gap="2" wrap="wrap">
              <Text size="1" weight="medium" className="white">
                PDF
              </Text>
              <Text size="1" weight="medium" className="white">
                DOCX
              </Text>
              <Text size="1" weight="medium" className="white">
                DOC
              </Text>
              <Text size="1" weight="medium" className="white">
                PPTX
              </Text>
              <Text size="1" weight="medium" className="white">
                PPT
              </Text>
            </Flex>

            <Flex direction="row" gap="2" wrap="wrap">
              <Text size="1" weight="medium" className="white">
                XLSX
              </Text>
              <Text size="1" weight="medium" className="white">
                XLS
              </Text>
              <Text size="1" weight="medium" className="white">
                JPEG
              </Text>
              <Text size="1" weight="medium" className="white">
                JPG
              </Text>
              <Text size="1" weight="medium" className="white">
                PNG
              </Text>
            </Flex>
          </Flex>
        </Flex>
      </Flex>
      {files.length > 0 && !isUploading && (
        <Flex direction="row" gap="2" wrap="wrap" width="100%">
          {files.map((file) => (
            <BetterButton
              key={file.name}
              onClick={(e) => {
                e.stopPropagation();
                onFileRemove(file.name);
              }}
            >
              <Text size="2" style={{ color: "hsla(0, 0%, 100%, 0.9)" }}>
                {file.name}
              </Text>
              <svg
                width="16px"
                height="16px"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M7 17L16.8995 7.10051"
                  stroke="#FFFFFF"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <path
                  d="M7 7.00001L16.8995 16.8995"
                  stroke="#FFFFFF"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </BetterButton>
          ))}
        </Flex>
      )}
    </>
  );
}
