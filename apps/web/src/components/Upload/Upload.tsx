import { Flex, Text } from "@radix-ui/themes";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useAuth } from "react-oidc-context";
import "./Upload.css";
import BetterButton from "../BetterButton/BetterButton";

interface UploadProps {
  onFileUpload: (files: File[]) => void;
  onFileRemove: (fileName: string) => void;
  files: File[];
  isAuthenticated: boolean;
}

export default function Upload({
  onFileUpload,
  onFileRemove,
  files,
  isAuthenticated,
}: UploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      onFileUpload(acceptedFiles);
    },
    [onFileUpload]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
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
    },
    multiple: true,
    noClick: true,
  });

  const auth = useAuth();

  return (
    <>
      <Flex direction="row" width="100%" gap="4" mb="16px">
        <Text
          size="5"
          weight="bold"
          style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
        >
          Create Ingestion Tasks
        </Text>
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
              UPLOADED {files.length} {files.length === 1 ? "FILE" : "FILES"}
            </Text>
          </Flex>
        )}
      </Flex>

      <Flex
        {...(isAuthenticated ? getRootProps() : {})}
        direction="row"
        width="100%"
        height="200px"
        align="center"
        justify="center"
        className={`upload-container ${!isAuthenticated ? "inactive" : ""} ${
          files.length > 0 ? "has-files" : ""
        }`}
        style={{ cursor: "pointer" }}
        onClick={isAuthenticated ? open : () => auth.signinRedirect()}
      >
        {isAuthenticated && <input {...getInputProps()} />}
        <Flex
          direction="column"
          py="24px"
          px="32px"
          style={{ border: "1px dashed hsla(0, 0%, 100%, 0.2)" }}
        >
          <Text size="7" weight="bold" className="white">
            {!isAuthenticated
              ? "Log In to start uploading"
              : files.length > 0
                ? `${files.length} ${files.length === 1 ? "File" : "Files"} Uploaded`
                : isDragActive
                  ? "Drop files here"
                  : "Upload Documents"}
          </Text>
          {isAuthenticated && (
            <Text
              size="4"
              className="white"
              weight="light"
              style={{ marginTop: "8px" }}
            >
              {files.length > 0
                ? `${files.length} ${files.length === 1 ? "file" : "files"} selected`
                : "Drag and drop documents or click"}
            </Text>
          )}
        </Flex>
      </Flex>
      {files.length > 0 && (
        <Flex direction="row" gap="2" wrap="wrap" width="100%">
          {files.map((file) => (
            <BetterButton
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
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
                <path
                  d="M7 7.00001L16.8995 16.8995"
                  stroke="#FFFFFF"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                />
              </svg>
            </BetterButton>
          ))}
        </Flex>
      )}
    </>
  );
}
