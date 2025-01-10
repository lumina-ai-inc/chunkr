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
    },
    multiple: true,
    noClick: true,
  });

  const auth = useAuth();

  return (
    <>
      <Flex direction="row" width="100%" gap="4" mb="16px">
        <Flex direction="row" gap="2" align="center">
          <svg
            width="24px"
            height="24px"
            viewBox="0 0 25 24"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <g clip-path="url(#clip0_113_1479)">
              <path
                d="M19.75 9.25V20.25C19.75 20.8 19.3 21.25 18.75 21.25H6.25C5.7 21.25 5.25 20.8 5.25 20.25V3.75C5.25 3.2 5.7 2.75 6.25 2.75H13.25"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M13.25 9.25H19.75L13.25 2.75V9.25Z"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M10 15.25L12.5 12.75L15 15.25"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
              <path
                d="M12.5 13.75V18.25"
                stroke="#FFF"
                stroke-width="1.5"
                stroke-linecap="round"
                stroke-linejoin="round"
              />
            </g>
            <defs>
              <clipPath id="clip0_113_1479">
                <rect
                  width="24"
                  height="24"
                  fill="white"
                  transform="translate(0.5)"
                />
              </clipPath>
            </defs>
          </svg>
          <Text
            size="5"
            weight="bold"
            style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
          >
            Create Tasks
          </Text>
        </Flex>

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
          py="24px"
          px="32px"
          style={{ border: "1px dashed hsla(0, 0%, 100%, 0.2)" }}
        >
          <Text size="7" weight="bold" className="white">
            {isUploading
              ? "Processing Files..."
              : files.length > 0
                ? `${files.length} ${files.length === 1 ? "File" : "Files"} Uploaded`
                : "Upload Files"}
          </Text>
          <Text
            size="4"
            className="white"
            weight="medium"
            style={{ marginTop: "8px" }}
          >
            {files.length > 0
              ? `${files.length} ${files.length === 1 ? "file" : "files"} selected`
              : "Drag and drop documents or click"}
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
