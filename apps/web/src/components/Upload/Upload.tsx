import { Flex, Text } from "@radix-ui/themes";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import "./Upload.css";

interface UploadProps {
  onFileUpload: (file: File, fileContent: ArrayBuffer) => void;
  onFileRemove: () => void;
  isUploaded: boolean;
  fileName: string;
}

export default function Upload({
  onFileUpload,
  onFileRemove,
  isUploaded,
  fileName,
}: UploadProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        const reader = new FileReader();
        reader.onload = (event) => {
          if (event.target && event.target.result instanceof ArrayBuffer) {
            onFileUpload(file, event.target.result);
          }
        };
        reader.readAsArrayBuffer(file);
      }
    },
    [onFileUpload]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    noClick: true, // Prevent opening file dialog on click
  });

  const handleClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (isUploaded) {
      onFileRemove();
    }
    open(); // Open file dialog
  };

  return (
    <Flex
      {...getRootProps()}
      direction="row"
      width="100%"
      height="302px"
      align="center"
      justify="center"
      className="upload-container"
      style={{ cursor: "pointer" }}
      onClick={handleClick}
    >
      <input {...getInputProps()} />
      <Flex
        direction="column"
        py="10px"
        px="12px"
        style={{ border: "1px dashed var(--Colors-Cyan-6, #9DDDE7)" }}
      >
        <Text size="6" weight="bold" className="cyan-1">
          {isUploaded
            ? "File Uploaded"
            : isDragActive
              ? "Drop PDF here"
              : "Upload Document"}
        </Text>
        {isUploaded ? (
          <Text size="2" className="cyan-3" style={{ marginTop: "8px" }}>
            {fileName}
          </Text>
        ) : (
          <Text size="2" className="cyan-3" style={{ marginTop: "8px" }}>
            Drag and drop a PDF or click to select
          </Text>
        )}
      </Flex>
    </Flex>
  );
}
