import { Flex, Text } from "@radix-ui/themes";
import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

interface UploadProps {
  onFileUpload: (file: File) => void;
}

export default function Upload({ onFileUpload }: UploadProps) {
  const [isUploaded, setIsUploaded] = useState(false);
  const [fileName, setFileName] = useState("");

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0 && !isUploaded) {
        const file = acceptedFiles[0];
        onFileUpload(file);
        setFileName(file.name);
        setIsUploaded(true);
      }
    },
    [onFileUpload, isUploaded]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
    disabled: isUploaded,
  });

  return (
    <Flex
      {...getRootProps()}
      direction="row"
      width="100%"
      height="302px"
      align="center"
      justify="center"
      style={{
        backgroundColor: "#061D22",
        borderRadius: "8px",
        border: `4px solid ${
          isUploaded
            ? "var(--cyan-8)"
            : isDragActive
              ? "var(--cyan-7)"
              : "var(--cyan-5)"
        }`,
        boxShadow: "0px 0px 16px 0px rgba(12, 12, 12, 0.25)",
        cursor: isUploaded ? "default" : "pointer",
        opacity: isUploaded ? 0.7 : 1,
      }}
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
