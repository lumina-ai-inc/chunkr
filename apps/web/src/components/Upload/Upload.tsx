import { Flex, Text } from "@radix-ui/themes";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import { useAuth } from "react-oidc-context";
import "./Upload.css";

interface UploadProps {
  onFileUpload: (file: File, fileContent: ArrayBuffer) => void;
  onFileRemove: () => void;
  isUploaded: boolean;
  fileName: string;
  isAuthenticated: boolean;
}

export default function Upload({
  onFileUpload,
  onFileRemove,
  isUploaded,
  fileName,
  isAuthenticated,
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

  const navigate = useNavigate();
  const auth = useAuth();

  const handleContainerClick = () => {
    if (isAuthenticated) {
      if (isUploaded) {
        onFileRemove();
      }
      open(); // Open file dialog
    } else {
      auth.signinRedirect(); // Redirect to login when not authenticated
    }
  };

  const DemoPdfLink = () => (
    <div className="demo-pdf-link-container">
      <Text
        size="3"
        weight="medium"
        className="demo-pdf-text"
        style={{
          cursor: "pointer",
        }}
        onClick={() => navigate("/task/da91192d-efd0-4924-9e1f-c973ebc3c31d/8")}
      >
        Click here for demo PDF
      </Text>
    </div>
  );

  return (
    <>
      <Flex
        {...(isAuthenticated ? getRootProps() : {})}
        direction="row"
        width="100%"
        height="302px"
        align="center"
        justify="center"
        className={`upload-container ${!isAuthenticated ? "inactive" : ""}`}
        style={{ cursor: "pointer" }}
        onClick={
          isAuthenticated ? handleContainerClick : () => auth.signinRedirect()
        }
      >
        {isAuthenticated && <input {...getInputProps()} />}
        <Flex
          direction="column"
          py="24px"
          px="32px"
          style={{ border: "1px dashed hsla(0, 0%, 100%, 0.2)" }}
        >
          <Text size="7" weight="medium" className="white">
            {!isAuthenticated
              ? "Log In to test"
              : isUploaded
                ? "File Uploaded"
                : isDragActive
                  ? "Drop PDF here"
                  : "Upload Document"}
          </Text>
          {isAuthenticated && (
            <Text
              size="4"
              className="white"
              weight="light"
              style={{ marginTop: "8px" }}
            >
              {isUploaded ? fileName : "Drag and drop a PDF or click to select"}
            </Text>
          )}
        </Flex>
      </Flex>
      <DemoPdfLink />
    </>
  );
}
