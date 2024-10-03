import { Flex, Text } from "@radix-ui/themes";
import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { useAuth } from "react-oidc-context";
import "./Upload.css";

interface UploadProps {
  onFileUpload: (file: File) => void;
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
            onFileUpload(file);
          }
        };
        reader.readAsArrayBuffer(file);
      }
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
    multiple: false,
    noClick: true,
  });

  // const navigate = useNavigate();
  const auth = useAuth();

  const handleContainerClick = () => {
    if (isAuthenticated) {
      if (isUploaded) {
        onFileRemove();
      }
      open();
    } else {
      auth.signinRedirect();
    }
  };

  // const DemoPdfLink = () => (
  //   <div className="demo-pdf-link-container">
  //     <Text
  //       size="3"
  //       weight="medium"
  //       className="demo-pdf-text"
  //       style={{
  //         cursor: "pointer",
  //       }}
  //       onClick={() => navigate("/task/da91192d-efd0-4924-9e1f-c973ebc3c31d/8")}
  //     >
  //       Click here for demo PDF
  //     </Text>
  //   </div>
  // );

  return (
    <>
      <Flex
        className="model-title-container"
        align="center"
        direction="row"
        mb="16px"
        gap="4"
        width="100%"
      >
        <Text
          size="4"
          weight="bold"
          style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
        >
          CREATE INGESTION TASK
        </Text>
        {isUploaded && (
          <Flex
            direction="row"
            gap="2"
            style={{
              padding: "4px 8px",
              backgroundColor: "hsla(180, 100%, 100%)",
              borderRadius: "4px",
            }}
          >
            <Text size="2" weight="bold" style={{ opacity: 0.9 }}>
              UPLOADED
            </Text>
          </Flex>
        )}
      </Flex>
      <Flex
        {...(isAuthenticated ? getRootProps() : {})}
        direction="row"
        width="100%"
        height="302px"
        align="center"
        justify="center"
        className={`upload-container ${!isAuthenticated ? "inactive" : ""} ${isUploaded ? "is-uploaded" : ""}`}
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
          <Text size="7" weight="bold" className="white">
            {!isAuthenticated
              ? "Log In to start uploading"
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
              {isUploaded
                ? fileName
                : "Drag and drop a document or click to select"}
            </Text>
          )}
        </Flex>
      </Flex>
      {/* <DemoPdfLink /> */}
    </>
  );
}
