import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import Upload from "./Upload";
import "./UploadMain.css";
import BetterButton from "../BetterButton/BetterButton";
import { Model, UploadForm } from "../../models/upload.model";
import { uploadFileStep } from "../../services/chunkMyDocs";
import * as pdfjsLib from "pdfjs-dist";

pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

export default function UploadMain() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [pageCount, setPageCount] = useState<number | null>(null);
  const [model, setModel] = useState<Model>(Model.Fast);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleFileUpload = async (uploadedFile: File) => {
    setFile(uploadedFile);
    setFileName(uploadedFile.name);

    if (uploadedFile.type === "application/pdf") {
      try {
        const arrayBuffer = await uploadedFile.arrayBuffer();
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;

        setPageCount(pdf.numPages);
      } catch (error) {
        console.error("Error reading PDF:", error);
        setPageCount(null);
      }
    } else {
      setPageCount(null);
    }
  };

  const handleFileRemove = () => {
    setFile(null);
    setFileName("");
  };

  const handleModelToggle = () => {
    setModel(model === Model.Fast ? Model.HighQuality : Model.Fast);
  };

  const handleRun = async () => {
    if (!file) {
      console.error("No file uploaded");
      return;
    }

    setIsLoading(true);
    setError(null);
    const payload: UploadForm = {
      file,
      model,
    };

    try {
      const taskResponse = await uploadFileStep(payload);
      navigate(`/task/${taskResponse.task_id}/${pageCount}`);
    } catch (error) {
      console.error("Error uploading file:", error);
      setError("Failed to upload file. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  };

  if (error) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100%",
          width: "100%",
        }}
      >
        <Link to="/" style={{ textDecoration: "none" }}>
          <div
            style={{
              color: "var(--red-9)",
              padding: "8px 12px",
              border: "2px solid var(--red-12)",
              borderRadius: "4px",
              backgroundColor: "var(--red-7)",
              cursor: "pointer",
              transition: "background-color 0.2s ease",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.backgroundColor = "var(--red-8)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.backgroundColor = "var(--red-7)")
            }
          >
            {error}
          </div>
        </Link>
      </div>
    );
  }

  return (
    <Flex direction="column" width="100%">
      <Upload
        onFileUpload={handleFileUpload}
        onFileRemove={handleFileRemove}
        isUploaded={!!file}
        fileName={fileName}
      />
      <Flex
        direction="row"
        height="64px"
        width="100%"
        mt="40px"
        className="toggle-container"
        onClick={handleModelToggle}
      >
        <Flex
          direction="column"
          height="100%"
          justify="center"
          className={model === Model.Fast ? "toggle-active" : "toggle"}
          style={{ borderTopLeftRadius: "4px", borderBottomLeftRadius: "4px" }}
        >
          <Text size="5" weight="medium">
            Fast{"  "}
            <Text size="3" weight="light" style={{ opacity: "0.8" }}>
              (7 pages/s)
            </Text>
          </Text>
        </Flex>
        <Flex
          direction="column"
          height="100%"
          justify="center"
          className={model === Model.HighQuality ? "toggle-active" : "toggle"}
          style={{
            borderTopRightRadius: "4px",
            borderBottomRightRadius: "4px",
          }}
        >
          <Text size="5" weight="medium">
            High Quality{" "}
            <Text size="3" weight="light" style={{ opacity: "0.8" }}>
              (1 page/s)
            </Text>
          </Text>
        </Flex>
      </Flex>
      <Flex direction="row" width="100%" mt="32px">
        <BetterButton
          padding="16px 64px"
          onClick={handleRun}
          active={!!file && !isLoading}
        >
          <Text size="5" weight="medium">
            {isLoading ? "Uploading..." : "Run"}
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
