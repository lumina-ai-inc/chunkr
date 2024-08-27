import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import Upload from "./Upload";
import "./UploadMain.css";
import BetterButton from "../BetterButton/BetterButton";
import { Model, UploadForm } from "../../models/upload.model";
import { uploadFileStep } from "../../services/chunkMyDocs";

export default function UploadMain() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [model, setModel] = useState<Model>(Model.Fast);
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();

  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
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
    const payload: UploadForm = {
      file,
      model,
    };
    console.log("Component Payload:", payload);

    try {
      const taskResponse = await uploadFileStep(payload);
      console.log("Task Response:", taskResponse);
      // Navigate to the StatusView page with the task ID as a search parameter
      navigate(`/status?taskId=${taskResponse.task_id}`);
    } catch (error) {
      console.error("Error uploading file:", error);
      // Handle error (e.g., show an error message to the user)
    } finally {
      setIsLoading(false);
    }
  };

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
          <Text size="4" weight="medium">
            Fast
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
          <Text size="4" weight="medium">
            High Quality
          </Text>
        </Flex>
      </Flex>
      <Flex direction="row" width="100%" mt="32px">
        <BetterButton
          padding="16px 64px"
          onClick={handleRun}
          active={!!file && !isLoading}
        >
          <Text size="4" weight="medium">
            {isLoading ? "Uploading..." : "Run"}
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
