import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import Upload from "./Upload";
import "./UploadMain.css";
import BetterButton from "../BetterButton/BetterButton";
import { Model, UploadForm } from "../../models/upload.model";
import uploadFile from "../../services/uploadFileApi";
// import { extractFile } from "../../services/extractFileApi";

export default function UploadMain() {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [model, setModel] = useState<Model>(Model.Fast);

  const handleFileUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
    console.log("Uploaded file:", uploadedFile);
  };

  const handleFileRemove = () => {
    setFile(null);
    setFileName("");
  };

  const handleModelToggle = () => {
    setModel(model === Model.Fast ? Model.HighQuality : Model.Fast);
  };

  const handleRun = () => {
    if (!file) {
      console.error("No file uploaded");
      return;
    }

    const payload: UploadForm = {
      file,
      model,
    };
    console.log("Component Payload:", payload);

    const task = uploadFile(payload);
    console.log("Task:", task);
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
        <BetterButton padding="16px 64px" onClick={handleRun} active={!!file}>
          <Text size="4" weight="medium">
            Run
          </Text>
        </BetterButton>
      </Flex>
    </Flex>
  );
}
