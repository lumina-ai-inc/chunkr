import { Flex, Text } from "@radix-ui/themes";
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import Upload from "./Upload";
import "./UploadMain.css";
import { Model, UploadForm } from "../../models/upload.model";
import { uploadFile } from "../../services/uploadFileApi";
import HighQualityImage from "../../assets/cards/highQualityImage.webp";
import HighQualityImageJPG from "../../assets/cards/highQualityImage.jpg";
import FastImage from "../../assets/cards/fastImage.webp";
import FastImageJPG from "../../assets/cards/fastImage.jpg";

export default function UploadMain({
  isAuthenticated,
}: {
  isAuthenticated: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState("");
  const [model, setModel] = useState<Model>(Model.HighQuality);
  const [ocrStrategy, setOcrStrategy] = useState<"Auto" | "All" | "Off">(
    "Auto"
  );
  const [intelligentChunking, setIntelligentChunking] = useState(true);
  const [chunkLength, setChunkLength] = useState<number>(512);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleFileUpload = async (uploadedFile: File) => {
    setFile(uploadedFile);
    setFileName(uploadedFile.name);
  };

  const handleFileRemove = () => {
    setFile(null);
    setFileName("");
  };

  const handleModelToggle = (selectedModel: Model) => {
    if (model !== selectedModel) {
      setModel(selectedModel);
    }
  };

  const handleOcrStrategyChange = (strategy: "Auto" | "All" | "Off") => {
    setOcrStrategy(strategy);
  };

  const handleIntelligentChunkingToggle = () => {
    setIntelligentChunking(!intelligentChunking);
    if (intelligentChunking) {
      setChunkLength(0);
    } else {
      setChunkLength(512);
    }
  };

  const handleChunkLengthChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value === "") {
      setChunkLength(512);
    } else {
      const numValue = parseInt(value, 10);
      if (!isNaN(numValue)) {
        setChunkLength(numValue);
        if (numValue === 0) {
          setIntelligentChunking(false);
        }
      } else {
        setChunkLength(512);
      }
    }
  };

  const handleChunkLengthBlur = () => {
    if (chunkLength === 0) {
      setChunkLength(512);
      setIntelligentChunking(true);
    }
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
      ocr_strategy: ocrStrategy,
      target_chunk_length: intelligentChunking ? chunkLength : 0,
    };

    try {
      const taskResponse = await uploadFile(payload);
      navigate(
        `/task/${taskResponse.task_id}?pageCount=${taskResponse.page_count}`
      );
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
        isAuthenticated={isAuthenticated}
      />
      {isAuthenticated && (
        <>
          <Flex
            className="model-title-container"
            align="center"
            direction="row"
            mt="40px"
            gap="4"
            width="100%"
          >
            <Text
              size="4"
              weight="bold"
              style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
            >
              SELECT SEGMENTATION
            </Text>
            <Flex
              direction="row"
              gap="2"
              style={{
                padding: "4px 8px",
                backgroundColor: "hsla(180, 100%, 100%)",
                borderRadius: "4px",
              }}
            >
              {model === Model.HighQuality && (
                <Text size="2" weight="bold" style={{ opacity: 0.9 }}>
                  HIGH QUALITY
                </Text>
              )}
              {model === Model.Fast && (
                <Text size="2" weight="bold" style={{ opacity: 0.9 }}>
                  FAST
                </Text>
              )}
            </Flex>
          </Flex>
          <Flex
            width="100%"
            gap="4"
            mt="16px"
            p="16px"
            className="toggle-container"
            style={{
              borderRadius: "8px",
            }}
          >
            <Flex
              direction="column"
              height="100%"
              justify="end"
              minHeight="fit-content"
              className={model === Model.Fast ? "toggle-active" : "toggle"}
              style={{
                position: "relative",
                overflow: "hidden",
                zIndex: 1,
                borderRadius: "8px",
              }}
              onClick={() => handleModelToggle(Model.Fast)}
            >
              <div
                className="card-gradient-overlay"
                style={{
                  zIndex: 2,
                }}
              ></div>
              <picture style={{ zIndex: 1 }}>
                <source srcSet={FastImage} type="image/webp" />
                <img src={FastImageJPG} alt="Fast" className="card-image" />
              </picture>
              <Flex
                direction="column"
                className="toggle-icon-container"
                height="100%"
                justify="end"
                style={{
                  position: "relative",
                  zIndex: 3,
                }}
              >
                <Flex direction="row" gap="2" align="center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    className={
                      model === Model.Fast
                        ? "toggle-icon-active"
                        : "toggle-icon"
                    }
                  >
                    <rect
                      width="24"
                      height="24"
                      fill="white"
                      fillOpacity="0.01"
                    />
                    <path
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M13.9146 0.0645665C14.2537 0.209661 14.4497 0.567771 14.389 0.931576L12.9443 9.60002H20C20.3028 9.60002 20.58 9.77122 20.7155 10.0423C20.851 10.3133 20.8217 10.6376 20.6398 10.88L11.0399 23.68C10.8186 23.975 10.4243 24.0805 10.0852 23.9355C9.74612 23.7904 9.55017 23.4323 9.61081 23.0685L11.0555 14.4H3.99995C3.69694 14.4 3.41993 14.2288 3.28441 13.9578C3.14891 13.6868 3.17814 13.3624 3.35995 13.12L12.9599 0.320061C13.1812 0.0250008 13.5755 -0.0805282 13.9146 0.0645665ZM5.59995 12.8H11.9999C12.2351 12.8 12.4583 12.9035 12.6103 13.0829C12.7623 13.2623 12.8277 13.4996 12.789 13.7315L11.7284 20.0954L18.4 11.2H11.9999C11.7648 11.2 11.5415 11.0965 11.3895 10.9171C11.2375 10.7377 11.1721 10.5005 11.2108 10.2685L12.2715 3.90467L5.59995 12.8Z"
                      fill="white"
                    />
                  </svg>
                  <Text
                    size="6"
                    weight="bold"
                    style={{ position: "relative", zIndex: 1 }}
                  >
                    Fast
                  </Text>
                  <Flex
                    direction="row"
                    gap="2"
                    ml="2"
                    style={{
                      padding: "4px 8px",
                      border: "1px solid hsla(180, 100%, 100%, 0.1)",
                      backgroundColor: "hsla(180, 100%, 100%, 0.2)",
                      borderRadius: "4px",
                    }}
                  >
                    <Text size="2" weight="bold" style={{ opacity: 0.9 }}>
                      CPU
                    </Text>
                  </Flex>
                </Flex>

                <Text
                  size="4"
                  weight="medium"
                  mt="4"
                  style={{ maxWidth: "400px" }}
                >
                  <b>LightGBM</b>
                </Text>
                <Text size="4" weight="medium" mt="2">
                  Blazing speed - perfect for high volume tasks
                </Text>
                <Text size="2" weight="light" style={{ opacity: 0.8 }}>
                  $0.005/page | 1000 pages for free
                </Text>
              </Flex>
            </Flex>

            <Flex
              direction="column"
              height="100%"
              justify="end"
              minHeight="fit-content"
              className={
                model === Model.HighQuality ? "toggle-active" : "toggle"
              }
              style={{
                position: "relative",
                overflow: "hidden",
                zIndex: 1,
                borderRadius: "8px",
              }}
              onClick={() => handleModelToggle(Model.HighQuality)}
            >
              <div
                className="card-gradient-overlay"
                style={{
                  zIndex: 2,
                }}
              ></div>
              <picture style={{ zIndex: 1 }}>
                <source srcSet={HighQualityImage} type="image/webp" />
                <img
                  src={HighQualityImageJPG}
                  alt="High Quality"
                  className="card-image"
                />
              </picture>
              <Flex
                direction="column"
                className="toggle-icon-container"
                height="100%"
                justify="end"
                style={{
                  position: "relative",
                  zIndex: 3,
                }}
              >
                <Flex direction="row" gap="2" align="center">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    className={
                      model === Model.HighQuality
                        ? "toggle-icon-active"
                        : "toggle-icon"
                    }
                  >
                    <rect
                      width="24"
                      height="24"
                      fill="white"
                      fillOpacity="0.01"
                    />
                    <path
                      fillRule="evenodd"
                      clipRule="evenodd"
                      d="M1.40332 12.0033C1.40332 6.1491 6.1491 1.40332 12.0033 1.40332C17.8574 1.40332 22.6034 6.1491 22.6034 12.0033C22.6034 17.8574 17.8574 22.6034 12.0033 22.6034C6.1491 22.6034 1.40332 17.8574 1.40332 12.0033ZM2.95837 11.2C3.34176 6.82622 6.82622 3.34176 11.2 2.95837V7.20005C11.2 7.64187 11.5582 8.00005 12 8.00005C12.4419 8.00005 12.8 7.64187 12.8 7.20005V2.95779C17.177 3.33829 20.6646 6.82403 21.0483 11.2H16.8C16.3582 11.2 16 11.5582 16 12C16 12.4419 16.3582 12.8 16.8 12.8H21.0488C20.6682 17.179 17.179 20.6682 12.8 21.0488V16.8C12.8 16.3582 12.4419 16 12 16C11.5582 16 11.2 16.3582 11.2 16.8V21.0483C6.82403 20.6646 3.33829 17.177 2.95779 12.8H7.20005C7.64187 12.8 8.00005 12.4419 8.00005 12C8.00005 11.5582 7.64187 11.2 7.20005 11.2H2.95837Z"
                      fill="white"
                    />
                  </svg>
                  <Text
                    size="6"
                    weight="bold"
                    style={{ position: "relative", zIndex: 1 }}
                  >
                    High Quality
                  </Text>
                  <Flex
                    direction="row"
                    gap="2"
                    ml="2"
                    style={{
                      padding: "4px 8px",
                      border: "1px solid hsla(180, 100%, 100%, 0.1)",
                      backgroundColor: "hsla(180, 100%, 100%, 0.2)",
                      borderRadius: "4px",
                    }}
                  >
                    <Text size="2" weight="bold" style={{ opacity: 0.9 }}>
                      GPU
                    </Text>
                  </Flex>
                </Flex>
                <Text
                  size="4"
                  weight="medium"
                  mt="4"
                  style={{ maxWidth: "400px" }}
                >
                  <b>VGT</b>
                </Text>
                <Text
                  size="4"
                  weight="medium"
                  mt="2"
                  style={{ maxWidth: "400px" }}
                >
                  Higher accuracy + better image segmentation
                </Text>
                <Text size="2" weight="light" style={{ opacity: 0.8 }}>
                  $0.01/page | 500 pages for free
                </Text>
              </Flex>
            </Flex>
          </Flex>
          <Flex direction="row" gap="88px" wrap="wrap">
            <Flex
              className="ocr-strategy-container"
              direction="column"
              mt="40px"
              gap="4"
              width="fit-content"
            >
              <Text
                size="4"
                weight="bold"
                style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
              >
                OCR STRATEGY
              </Text>
              <Flex
                direction="row"
                gap="2"
                className="ocr-strategy-toggle"
                style={{
                  backgroundColor: "hsla(180, 100%, 100%, 0.1)",
                  borderRadius: "8px",
                  padding: "8px",
                }}
              >
                {["Auto", "All", "Off"].map((strategy) => (
                  <Flex
                    key={strategy}
                    align="center"
                    justify="center"
                    className={
                      ocrStrategy === strategy
                        ? "ocr-strategy-active"
                        : "ocr-strategy"
                    }
                    onClick={() =>
                      handleOcrStrategyChange(
                        strategy as "Auto" | "All" | "Off"
                      )
                    }
                    style={{
                      padding: "8px 16px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      transition: "background-color 0.2s ease",
                    }}
                  >
                    <Text
                      size="2"
                      weight="bold"
                      style={{
                        color:
                          ocrStrategy === strategy
                            ? "hsl(0, 0%, 0%)"
                            : "hsl(0, 0%, 100%, 0.7)",
                      }}
                    >
                      {strategy.toUpperCase()}
                    </Text>
                  </Flex>
                ))}
              </Flex>
            </Flex>
            <Flex
              className="chunk-length-container"
              direction="column"
              mt="40px"
              gap="4"
              width="fit-content"
            >
              <Text
                size="4"
                weight="bold"
                style={{ color: "hsl(0, 0%, 100%, 0.98)" }}
              >
                INTELLIGENT CHUNKING
              </Text>
              <Flex direction="row" gap="5" wrap="wrap">
                <Flex
                  direction="row"
                  gap="2"
                  width="fit-content"
                  className="chunk-length-toggle"
                  style={{
                    backgroundColor: "hsla(180, 100%, 100%, 0.1)",
                    borderRadius: "8px",
                    padding: "8px",
                  }}
                >
                  <Flex
                    align="center"
                    justify="center"
                    className={
                      intelligentChunking
                        ? "chunk-length-active"
                        : "chunk-length"
                    }
                    onClick={handleIntelligentChunkingToggle}
                    style={{
                      padding: "8px 16px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      transition: "background-color 0.2s ease",
                    }}
                  >
                    <Text
                      size="2"
                      weight="bold"
                      style={{
                        color: intelligentChunking
                          ? "hsl(0, 0%, 0%)"
                          : "hsl(0, 0%, 100%, 0.7)",
                      }}
                    >
                      ON
                    </Text>
                  </Flex>
                  <Flex
                    align="center"
                    justify="center"
                    className={
                      !intelligentChunking
                        ? "chunk-length-active"
                        : "chunk-length"
                    }
                    onClick={handleIntelligentChunkingToggle}
                    style={{
                      padding: "8px 16px",
                      borderRadius: "6px",
                      cursor: "pointer",
                      transition: "background-color 0.2s ease",
                    }}
                  >
                    <Text
                      size="2"
                      weight="bold"
                      style={{
                        color: !intelligentChunking
                          ? "hsl(0, 0%, 0%)"
                          : "hsl(0, 0%, 100%, 0.7)",
                      }}
                    >
                      OFF
                    </Text>
                  </Flex>
                </Flex>
                {intelligentChunking && (
                  <Flex
                    direction="row"
                    align="center"
                    justify="center"
                    style={{
                      backgroundColor: "hsla(180, 100%, 100%, 0.1)",
                      borderRadius: "6px",
                      padding: "8px",
                      maxWidth: "fit-content",
                    }}
                  >
                    <input
                      type="number"
                      min="0"
                      value={chunkLength}
                      onChange={handleChunkLengthChange}
                      onBlur={handleChunkLengthBlur}
                      className="chunk-length-input"
                      style={{
                        display: "flex",
                        border: "none",
                        fontSize: "14px",
                        fontWeight: "bold",
                        width: "auto",
                        maxWidth: "88px",
                        marginTop: "0px",
                        marginRight: "12px",
                        appearance: "textfield",
                        MozAppearance: "textfield",
                        WebkitAppearance: "textfield",
                      }}
                    />
                    <Text
                      size="2"
                      weight="bold"
                      style={{
                        color: "hsl(0, 0%, 100%, 0.7)",
                        paddingRight: "12px",
                      }}
                    >
                      {chunkLength === 0 ? "" : "~words/chunk"}
                    </Text>
                  </Flex>
                )}
              </Flex>
            </Flex>
          </Flex>

          <Flex direction="row" width="100%" mt="40px">
            <Flex
              direction="column"
              height="72px"
              justify="center"
              align="center"
              className={!!file && !isLoading ? "run-active" : "run"}
              style={{
                borderRadius: "8px",
                cursor: !!file && !isLoading ? "pointer" : "not-allowed",
              }}
              onClick={!!file && !isLoading ? handleRun : undefined}
            >
              <Text size="5" weight="bold">
                {isLoading ? "Uploading..." : "Run"}
              </Text>
            </Flex>
          </Flex>
        </>
      )}
    </Flex>
  );
}
