import { Progress, Text, Flex, Code } from "@radix-ui/themes";
import "./Statusview.css";
import { TaskResponse, Status } from "../../models/task.model";
import { Link, useNavigate } from "react-router-dom";

interface StatusViewProps {
  task: TaskResponse;
  pageCount: number;
}

export default function StatusView({ task, pageCount }: StatusViewProps) {
  const navigate = useNavigate();

  const handleRetry = () => {
    navigate("/");
  };

  const isHighQuality = task?.configuration.model === "HighQuality";
  const pagesPerSecond = isHighQuality ? 1 : 7;
  const calculatedDuration = Math.max(1, Math.ceil(pageCount / pagesPerSecond));
  const durationInSeconds = isHighQuality
    ? Math.max(12, calculatedDuration)
    : calculatedDuration;
  const durationString = `${durationInSeconds}s` as const;

  return (
    <Flex direction="column">
      {task?.status !== Status.Failed && (
        <Progress
          color="gray"
          duration={durationString}
          style={{
            height: "88px",
            borderRadius: "0px",
            color: "rgba(255, 255, 255, 0.3)",
          }}
        />
      )}
      <Flex direction="column" gap="4" className="status-title">
        {task?.status === Status.Failed ? (
          <Flex className="retry-button" onClick={handleRetry}>
            <Text size="6" weight="medium" style={{ color: "#FFF" }}>
              Retry
            </Text>
          </Flex>
        ) : (
          <Flex direction="column" className="status-title-badge">
            <Text
              size="6"
              weight="medium"
              style={{ color: "rgba(255, 255, 255, 0.8)" }}
            >
              {task?.configuration.model}
            </Text>
          </Flex>
        )}
        <Flex direction="column" className="status-title-badge">
          <Text
            size="6"
            weight="medium"
            style={{ color: "rgba(255, 255, 255, 0.8)" }}
          >
            OCR: {task?.configuration.ocr_strategy}
          </Text>
        </Flex>

        <Text
          size="9"
          weight="bold"
          className="pulsing-text status-title-text"
          trim="both"
          style={{ color: "#FFF" }}
        >
          {task?.status}
        </Text>
      </Flex>
      <Flex direction="column" gap="48px" className="message-container">
        <Code
          size="5"
          weight="medium"
          style={{
            color: "rgba(255, 255, 255, 0.8)",
            backgroundColor: "rgba(255, 255, 255, 0.05)",
          }}
        >
          {task?.message}
        </Code>
      </Flex>
      <Flex direction="column" gap="48px" className="status-items">
        <Flex direction="row" gap="4" className="status-item" wrap="wrap">
          <Text
            size="8"
            weight="bold"
            className="status-item-title"
            trim="both"
            style={{ color: "rgba(255, 255, 255, 0.6)" }}
          >
            Task ID
          </Text>
          <Text size="4" weight="regular" style={{ color: "#FFF" }} trim="both">
            {task?.task_id}
          </Text>
        </Flex>
        <Flex direction="row" gap="4" className="status-item" wrap="wrap">
          <Text
            size="8"
            weight="bold"
            className="status-item-title"
            trim="both"
            style={{ color: "rgba(255, 255, 255, 0.6)" }}
          >
            Created
          </Text>
          <Text size="4" weight="regular" style={{ color: "#FFF" }} trim="both">
            {task?.created_at.toLocaleString()}
          </Text>
        </Flex>
        {task?.expires_at && (
          <Flex direction="row" gap="4" className="status-item" wrap="wrap">
            <Text
              size="8"
              weight="bold"
              className="status-item-title"
              trim="both"
              style={{ color: "rgba(255, 255, 255, 0.6)" }}
            >
              Expires
            </Text>
            <Text
              size="4"
              weight="regular"
              style={{ color: "#FFF" }}
              trim="both"
            >
              {task?.expires_at?.toLocaleString() || "N/A"}
            </Text>
          </Flex>
        )}
      </Flex>
      <Link to="/">
        <Flex className="logo-status" direction="row" gap="4" align="center">
          <div className="logo-container">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
            >
              <path
                d="M5.88 9.78C6.42518 9.99822 7.02243 10.0516 7.59768 9.93364C8.17294 9.81564 8.70092 9.53139 9.11616 9.11616C9.53139 8.70092 9.81564 8.17294 9.93364 7.59768C10.0516 7.02243 9.99822 6.42518 9.78 5.88C10.4143 5.70922 10.975 5.33496 11.3761 4.81468C11.7771 4.29441 11.9963 3.65689 12 3C13.78 3 15.5201 3.52784 17.0001 4.51677C18.4802 5.50571 19.6337 6.91131 20.3149 8.55585C20.9961 10.2004 21.1743 12.01 20.8271 13.7558C20.4798 15.5016 19.6226 17.1053 18.364 18.364C17.1053 19.6226 15.5016 20.4798 13.7558 20.8271C12.01 21.1743 10.2004 20.9961 8.55585 20.3149C6.91131 19.6337 5.50571 18.4802 4.51677 17.0001C3.52784 15.5201 3 13.78 3 12C3.65689 11.9963 4.29441 11.7771 4.81468 11.3761C5.33496 10.975 5.70922 10.4143 5.88 9.78Z"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <Text size="4" weight="bold" className="logo-title">
              chunkr
            </Text>
          </div>
        </Flex>
      </Link>
    </Flex>
  );
}
