import { Flex, Text, Badge, Separator } from "@radix-ui/themes";
import "./Taskcard.css";
import { TaskResponse, Status } from "../../models/task.model";

export interface TaskCardProps extends TaskResponse {
  onClick?: () => void;
}

const statusColors = {
  [Status.Starting]: "#007AFF",
  [Status.Processing]: "#FF9500",
  [Status.Succeeded]: "#34C759",
  [Status.Failed]: "#FF3B30",
  [Status.Canceled]: "#FF9500",
};

export default function TaskCard({ onClick, ...task }: TaskCardProps) {
  const statusColor = statusColors[task.status];

  return (
    <Flex
      direction="column"
      className="task-card"
      onClick={onClick}
      width="100%"
    >
      <Flex justify="between" align="center" mb="2">
        <Text size="1" style={{ color: "rgba(255, 255, 255, 0.6)" }}>
          {new Date(task.created_at).toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
            hour12: true,
          })}
        </Text>
        <Badge
          size="1"
          style={{
            backgroundColor: statusColor,
            color: "#FFF",
            padding: "4px 8px",
            borderRadius: "12px",
            fontWeight: "500",
          }}
        >
          {task.status}
        </Badge>
      </Flex>
      <Text size="3" weight="bold" mb="1" style={{ color: "#FFF" }}>
        {task.file_name}
      </Text>
      <Text
        size="4"
        mb="2"
        weight="medium"
        style={{ color: "rgba(255, 255, 255, 1)" }}
      >
        {task.message}
      </Text>
      <Flex gap="2" wrap="wrap">
        <Text
          size="2"
          weight="medium"
          style={{ color: "rgba(255, 255, 255, 0.9)" }}
        >
          Model: {task.configuration.model}
        </Text>
        <Separator
          size="4"
          orientation="vertical"
          style={{ background: "rgba(255, 255, 255, 0.9)" }}
        />
        <Text
          size="2"
          weight="medium"
          style={{ color: "rgba(255, 255, 255, 0.9)" }}
        >
          OCR: {task.configuration.ocr_strategy}
        </Text>
        <Separator
          size="4"
          orientation="vertical"
          style={{ background: "rgba(255, 255, 255, 0.9)" }}
        />
        <Text
          size="2"
          weight="medium"
          style={{ color: "rgba(255, 255, 255, 0.9)" }}
        >
          Target Chunk Length: {task.configuration.target_chunk_length || "N/A"}
        </Text>
      </Flex>
    </Flex>
  );
}
