import { Flex, Text, Badge } from "@radix-ui/themes";
import "./Taskcard.css";
import { TaskResponse, Status } from "../../models/task.model";

export interface TaskCardProps extends TaskResponse {
  onClick?: () => void;
}

const statusColors = {
  [Status.Starting]: "cyan-4",
  [Status.Processing]: "cyan-8",
  [Status.Succeeded]: "cyan-11",
  [Status.Failed]: "red-8",
  [Status.Canceled]: "amber-8",
};

export default function TaskCard({ onClick, ...task }: TaskCardProps) {
  const statusColor = statusColors[task.status];

  return (
    <Flex
      direction="row"
      align="center"
      justify="between"
      className="task-card"
      onClick={onClick}
    >
      <Text size="2" weight="bold" className="cyan-3">
        {new Date(task.created_at).toLocaleString("en-US", {
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit",
          hour12: true,
        })}
      </Text>
      <Text
        size="2"
        className="cyan-2"
        style={{ flex: 1, marginLeft: "16px", marginRight: "16px" }}
      >
        {task.message}
      </Text>
      <Text size="2" className="cyan-4" style={{ marginRight: "16px" }}>
        {task.configuration.model}
      </Text>
      <Text size="2" className="cyan-4" style={{ marginRight: "16px" }}>
        {task.configuration.target_chunk_length || "N/A"}
      </Text>
      <Badge size="1" style={{ backgroundColor: statusColor }}>
        {task.status}
      </Badge>
    </Flex>
  );
}
