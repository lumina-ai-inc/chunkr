import { Flex, Text } from "@radix-ui/themes";
import "./Taskcard.css";
import { TaskResponse } from "../../models/task.model";

export interface TaskCardProps extends TaskResponse {
  onClick?: () => void;
}

export default function TaskCard({ onClick, ...task }: TaskCardProps) {
  return (
    <Flex
      direction="row"
      align="center"
      justify="between"
      className="task-card"
      onClick={onClick}
    >
      <Flex direction="row" align="center" justify="between">
        <Text size="2" weight="bold" style={{ color: "var(--cyan-3)" }}>
          {new Date(task.created_at).toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit",
            hour12: true,
          })}
        </Text>
      </Flex>
    </Flex>
  );
}
