import { useQuery } from "react-query";
import { getTask } from "../services/uploadFileApi";
import { TaskResponse, Status } from "../models/task.model";
import { useState } from "react";

export function useTaskQuery(taskId: string | undefined) {
  const [taskResponse, setTaskResponse] = useState<TaskResponse | null>(null);
  const staleTime = taskResponse?.status === Status.Succeeded ? 600000 : 1000;
  return useQuery<TaskResponse, Error>(
    ["task", taskId],
    () => getTask(taskId!),
    {
      enabled: !!taskId,
      refetchInterval: (data) =>
        data && data.status === Status.Succeeded ? false : 1000,
      staleTime: staleTime,
      onSuccess: (data) => {
        setTaskResponse(data);
      },
    }
  );
}
