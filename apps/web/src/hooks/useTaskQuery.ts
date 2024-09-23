import { useQuery } from "react-query";
import { getTask, getTasks } from "../services/taskApi";
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
        data &&
        (data.status === Status.Succeeded || data.status === Status.Failed)
          ? false
          : 1000,
      staleTime: staleTime,
      onSuccess: (data) => {
        setTaskResponse(data);
      },
    }
  );
}

export function useTasksQuery(page: number, limit: number) {
  return useQuery<TaskResponse[], Error>(
    [`tasks-${page}-${limit}`],
    () => getTasks(page, limit),
    {
      staleTime: 5000,
      refetchInterval: 5000,
      refetchIntervalInBackground: false,
    }
  );
}
