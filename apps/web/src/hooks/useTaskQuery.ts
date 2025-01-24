import { useQuery } from "react-query";
import { getTask, getTasks } from "../services/taskApi";
import { TaskResponse, Status } from "../models/taskResponse.model";
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

export function useTasksQuery(
  page?: number,
  limit?: number,
  start?: string,
  end?: string
) {
  return useQuery<TaskResponse[], Error>(
    ["tasks", page, limit, start, end],
    () => getTasks(page, limit, start, end),
    {
      staleTime: 50000,
      refetchInterval:
        start && new Date(start) > new Date(Date.now() - 24 * 60 * 60 * 1000)
          ? 50000 // Refetch every 50s for recent data
          : false, // Don't refetch for historical data
      refetchIntervalInBackground: false,
    }
  );
}
