import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Viewer } from "../../pages/Viewer/Viewer";
import StatusView from "../../pages/Status/StatusView";
import { getTask } from "../../services/uploadFileApi";
import { TaskResponse, Status } from "../../models/task.model";
import Loader from "../../pages/Loader/Loader";

export const Task: React.FC = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [task, setTask] = useState<TaskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTaskStatus = async () => {
      if (!taskId) {
        setError("No task ID provided");
        setIsLoading(false);
        return;
      }

      try {
        const taskResponse = await getTask(taskId);
        setTask(taskResponse);

        if (taskResponse.status === Status.Succeeded) {
          navigate(
            `/viewer?output_file_url=${encodeURIComponent(taskResponse.output_file_url || "")}&input_file_url=${encodeURIComponent(taskResponse.input_file_url || "")}`
          );
        }
      } catch (error) {
        console.error("Error fetching task data:", error);
        setError("Failed to fetch task status. Please try again later.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchTaskStatus();
    const pollInterval = setInterval(fetchTaskStatus, 5000); // Poll every 5 seconds

    return () => clearInterval(pollInterval);
  }, [taskId, navigate]);

  if (isLoading) {
    return <Loader />;
  }

  if (error) {
    return <div>{error}</div>;
  }

  if (!task) {
    return <div>No task data available</div>;
  }

  if (task.status === Status.Succeeded) {
    return <Viewer />;
  }

  return <StatusView />;
};
