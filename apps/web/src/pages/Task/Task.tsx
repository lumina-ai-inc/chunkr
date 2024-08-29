import { useParams, Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { getTask } from "../../services/uploadFileApi";
import { TaskResponse, Status } from "../../models/task.model";
import Loader from "../Loader/Loader";
import StatusView from "../../components/Status/StatusView";
import { Viewer } from "../../components/Viewer/Viewer";

export default function Task() {
  const { taskId, pageCount } = useParams<{
    taskId: string;
    pageCount: string;
  }>();

  const [taskResponse, setTaskResponse] = useState<TaskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const pollTask = async () => {
      if (!taskId) return;

      try {
        const response = await getTask(taskId);
        setTaskResponse(response);

        if (response.status !== Status.Succeeded) {
          setTimeout(() => pollTask(), 1000);
        }
        // If status is Succeeded, we don't set up another timeout
      } catch (err) {
        setError("Failed to fetch task status");
        console.error(err);
      }
    };

    pollTask();
  }, [taskId]);

  if (!taskResponse) {
    return <Loader />;
  }

  if (error) {
    return (
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
    );
  }

  if (taskResponse.status !== Status.Succeeded) {
    return (
      <StatusView task={taskResponse} pageCount={Number(pageCount) || 10} />
    );
  }

  return taskResponse.output_file_url && taskResponse.input_file_url ? (
    <Viewer
      outputFileUrl={taskResponse.output_file_url}
      inputFileUrl={taskResponse.input_file_url}
    />
  ) : (
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
  );
}
