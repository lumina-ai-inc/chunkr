import { useParams, Link } from "react-router-dom";
import { Status } from "../../models/task.model";
import Loader from "../Loader/Loader";
import StatusView from "../../components/Status/StatusView";
import { Viewer } from "../../components/Viewer/Viewer";
import { useTaskQuery } from "../../hooks/useTaskQuery";

export default function Task() {
  const { taskId, pageCount } = useParams<{
    taskId: string;
    pageCount: string;
  }>();

  const { data: taskResponse, error, isLoading } = useTaskQuery(taskId);
  console.log(taskResponse);

  if (isLoading) {
    return <Loader />;
  }

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
          {error.message}
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
