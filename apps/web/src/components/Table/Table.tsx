import { useMemo, useState } from "react";
import {
  MaterialReactTable,
  useMaterialReactTable,
  type MRT_ColumnDef,
  type MRT_PaginationState,
} from "material-react-table";
import { IconButton, Tooltip } from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { useNavigate } from "react-router-dom";
import { TaskResponse } from "../../models/task.model";
import { useTasksQuery } from "../../hooks/useTaskQuery";

const Table = () => {
  const navigate = useNavigate();
  const [pagination, setPagination] = useState<MRT_PaginationState>({
    pageIndex: 0,
    pageSize: 10,
  });

  const {
    data: { data = [], meta } = {},
    isError,
    isRefetching,
    isLoading,
    refetch,
  } = useTasksQuery(pagination.pageIndex + 1, pagination.pageSize);

  const handleTaskClick = (task: TaskResponse) => {
    navigate(`/task/${task.task_id}?pageCount=${task.page_count}`);
  };

  const columns = useMemo<MRT_ColumnDef<TaskResponse>[]>(
    () => [
      {
        accessorKey: "task_id",
        header: "Task ID",
      },
      {
        accessorKey: "status",
        header: "Status",
      },
      {
        accessorKey: "configuration.model",
        header: "Model",
      },
      {
        accessorKey: "page_count",
        header: "Pages",
      },
      {
        accessorKey: "created_at",
        header: "Created At",
        Cell: ({ cell }) => new Date(cell.getValue<string>()).toLocaleString(),
      },
    ],
    []
  );

  const table = useMaterialReactTable({
    columns,
    data,
    initialState: { showColumnFilters: false },
    manualPagination: true,
    muiToolbarAlertBannerProps: isError
      ? {
          color: "error",
          children: "Error loading data",
        }
      : undefined,
    onPaginationChange: setPagination,
    renderTopToolbarCustomActions: () => (
      <Tooltip arrow title="Refresh Data">
        <IconButton onClick={() => refetch()}>
          <RefreshIcon />
        </IconButton>
      </Tooltip>
    ),
    rowCount: meta?.totalRowCount ?? 0,
    state: {
      isLoading,
      pagination,
      showAlertBanner: isError,
      showProgressBars: isRefetching,
    },
    muiTableBodyRowProps: ({ row }) => ({
      onClick: () => handleTaskClick(row.original),
      sx: { cursor: "pointer" },
    }),
    enablePagination: true,
    paginationDisplayMode: "pages",
    positionPagination: "bottom",
    muiPaginationProps: {
      color: "primary",
      shape: "rounded",
    },
  });

  return <MaterialReactTable table={table} />;
};

export default Table;
