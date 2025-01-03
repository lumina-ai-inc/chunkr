import { useMemo, useState } from "react";
import {
  MaterialReactTable,
  type MRT_ColumnDef,
  type MRT_PaginationState,
} from "material-react-table";
import { IconButton, Tooltip, createTheme, ThemeProvider } from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { useNavigate } from "react-router-dom";
import { TaskResponse } from "../../models/task.model";
import { useTasksQuery } from "../../hooks/useTaskQuery";
import useUser from "../../hooks/useUser";
import "./Table.css";

const Table = () => {
  const navigate = useNavigate();
  const [pagination, setPagination] = useState<MRT_PaginationState>({
    pageIndex: 0,
    pageSize: 10,
  });

  const { data: user } = useUser();
  const totalTasks = user?.task_count || 0;

  const {
    data: tasks,
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

  const tableTheme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: "dark",
          primary: {
            main: "rgba(255, 255, 255, 0.8)",
          },
          info: {
            main: "#67e8f9",
          },
          background: {
            default: "rgba(0, 0, 0, 0)",
            paper: "rgba(0, 0, 0, 0)",
          },
        },
        typography: {
          button: {
            textTransform: "none",
            fontSize: "14px",
            fontWeight: 500,
          },
        },
        components: {
          MuiPaper: {
            styleOverrides: {
              root: {
                backgroundImage: "none",
                backgroundColor: "rgba(0, 0, 0)",
                border: "1px solid rgba(255, 255, 255, 0.03)",
                boxShadow: "none",
                backdropFilter: "none",
              },
            },
          },
          MuiTableHead: {
            styleOverrides: {
              root: {
                "& .MuiTableCell-head": {
                  color: "rgba(255, 255, 255, 0.8)",
                  fontSize: "14px",
                  fontWeight: 500,
                  borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
                  padding: "24px",
                  backgroundColor: "rgba(0,0,0)",
                },
              },
            },
          },
          MuiTableBody: {
            styleOverrides: {
              root: {
                overflow: "auto",
                "& .MuiTableRow-root": {
                  "&:hover": {
                    backgroundColor: "rgba(0, 0, 0, 0)",
                    backdropFilter: "blur(8px)",
                  },
                  transition: "all 0.2s ease",
                },
                "& .MuiTableCell-body": {
                  color: "rgba(255, 255, 255, 0.8)",
                  fontSize: "14px",
                  padding: "16px 24px",
                  borderBottom: "1px solid rgba(255, 255, 255, 0.03)",
                },
              },
            },
          },
          MuiIconButton: {
            styleOverrides: {
              root: {
                color: "rgba(255, 255, 255, 0.8)",
                padding: "8px",
                "&:hover": {
                  backgroundColor: "rgba(255, 255, 255, 0.05)",
                },
                transition: "all 0.2s ease",
              },
            },
          },
          MuiTablePagination: {
            styleOverrides: {
              root: {
                color: "rgba(255, 255, 255, 0.8)",
              },
              select: {
                color: "rgba(255, 255, 255, 0.8)",
              },
              selectIcon: {
                color: "rgba(255, 255, 255, 0.8)",
              },
            },
          },
          MuiTooltip: {
            styleOverrides: {
              tooltip: {
                fontSize: "14px",
                background: "rgba(0, 0, 0, 0)",
                backdropFilter: "blur(8px)",
                padding: "8px 12px",
                borderRadius: "6px",
              },
            },
          },
          MuiToolbar: {
            styleOverrides: {
              root: {
                padding: "24px",
              },
            },
          },
        },
      }),
    []
  );

  return (
    <ThemeProvider theme={tableTheme}>
      <MaterialReactTable
        columns={columns}
        data={tasks || []}
        enableColumnOrdering
        enableColumnPinning
        enableRowSelection
        enablePagination
        manualPagination
        enableStickyHeader
        enableStickyFooter
        muiTableContainerProps={{
          sx: { height: "calc(100% - 120px)", width: "100%" },
        }}
        onPaginationChange={setPagination}
        rowCount={totalTasks}
        state={{
          isLoading,
          pagination,
          showAlertBanner: isError,
          showProgressBars: isRefetching,
        }}
        muiToolbarAlertBannerProps={
          isError
            ? {
                color: "error",
                children: "Error loading data",
              }
            : undefined
        }
        renderTopToolbarCustomActions={() => (
          <Tooltip arrow title="Refresh Data">
            <IconButton onClick={() => refetch()}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        )}
        muiTableBodyRowProps={({ row }) => ({
          onClick: () => handleTaskClick(row.original),
          sx: { cursor: "pointer" },
        })}
      />
    </ThemeProvider>
  );
};

export default Table;
