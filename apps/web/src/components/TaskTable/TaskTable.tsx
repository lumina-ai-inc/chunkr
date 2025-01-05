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
import "./TaskTable.css";

const TaskTable = () => {
  const navigate = useNavigate();
  const searchParams = new URLSearchParams(window.location.search);
  const tablePageIndex = parseInt(searchParams.get("tablePageIndex") || "0");
  const tablePageSize = parseInt(searchParams.get("tablePageSize") || "10");

  const [pagination, setPagination] = useState<MRT_PaginationState>({
    pageIndex: tablePageIndex,
    pageSize: tablePageSize,
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
    navigate(
      `/newDashboard?taskId=${task.task_id}&pageCount=${task.page_count || 10}&tablePageIndex=${pagination.pageIndex}&tablePageSize=${pagination.pageSize}`
    );
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
            main: "#ffffff",
          },
          info: {
            main: "rgb(2, 8, 9)",
          },
          background: {
            default: "rgb(2, 8, 9)",
            paper: "rgb(2, 8, 9)",
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
                boxShadow: "none",
                backdropFilter: "none",
                height: "100%",
                zIndex: 1000,
              },
            },
          },
          MuiTableHead: {
            styleOverrides: {
              root: {
                "& .MuiTableCell-head": {
                  color: "rgb(255, 255, 255, 0.9)",
                  backgroundColor: "rgb(2, 8, 9, 0.7)",
                  fontSize: "16px",
                  fontWeight: 600,
                  borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
                  padding: "14px",
                  paddingBottom: "16px",
                  paddingTop: "18px",
                },
              },
            },
          },
          MuiTableBody: {
            styleOverrides: {
              root: {
                "& .MuiTableRow-root": {
                  backgroundColor: "rgb(2, 8, 9) !important",
                  "&:hover": {
                    backgroundColor: "rgb(2, 8, 9) !important",
                    backdropFilter: "blur(8px)",
                  },
                  transition: "all 0.2s ease",
                },
                "& .MuiTableCell-body": {
                  color: "rgba(255, 255, 255, 0.8)",
                  fontSize: "14px",
                  padding: "16px 14px",
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
                borderRadius: "6px",
                "&:hover": {
                  backgroundColor: "rgba(255, 255, 255, 0.05)",
                },
                transition: "all 0.2s ease",
              },
            },
          },
          MuiSvgIcon: {
            styleOverrides: {
              root: {
                color: "rgba(255, 255, 255, 0.8)",
                height: "20px",
                width: "20px",
              },
            },
          },
          MuiTableContainer: {
            styleOverrides: {
              root: {
                backgroundColor: "rgb(2, 8, 9)",
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
                background: "rgb(2, 8, 9)",
                backdropFilter: "blur(8px)",
                padding: "8px 12px",
                borderRadius: "6px",
                height: "fit-content",
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
          MuiCheckbox: {
            styleOverrides: {
              root: {
                borderRadius: "6px",
              },
            },
          },
          MuiList: {
            styleOverrides: {
              root: {
                padding: "4px 8px",
              },
            },
          },
          MuiMenuItem: {
            styleOverrides: {
              root: {
                fontSize: "14px !important",
                color: "rgba(255, 255, 255, 0.9) !important",
                padding: "8px 12px !important",
                borderRadius: "6px",
                margin: "4px 0px !important",
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
        muiPaginationProps={{
          rowsPerPageOptions: [10, 20, 50, 100],
          defaultValue: 20,
        }}
        muiTableContainerProps={{
          sx: {
            height: "calc(100% - 116px)",
            width: "100%",
          },
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

export default TaskTable;
