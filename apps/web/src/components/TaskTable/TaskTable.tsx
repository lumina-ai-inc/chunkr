import { useMemo, useState, useEffect } from "react";
import {
  MaterialReactTable,
  type MRT_ColumnDef,
  type MRT_PaginationState,
  type MRT_Row,
} from "material-react-table";
import { IconButton, Tooltip, createTheme, ThemeProvider } from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import { useNavigate } from "react-router-dom";
import { TaskResponse } from "../../models/taskResponse.model";
import { useTasksQuery } from "../../hooks/useTaskQuery";
import useUser from "../../hooks/useUser";
import "./TaskTable.css";
import { Flex } from "@radix-ui/themes";
import DeleteIcon from "@mui/icons-material/Delete";
import { deleteTasks } from "../../services/crudApi";
import toast from "react-hot-toast";
import { Box } from "@mui/material";
import { Status } from "../../models/taskResponse.model";

const TaskTable = () => {
  const navigate = useNavigate();
  const searchParams = new URLSearchParams(window.location.search);

  // Set default values if params don't exist
  if (!searchParams.has("tablePageIndex"))
    searchParams.set("tablePageIndex", "0");
  if (!searchParams.has("tablePageSize"))
    searchParams.set("tablePageSize", "20");

  // Update URL with default params if needed
  if (!window.location.search) {
    navigate(
      {
        search: searchParams.toString(),
      },
      { replace: true }
    );
  }

  const tablePageIndex = parseInt(searchParams.get("tablePageIndex") || "0");
  const tablePageSize = parseInt(searchParams.get("tablePageSize") || "20");

  const [pagination, setPagination] = useState<MRT_PaginationState>({
    pageIndex: tablePageIndex,
    pageSize: tablePageSize,
  });

  // Add effect to update URL when pagination changes
  useEffect(() => {
    const newParams = new URLSearchParams(window.location.search);
    newParams.set("tablePageIndex", pagination.pageIndex.toString());
    newParams.set("tablePageSize", pagination.pageSize.toString());

    navigate(
      {
        search: newParams.toString(),
      },
      { replace: true }
    );
  }, [pagination, navigate]);

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
      `/dashboard?taskId=${task.task_id}&pageCount=${task.page_count || 10}&tablePageIndex=${pagination.pageIndex}&tablePageSize=${pagination.pageSize}`
    );
  };

  console.log(tasks);

  const columns = useMemo<MRT_ColumnDef<TaskResponse>[]>(
    () => [
      {
        accessorKey: "file_name",
        header: "File Name",
        Cell: ({ cell }) => (
          <Tooltip arrow title={cell.getValue<string>()}>
            <div
              style={{
                maxWidth: "200px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
            >
              {cell.getValue<string>()}
            </div>
          </Tooltip>
        ),
      },
      {
        accessorKey: "task_id",
        header: "Task ID",
        Cell: ({ cell }) => {
          const fullId = cell.getValue<string>();
          return (
            <Tooltip arrow title={fullId}>
              <div>{fullId.substring(0, 8)}...</div>
            </Tooltip>
          );
        },
      },
      {
        accessorKey: "page_count",
        header: "Pages",
      },
      {
        accessorKey: "status",
        header: "Status",
        Cell: ({ row }) => (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor:
                  row.original.status === Status.Starting
                    ? "#3498db" // blue
                    : row.original.status === Status.Processing
                      ? "#f39c12" // orange
                      : row.original.status === Status.Failed
                        ? "#e74c3c" // red
                        : "transparent",
                display:
                  row.original.status === Status.Succeeded ? "none" : "block",
              }}
            />
            {row.original.status}
          </Box>
        ),
      },
      {
        accessorKey: "message",
        header: "Message",
        Cell: ({ cell }) => cell.getValue<string>(),
      },
      {
        accessorKey: "created_at",
        header: "Created At",
        Cell: ({ cell }) => new Date(cell.getValue<string>()).toLocaleString(),
      },
      {
        accessorKey: "finished_at",
        header: "Finished At",
        Cell: ({ cell }) => new Date(cell.getValue<string>()).toLocaleString(),
      },
      {
        accessorKey: "expires_at",
        header: "Expires At",
        Cell: ({ cell }) => {
          const dateValue = cell.getValue<string>();
          return dateValue && dateValue !== "Null"
            ? new Date(dateValue).toLocaleString()
            : "N/A";
        },
      },
    ],
    []
  );

  const renderDetailPanel = ({ row }: { row: MRT_Row<TaskResponse> }) => (
    <div
      style={{
        padding: "16px",
        borderRadius: "8px",
        backgroundColor: "rgb(255, 255, 255, 0.05)",
      }}
    >
      <pre
        style={{
          whiteSpace: "pre-wrap",
          wordBreak: "break-all",
          color: "rgba(255, 255, 255, 0.85)",
        }}
      >
        {JSON.stringify(row.original.configuration, null, 2)}
      </pre>
    </div>
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
                backgroundColor: "rgb(2, 8, 9) !important",
                border: "1px solid rgba(255, 255, 255, 0.1)",
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
                  alignContent: "center",
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
          MuiStack: {
            styleOverrides: {
              root: {
                color: "rgba(255, 255, 255, 0.95)",
              },
            },
          },
        },
      }),
    []
  );

  // Add this function to handle deletion of selected rows
  const handleDeleteSelected = async () => {
    const selectedTaskIds = Object.keys(rowSelection);
    if (selectedTaskIds.length === 0) return;

    try {
      setRowSelection({});
      await deleteTasks(selectedTaskIds);

      toast.success(
        `Successfully deleted ${selectedTaskIds.length} task${selectedTaskIds.length === 1 ? "" : "s"}`
      );

      refetch();
    } catch (error) {
      console.error("Error deleting tasks:", error);
      toast.error("Failed to delete tasks. Please try again.");
    }
  };

  // Add state for row selection
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});

  return (
    <Flex p="24px" direction="column" width="100%" height="100%">
      <ThemeProvider theme={tableTheme}>
        <MaterialReactTable
          columns={columns}
          data={tasks || []}
          enableColumnPinning
          enableRowSelection
          enablePagination
          manualPagination
          enableStickyHeader
          enableStickyFooter
          rowPinningDisplayMode="select-sticky"
          getRowId={(row) => row.task_id}
          muiPaginationProps={{
            rowsPerPageOptions: [10, 20, 50, 100],
            defaultValue: 20,
          }}
          muiTableContainerProps={{
            sx: {
              height: "calc(100% - 112px)",
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
            rowSelection,
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
            <Flex gap="2">
              <Tooltip arrow title="Refresh Data">
                <IconButton onClick={() => refetch()}>
                  <RefreshIcon />
                </IconButton>
              </Tooltip>
              {Object.keys(rowSelection).length > 0 && (
                <Tooltip arrow title="Delete Selected">
                  <IconButton color="error" onClick={handleDeleteSelected}>
                    <DeleteIcon />
                  </IconButton>
                </Tooltip>
              )}
            </Flex>
          )}
          enableExpanding
          renderDetailPanel={renderDetailPanel}
          muiTableBodyRowProps={({ row }) => ({
            onClick: (event) => {
              if (
                row.original.message === "Task succeeded" &&
                !(event.target as HTMLElement)
                  .closest(".MuiTableCell-root")
                  ?.classList.contains("MuiTableCell-paddingNone")
              ) {
                handleTaskClick(row.original);
              }
            },
            sx: {
              cursor:
                row.original.message === "Task succeeded"
                  ? "pointer"
                  : "default",
              "&:hover": {
                backgroundColor:
                  row.original.message === "Task succeeded"
                    ? "rgba(255, 255, 255, 0.05) !important"
                    : "rgb(2, 8, 9) !important",
              },
              "&.Mui-TableBodyCell-DetailPanel": {
                height: 0,
                "& > td": {
                  padding: 0,
                },
              },
            },
          })}
          onRowSelectionChange={setRowSelection}
        />
      </ThemeProvider>
    </Flex>
  );
};

export default TaskTable;
