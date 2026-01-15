import { useMemo, useState, useEffect } from "react";
import {
  MaterialReactTable,
  type MRT_ColumnDef,
  type MRT_PaginationState,
  // type MRT_Row, // Reserved for future detail panel
} from "material-react-table";
import { IconButton, Tooltip, createTheme, ThemeProvider, Button, Dialog, DialogTitle, DialogContent, DialogActions } from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
// import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import FolderIcon from "@mui/icons-material/Folder";
import DescriptionIcon from "@mui/icons-material/Description";
import AccountBalanceIcon from "@mui/icons-material/AccountBalance";
import AssessmentIcon from "@mui/icons-material/Assessment";
import { useSearchParams } from "react-router-dom";
import { TaskResponse } from "../../models/taskResponse.model";
import { useTasksQuery } from "../../hooks/useTaskQuery";
import useUser from "../../hooks/useUser";
import { useAuth } from "react-oidc-context";
import "./TaskTable.css";
import { Flex, Text } from "@radix-ui/themes";
import { deleteTasks, cancelTasks, updateTask } from "../../services/crudApi";
import toast from "react-hot-toast";
import { Box } from "@mui/material";
import { Status } from "../../models/taskResponse.model";
import BetterButton from "../BetterButton/BetterButton";
// import ReactJson from "react-json-view"; // Reserved for future detail panel
import UploadDialog from "../Upload/UploadDialog";
import Loader from "../../pages/Loader/Loader";

// Extended interface for folder-like structure
interface PackageItem {
  id: string;
  type: 'package' | 'files' | 'records' | 'credit-risk';
  packageName?: string;
  task?: TaskResponse;
  isExpanded?: boolean;
  parentId?: string;
}

const TaskTable = ({ context = "extracts" }: { context?: "extracts" | "flows" }) => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [rowSelection, setRowSelection] = useState<Record<string, boolean>>({});
  const [expandedPackages, setExpandedPackages] = useState<Record<string, boolean>>({});
  const [showCreditRiskGraph, setShowCreditRiskGraph] = useState<string | null>(null);

  // Set default values if params don't exist
  if (!searchParams.has("tablePageIndex"))
    searchParams.set("tablePageIndex", "0");
  if (!searchParams.has("tablePageSize"))
    searchParams.set("tablePageSize", "20");

  // Update URL with default params if needed
  useEffect(() => {
    let madeChange = false;
    if (!searchParams.has("tablePageIndex")) {
      searchParams.set("tablePageIndex", "0");
      madeChange = true;
    }
    if (!searchParams.has("tablePageSize")) {
      searchParams.set("tablePageSize", "20");
      madeChange = true;
    }
    if (madeChange) {
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, setSearchParams]);

  const tablePageIndex = parseInt(searchParams.get("tablePageIndex") || "0");
  const tablePageSize = parseInt(searchParams.get("tablePageSize") || "20");

  const [pagination, setPagination] = useState<MRT_PaginationState>({
    pageIndex: tablePageIndex,
    pageSize: tablePageSize,
  });

  // Add effect to update URL when pagination changes
  useEffect(() => {
    const newParams = new URLSearchParams(searchParams);
    newParams.set("tablePageIndex", pagination.pageIndex.toString());
    newParams.set("tablePageSize", pagination.pageSize.toString());
    setSearchParams(newParams, { replace: true });
  }, [pagination, setSearchParams]);

  const { data: user } = useUser();
  const auth = useAuth();
  const totalTasks = user?.task_count || 0;

  const {
    data: tasks,
    isError,
    isRefetching,
    isLoading,
    refetch,
  } = useTasksQuery(pagination.pageIndex + 1, pagination.pageSize);

  // Transform tasks into hierarchical package structure
  const packageItems = useMemo(() => {
    if (!tasks) return [];
    
    const items: PackageItem[] = [];
    
    tasks.forEach((task) => {
      let packageName = task.output?.file_name || `Package ${task.task_id.substring(0, 8)}`;
      // Trim file extension from package name
      const lastDotIndex = packageName.lastIndexOf('.');
      if (lastDotIndex > 0) {
        packageName = packageName.substring(0, lastDotIndex);
      }
      const packageId = `package-${task.task_id}`;
      
      // Add package row
      items.push({
        id: packageId,
        type: 'package',
        packageName,
        task,
        isExpanded: expandedPackages[packageId] || false,
      });
      
      // Add sub-items if package is expanded
      if (expandedPackages[packageId]) {
        items.push({
          id: `${packageId}-files`,
          type: 'files',
          task,
          parentId: packageId,
        });
        
        items.push({
          id: `${packageId}-records`,
          type: 'records',
          task,
          parentId: packageId,
        });
        
        items.push({
          id: `${packageId}-credit-risk`,
          type: 'credit-risk',
          task,
          parentId: packageId,
        });
      }
    });
    
    return items;
  }, [tasks, expandedPackages]);

  const togglePackage = (packageId: string) => {
    setExpandedPackages(prev => ({
      ...prev,
      [packageId]: !prev[packageId]
    }));
  };

  const handleTaskClick = (task: TaskResponse) => {
    const newParams = new URLSearchParams(searchParams);
    if (context === "flows") {
      newParams.set("view", "flows");
    }
    newParams.set("taskId", task.task_id);

    if (context === "extracts") {
      newParams.set("pageCount", (task.output?.page_count || 10).toString());
    }
    
    setSearchParams(newParams, { replace: true });
  };

  const columns = useMemo<MRT_ColumnDef<PackageItem>[]>(
    () => [
      {
        accessorKey: "packageName",
        header: context === "flows" ? "Flow Name" : "Name",
        Cell: ({ row }) => {
          const item = row.original;
          const paddingLeft = item.type === 'package' ? 0 : 32;
          
          if (item.type === 'package') {
            return (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation();
                    togglePackage(item.id);
                  }}
                  sx={{ 
                    p: 0.5,
                    transform: item.isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s ease'
                  }}
                >
                  <ChevronRightIcon />
                </IconButton>
                <FolderIcon sx={{ color: "#f39c12", fontSize: 20 }} />
                <Tooltip arrow title={item.packageName}>
                  <div
                    style={{
                      maxWidth: "200px",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {item.packageName}
                  </div>
                </Tooltip>
              </Box>
            );
          } else if (item.type === 'files') {
            return (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, pl: `${paddingLeft}px` }}>
                <DescriptionIcon className="sub-item-icon" />
                <Text size="2">Borrower Files</Text>
              </Box>
            );
          } else if (item.type === 'records') {
            return (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, pl: `${paddingLeft}px` }}>
                <AccountBalanceIcon className="sub-item-icon" />
                <Text size="2">Transaction Records</Text>
              </Box>
            );
          } else if (item.type === 'credit-risk') {
            return (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1, pl: `${paddingLeft}px` }}>
                <AssessmentIcon className="sub-item-icon" />
                <Text size="2">Credit Risk Models</Text>
              </Box>
            );
          }
          return null;
        },
      },
      {
        accessorKey: "task.task?.task_id",
        header: context === "flows" ? "Flow ID" : "Package Id",
        Cell: ({ row }) => {
          const item = row.original;
          if (item.type === 'package' && item.task) {
            const fullId = item.task.task_id;
            return (
              <Tooltip arrow title={fullId}>
                <div>{fullId.substring(0, 8)}...</div>
              </Tooltip>
            );
          }
          return item.type === 'package' ? null : '-';
        },
      },
      {
        accessorKey: "task.output.page_count",
        header: context === "flows" ? "Components" : "Pages",
        Cell: ({ row }) => {
          const item = row.original;
          if (item.type === 'package') {
            if (context === "flows") {
              return 8;
            }
            return item.task?.output?.page_count || 0;
          } else if (item.type === 'files') {
            return item.task?.output?.page_count || 0;
          } else if (item.type === 'records') {
            return 'Integration Records';
          } else if (item.type === 'credit-risk') {
            return 'AI Risk Analysis';
          }
          return null;
        },
      },
      {
        accessorKey: "task.status",
        header: "Status",
        Cell: ({ row }) => {
          const item = row.original;
          if (item.type === 'package' && item.task) {
            return (
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Box
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    backgroundColor:
                      item.task.status === Status.Starting
                        ? "#3498db" // blue
                        : item.task.status === Status.Processing
                        ? "#f39c12" // orange
                        : item.task.status === Status.Failed
                        ? "#e74c3c" // red
                        : "transparent",
                    display:
                      item.task.status === Status.Succeeded ? "none" : "block",
                  }}
                />
                {item.task.status}
              </Box>
            );
          } else if (item.type === 'files') {
            return (
              <Button
                variant="contained"
                size="small"
                sx={{
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  fontSize: '11px',
                  padding: '4px 12px',
                  minWidth: '70px',
                  textTransform: 'none',
                  '&:hover': {
                    backgroundColor: '#45a049'
                  }
                }}
              >
                Verify
              </Button>
            );
          } else if (item.type === 'records') {
            return (
              <Button
                variant="contained"
                size="small"
                sx={{
                  backgroundColor: '#2196F3',
                  color: 'white',
                  fontSize: '11px',
                  padding: '4px 12px',
                  minWidth: '70px',
                  textTransform: 'none',
                  '&:hover': {
                    backgroundColor: '#1976D2'
                  }
                }}
              >
                Live
              </Button>
            );
          } else if (item.type === 'credit-risk') {
            return (
              <Button
                variant="contained"
                size="small"
                sx={{
                  backgroundColor: '#FF9800',
                  color: 'white',
                  fontSize: '11px',
                  padding: '4px 8px',
                  minWidth: '70px',
                  textTransform: 'none',
                  '&:hover': {
                    backgroundColor: '#F57C00'
                  }
                }}
              >
                Score
              </Button>
            );
          }
          return null;
        },
      },
      {
        accessorKey: "task.created_at",
        header: "Created At",
        Cell: ({ row }) => {
          const item = row.original;
          if (item.type === 'package' && item.task) {
            return new Date(item.task.created_at).toLocaleString();
          }
          return item.type === 'package' ? null : '-';
        },
      },
      {
        accessorKey: "task.finished_at",
        header: "Finished At",
        Cell: ({ row }) => {
          const item = row.original;
          if (item.type === 'package' && item.task) {
            const dateValue = item.task.finished_at;
            return (item.task.status === Status.Succeeded ||
              item.task.status === Status.Failed) &&
              dateValue
              ? new Date(dateValue).toLocaleString()
              : "N/A";
          }
          return item.type === 'package' ? null : '-';
        },
      },
      {
        accessorKey: "task.expires_at",
        header: "Expires At",
        Cell: ({ row }) => {
          const item = row.original;
          if (item.type === 'package' && item.task) {
            const dateValue = item.task.expires_at;
            return dateValue && dateValue !== "Null"
              ? new Date(dateValue).toLocaleString()
              : "N/A";
          }
          return item.type === 'package' ? null : '-';
        },
      },
    ],
    [context, togglePackage]
  );

  // Reserved for future detail expansion
  /*
  const renderDetailPanel = ({ row }: { row: MRT_Row<PackageItem> }) => {
    const item = row.original;
    if (item.type === 'package' && item.task) {
      return (
        <div
          style={{
            padding: "16px",
            backgroundColor: "rgb(255, 255, 255, 0.05)",
          }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          <ReactJson
            src={item.task.configuration}
            theme="monokai"
            displayDataTypes={false}
            enableClipboard={false}
            style={{ backgroundColor: "transparent" }}
            collapsed={1}
            name={false}
          />
        </div>
      );
    } else if (item.type === 'records') {
      return (
        <div
          style={{
            padding: "16px",
            backgroundColor: "rgb(255, 255, 255, 0.05)",
          }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          <Text size="2" style={{ color: "#666" }}>
            Integration records pulled from finance systems would be displayed here.
            This includes data from selected providers like Plaid, Experian, Pinwheel, etc.
          </Text>
        </div>
      );
    } else if (item.type === 'credit-risk') {
      return (
        <div
          style={{
            padding: "16px",
            backgroundColor: "rgb(255, 255, 255, 0.05)",
          }}
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
        >
          <Text size="2" style={{ color: "#666" }}>
            Credit risk models and analytics would be displayed here.
            This includes risk scores, probability models, and credit assessment data.
          </Text>
        </div>
      );
    }
    return null;
  };
  */

  const tableTheme = useMemo(
    () =>
      createTheme({
        palette: {
          mode: "light",
          primary: {
            main: "#545454",
          },
          info: {
            main: "#545454",
          },
          background: {
            default: "#ffffff",
            paper: "#ffffff",
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
                backgroundColor: "#ffffff !important",
                border: "none",
              },
            },
          },
          MuiTableHead: {
            styleOverrides: {
              root: {
                "& .MuiTableCell-head": {
                  color: "#545454",
                  backgroundColor: "#ffffff",
                  fontSize: "16px",
                  fontWeight: 600,
                  borderBottom: "1px solid rgba(84, 84, 84, 0.1)",
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
                  backgroundColor: "#ffffff !important",
                  "&:hover": {
                    backgroundColor: "#f8f8f8 !important",
                    backdropFilter: "blur(8px)",
                  },
                  transition: "all 0.2s ease",
                },
                "& .MuiTableCell-body": {
                  color: "#545454",
                  fontSize: "14px",
                  padding: "16px 14px",
                  borderBottom: "1px solid rgba(84, 84, 84, 0.1)",
                },
              },
            },
          },
          MuiIconButton: {
            styleOverrides: {
              root: {
                color: "#545454",
                padding: "8px",
                borderRadius: "6px",
                backgroundColor: "#545454",
                "&:hover": {
                  backgroundColor: "#545454",
                },
                transition: "all 0.2s ease",
              },
            },
          },
          MuiSvgIcon: {
            styleOverrides: {
              root: {
                color: "#222",
                height: "20px",
                width: "20px",
                "&.sub-item-icon": {
                  color: "#000000 !important",
                  fontSize: "18px !important"
                }
              },
            },
          },
          MuiTableContainer: {
            styleOverrides: {
              root: {
                backgroundColor: "#ffffff",
              },
            },
          },
          MuiTablePagination: {
            styleOverrides: {
              root: {
                color: "#545454",
                "& .MuiTablePagination-select": {
                  color: "#545454",
                },
                "& .MuiTablePagination-selectIcon": {
                  color: "#545454",
                },
                "& .MuiIconButton-root": {
                  backgroundColor: "#545454",
                  color: "#ffffff",
                  "&:hover": {
                    backgroundColor: "#545454",
                  },
                },
                "& .MuiIconButton-root .MuiSvgIcon-root": {
                  color: "#ffffff",
                },
              },
              select: {
                color: "#545454",
              },
              selectIcon: {
                color: "#545454",
              },
            },
          },
          MuiTooltip: {
            styleOverrides: {
              tooltip: {
                fontSize: "14px",
                background: "#545454",
                backdropFilter: "blur(8px)",
                padding: "8px 12px",
                borderRadius: "6px",
                height: "fit-content",
                color: "#ffffff",
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
                color: "#222",
                '&.Mui-checked': {
                  color: "#222",
                },
                '& .MuiSvgIcon-root': {
                  color: "#222",
                },
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
                color: "#545454 !important",
                padding: "8px 12px !important",
                borderRadius: "6px",
                margin: "4px 0px !important",
                "&:hover": {
                  backgroundColor: "rgba(84, 84, 84, 0.05)",
                },
              },
            },
          },
          MuiStack: {
            styleOverrides: {
              root: {
                color: "#545454",
              },
            },
          },
          MuiPopover: {
            styleOverrides: {
              paper: {
                backgroundColor: "#ffffff !important",
                border: "none",
              },
            },
          },
          MuiMenu: {
            styleOverrides: {
              paper: {
                backgroundColor: "#ffffff !important",
                border: "none",
              },
            },
          },
        },
      }),
    []
  );

  const handleDeleteSelected = async () => {
    const selectedItemIds = Object.keys(rowSelection);
    if (selectedItemIds.length === 0) return;

    const deletableTaskIds = selectedItemIds
      .map((itemId) => {
        const item = packageItems.find((item) => item.id === itemId);
        return item?.type === 'package' && item.task ? item.task.task_id : null;
      })
      .filter((taskId): taskId is string => taskId !== null)
      .filter((taskId) => {
        const task = tasks?.find((t) => t.task_id === taskId);
        return (
          task?.status === Status.Succeeded ||
          task?.status === Status.Failed ||
          task?.status === Status.Cancelled
        );
      });

    if (deletableTaskIds.length === 0) {
      toast.error("Only Succeeded, Failed, or Cancelled packages can be deleted.");
      return;
    }

    const nonDeletableCount = selectedItemIds.length - deletableTaskIds.length;

    try {
      setRowSelection({});
      await deleteTasks(deletableTaskIds);

      toast.success(
        `Successfully deleted ${deletableTaskIds.length} package${
          deletableTaskIds.length === 1 ? "" : "s"
        }` +
          (nonDeletableCount > 0
            ? `. ${nonDeletableCount} item${
                nonDeletableCount === 1 ? " was" : "s were"
              } skipped as ${
                nonDeletableCount === 1 ? "it wasn't" : "they weren't"
              } a completed package.`
            : "")
      );

      refetch();
    } catch (error) {
      console.error("Error deleting packages:", error);
      toast.error("Failed to delete packages. Please try again.");
    }
  };

  const handleCancelSelected = async () => {
    const selectedItemIds = Object.keys(rowSelection);
    if (selectedItemIds.length === 0) return;

    const cancellableTaskIds = selectedItemIds
      .map((itemId) => {
        const item = packageItems.find((item) => item.id === itemId);
        return item?.type === 'package' && item.task ? item.task.task_id : null;
      })
      .filter((taskId): taskId is string => taskId !== null)
      .filter((taskId) => {
        const task = tasks?.find((t) => t.task_id === taskId);
        return task?.status === Status.Starting;
      });

    const nonCancellableTaskDetails = selectedItemIds
      .map((itemId) => {
        const item = packageItems.find((item) => item.id === itemId);
        return item?.type === 'package' && item.task ? item.task.task_id : null;
      })
      .filter((taskId): taskId is string => taskId !== null)
      .filter((taskId) => !cancellableTaskIds.includes(taskId))
      .map((taskId) => {
        const task = tasks?.find((t) => t.task_id === taskId);
        return task?.status === Status.Processing ? "processing" : "completed";
      });

    if (cancellableTaskIds.length === 0) {
      const statusMessage =
        nonCancellableTaskDetails[0] === "processing"
          ? "already processing"
          : "already completed";
      toast.error(
        `Cannot cancel ${
          selectedItemIds.length === 1 ? "this package" : "these packages"
        } as ${
          selectedItemIds.length === 1 ? "it is" : "they are"
        } ${statusMessage}.`
      );
      return;
    }

    try {
      await cancelTasks(cancellableTaskIds);
      setRowSelection({});

      toast.success(
        `Successfully cancelled ${cancellableTaskIds.length} package${
          cancellableTaskIds.length === 1 ? "" : "s"
        }` +
          (nonCancellableTaskDetails.length > 0
            ? `. ${nonCancellableTaskDetails.length} package${
                nonCancellableTaskDetails.length === 1 ? " was" : "s were"
              } skipped as ${
                nonCancellableTaskDetails.length === 1 ? "it was" : "they were"
              } already ${nonCancellableTaskDetails[0]}.`
            : "")
      );

      refetch();
    } catch (error) {
      console.error("Error cancelling packages:", error);
      toast.error("Failed to cancel packages. Please try again.");
    }
  };

  const handleRetrySelected = async () => {
    const selectedItemIds = Object.keys(rowSelection);
    if (selectedItemIds.length === 0) return;

    const retryableTaskIds = selectedItemIds
      .map((itemId) => {
        const item = packageItems.find((item) => item.id === itemId);
        return item?.type === 'package' && item.task ? item.task.task_id : null;
      })
      .filter((taskId): taskId is string => taskId !== null)
      .filter((taskId) => {
        const task = tasks?.find((t) => t.task_id === taskId);
        return task?.status === Status.Failed;
      });

    if (retryableTaskIds.length === 0) {
      toast.error("No packages can be retried. Only Failed packages can be retried.");
      return;
    }

    const nonRetryableCount = selectedItemIds.length - retryableTaskIds.length;

    try {
      setRowSelection({});
      // Call updateTask for each failed task with an empty object to trigger retry
      const retryPromises = retryableTaskIds.map((taskId) =>
        updateTask(taskId, {})
      );
      await Promise.all(retryPromises);

      toast.success(
        `Successfully initiated retry for ${retryableTaskIds.length} package${
          retryableTaskIds.length === 1 ? "" : "s"
        }` +
          (nonRetryableCount > 0
            ? `. ${nonRetryableCount} item${
                nonRetryableCount === 1 ? " was" : "s were"
              } skipped as ${
                nonRetryableCount === 1 ? "it wasn't" : "they weren't"
              } a Failed package.`
            : "")
      );

      refetch();
    } catch (error) {
      console.error("Error retrying packages:", error);
      toast.error("Failed to retry packages. Please try again.");
    }
  };

  const hasSelectedCancellableTasks = () => {
    return Object.keys(rowSelection).some((itemId) => {
      const item = packageItems.find((item) => item.id === itemId);
      if (item?.type === 'package' && item.task) {
        const task = tasks?.find((t) => t.task_id === item.task?.task_id);
        return task?.status === Status.Starting;
      }
      return false;
    });
  };

  const hasSelectedFailedTasks = () => {
    return Object.keys(rowSelection).some((itemId) => {
      const item = packageItems.find((item) => item.id === itemId);
      if (item?.type === 'package' && item.task) {
        const task = tasks?.find((t) => t.task_id === item.task?.task_id);
        return task?.status === Status.Failed;
      }
      return false;
    });
  };

  return (
    <Flex
      p="24px"
      direction="column"
      width="100%"
      height="100%"
      className="task-table-container"
    >
      {isLoading ? (
        <Flex
          width="100%"
          height="100%"
          align="center"
          justify="center"
          direction="column"
          gap="4"
        >
          <Loader />
        </Flex>
      ) : !tasks || tasks.length === 0 ? (
        <Flex
          width="90%"
          height="100%"
          align="center"
          justify="center"
          direction="column"
          gap="4"
        >
          <Flex
            width="85%"
            height="85%"
            style={{
              backgroundImage: "url('/pipeline-background.png')",
              backgroundSize: "cover",
              backgroundPosition: "left",
              backgroundRepeat: "no-repeat"
            }}
          />
          <UploadDialog
            auth={auth}
            onUploadComplete={() => {
              refetch();
            }}
          />
        </Flex>
      ) : (
        <ThemeProvider theme={tableTheme}>
          <MaterialReactTable
            columns={columns}
            data={packageItems}
            enableColumnPinning
            enableRowSelection
            enablePagination
            manualPagination
            enableStickyHeader
            enableStickyFooter
            rowPinningDisplayMode="select-sticky"
            getRowId={(row) => row.task?.task_id || row.id}
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
                  <>
                    {hasSelectedCancellableTasks() && (
                      <Tooltip arrow title="Cancel Selected">
                        <BetterButton onClick={handleCancelSelected}>
                          <svg
                            width="20"
                            height="20"
                            viewBox="0 0 25 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <g clipPath="url(#clip0_113_1439)">
                                                          <path
                              d="M9.25 15.25L15.75 8.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <path
                              d="M15.75 15.25L9.25 8.75"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            <path
                              d="M12.5 21.25C17.6086 21.25 21.75 17.1086 21.75 12C21.75 6.89137 17.6086 2.75 12.5 2.75C7.39137 2.75 3.25 6.89137 3.25 12C3.25 17.1086 7.39137 21.25 12.5 21.25Z"
                              stroke="#545454"
                              strokeWidth="1.5"
                              strokeMiterlimit="10"
                              strokeLinecap="round"
                            />
                            </g>
                            <defs>
                              <clipPath id="clip0_113_1439">
                                <rect
                                  width="24"
                                  height="24"
                                  fill="white"
                                  transform="translate(0.5)"
                                />
                              </clipPath>
                            </defs>
                          </svg>
                          <Text size="1">Cancel Packages</Text>
                        </BetterButton>
                      </Tooltip>
                    )}
                    {hasSelectedFailedTasks() && (
                      <Tooltip arrow title="Retry Selected Failed Packages">
                        <BetterButton onClick={handleRetrySelected}>
                          <svg
                            width="20"
                            height="20"
                            viewBox="0 0 25 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <g clipPath="url(#clip0_113_1417)">
                              <path
                                d="M16.25 15.25H20.75V19.75"
                                stroke="#545454"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              <path
                                d="M8.75 8.25H4.25V3.75"
                                stroke="#545454"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                              />
                              <path
                                d="M4.31523 8.25135C5.73695 5.15233 8.86701 3 12.4998 3C17.2006 3 21.0597 6.65439 21.4647 11.25M20.7508 15.6003C19.3619 18.7788 16.1902 21 12.4998 21C7.78048 21 3.90956 17.3676 3.53027 12.7462"
                                stroke="#545454"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                              />
                            </g>
                            <defs>
                              <clipPath id="clip0_113_1417">
                                <rect
                                  width="24"
                                  height="24"
                                  fill="white"
                                  transform="translate(0.5)"
                                />
                              </clipPath>
                            </defs>
                          </svg>

                          <Text size="1">Retry Package Creation</Text>
                        </BetterButton>
                      </Tooltip>
                    )}
                    <Tooltip arrow title="Delete Selected">
                      <BetterButton onClick={handleDeleteSelected}>
                        <svg
                          width="20"
                          height="20"
                          viewBox="0 0 25 24"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M4.25 4.75H20.75"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeMiterlimit="10"
                            strokeLinecap="round"
                          />
                          <path
                            d="M12.5 2.75V4.75"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeMiterlimit="10"
                            strokeLinecap="round"
                          />
                          <path
                            d="M14.2402 17.27V12.77"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeMiterlimit="10"
                            strokeLinecap="round"
                          />
                          <path
                            d="M10.75 17.25V12.75"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeMiterlimit="10"
                            strokeLinecap="round"
                          />
                          <path
                            d="M5.87012 8.75H19.1701"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeMiterlimit="10"
                          />
                          <path
                            d="M15.91 21.25H9.09C8.07 21.25 7.21 20.48 7.1 19.47L5.5 4.75H19.5L17.9 19.47C17.79 20.48 16.93 21.25 15.91 21.25V21.25Z"
                            stroke="#545454"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                        </svg>{" "}
                        <Text size="1">Delete Package</Text>
                      </BetterButton>
                    </Tooltip>
                  </>
                )}
              </Flex>
            )}
            enableExpanding={false}
            muiTableBodyRowProps={({ row }) => {
              const item = row.original;
              const isClickable = 
                item.type === 'package' ||
                (item.type === 'files' && item.task?.message === "Task succeeded") ||
                item.type === 'records' ||
                item.type === 'credit-risk';

              return {
                onClick: (event) => {
                  if (
                    !(event.target as HTMLElement)
                      .closest(".MuiTableCell-root")
                      ?.classList.contains("MuiTableCell-paddingNone")
                  ) {
                    if (item.type === 'package') {
                      // Toggle expansion for package rows
                      togglePackage(item.id);
                    } else if (item.type === 'files' && item.task) {
                      // Navigate to Document Extract for Files
                      if (context === "flows" || item.task.message === "Task succeeded") {
                        handleTaskClick(item.task);
                      } else if (
                        item.task.status !== Status.Failed &&
                        item.task.status !== Status.Succeeded
                      ) {
                        refetch();
                      }
                    } else if (item.type === 'records') {
                      // Show records detail panel
                      console.log('Show integration records from finance systems');
                    } else if (item.type === 'credit-risk') {
                      // Show credit risk graph
                      setShowCreditRiskGraph(item.task?.task_id || 'default');
                    }
                  }
                },
                sx: {
                  cursor: isClickable ? "pointer" : "default",
                  backgroundColor: item.type !== 'package' ? 'rgba(0,0,0,0.02)' : 'inherit',
                  "&:hover": {
                    backgroundColor: isClickable
                      ? "#f8f8f8 !important"
                      : "#ffffff !important",
                  },
                  "&.Mui-TableBodyCell-DetailPanel": {
                    height: 0,
                    "& > td": {
                      padding: 0,
                    },
                  },
                },
              };
            }}
            onRowSelectionChange={setRowSelection}
          />
        </ThemeProvider>
      )}
      
      {/* Credit Risk Graph Dialog */}
      <Dialog
        open={showCreditRiskGraph !== null}
        onClose={() => setShowCreditRiskGraph(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Credit Risk Analysis</DialogTitle>
        <DialogContent>
          <div style={{ padding: '20px', textAlign: 'center' }}>
            {/* Home-Equity Credit Risk Heat Map Dashboard */}
            <svg width="1000" height="700" viewBox="0 0 1000 700" style={{ border: '1px solid #e0e0e0', borderRadius: '8px', backgroundColor: '#f8f9fa' }}>
              <defs>
                {/* Risk Level Gradients */}
                <linearGradient id="criticalRisk" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#dc2626" stopOpacity="0.9"/>
                  <stop offset="100%" stopColor="#b91c1c" stopOpacity="0.8"/>
                </linearGradient>
                <linearGradient id="highRisk" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#ea580c" stopOpacity="0.9"/>
                  <stop offset="100%" stopColor="#c2410c" stopOpacity="0.8"/>
                </linearGradient>
                <linearGradient id="mediumRisk" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#eab308" stopOpacity="0.9"/>
                  <stop offset="100%" stopColor="#ca8a04" stopOpacity="0.8"/>
                </linearGradient>
                <linearGradient id="lowRisk" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#16a34a" stopOpacity="0.9"/>
                  <stop offset="100%" stopColor="#15803d" stopOpacity="0.8"/>
                </linearGradient>
                <linearGradient id="minimalRisk" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#0284c7" stopOpacity="0.9"/>
                  <stop offset="100%" stopColor="#0369a1" stopOpacity="0.8"/>
                </linearGradient>
                
                {/* Shadow Filter */}
                <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                  <feDropShadow dx="2" dy="2" stdDeviation="2" floodColor="black" floodOpacity="0.2"/>
                </filter>
              </defs>
              
              {/* Title */}
              <text x="500" y="35" textAnchor="middle" fontSize="22" fill="#1f2937" fontWeight="bold">
                Home-Equity Credit Risk Assessment Matrix
              </text>
              <text x="500" y="55" textAnchor="middle" fontSize="14" fill="#6b7280">
                HELOC Underwriting Dashboard - LTV vs DTI Risk Analysis
              </text>
              
              {/* Main Heat Map Grid */}
              <g transform="translate(80,80)">
                {/* Y-axis Label */}
                <text x="-40" y="200" fontSize="14" fill="#374151" fontWeight="bold" transform="rotate(-90, -40, 200)">
                  Debt-to-Income Ratio (DTI %)
                </text>
                
                {/* X-axis Label */}
                <text x="350" y="450" fontSize="14" fill="#374151" fontWeight="bold" textAnchor="middle">
                  Loan-to-Value Ratio (LTV %)
                </text>
                
                {/* DTI Labels (Y-axis) */}
                <text x="-15" y="75" fontSize="12" fill="#6b7280" textAnchor="end">50%+</text>
                <text x="-15" y="145" fontSize="12" fill="#6b7280" textAnchor="end">43%</text>
                <text x="-15" y="215" fontSize="12" fill="#6b7280" textAnchor="end">36%</text>
                <text x="-15" y="285" fontSize="12" fill="#6b7280" textAnchor="end">28%</text>
                <text x="-15" y="355" fontSize="12" fill="#6b7280" textAnchor="end">≤20%</text>
                
                {/* LTV Labels (X-axis) */}
                <text x="70" y="430" fontSize="12" fill="#6b7280" textAnchor="middle">≤60%</text>
                <text x="175" y="430" fontSize="12" fill="#6b7280" textAnchor="middle">70%</text>
                <text x="280" y="430" fontSize="12" fill="#6b7280" textAnchor="middle">80%</text>
                <text x="385" y="430" fontSize="12" fill="#6b7280" textAnchor="middle">85%</text>
                <text x="490" y="430" fontSize="12" fill="#6b7280" textAnchor="middle">90%+</text>
                
                {/* Heat Map Cells */}
                {/* Row 1: DTI 50%+ */}
                <rect x="35" y="50" width="70" height="50" fill="url(#lowRisk)" filter="url(#shadow)" rx="4"/>
                <text x="70" y="80" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">8.2%</text>
                
                <rect x="140" y="50" width="70" height="50" fill="url(#mediumRisk)" filter="url(#shadow)" rx="4"/>
                <text x="175" y="80" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">12.4%</text>
                
                <rect x="245" y="50" width="70" height="50" fill="url(#highRisk)" filter="url(#shadow)" rx="4"/>
                <text x="280" y="80" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">18.7%</text>
                
                <rect x="350" y="50" width="70" height="50" fill="url(#criticalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="385" y="80" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">28.9%</text>
                
                <rect x="455" y="50" width="70" height="50" fill="url(#criticalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="490" y="80" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">35.2%</text>
                
                {/* Row 2: DTI 43% */}
                <rect x="35" y="120" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="70" y="150" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">4.8%</text>
                
                <rect x="140" y="120" width="70" height="50" fill="url(#lowRisk)" filter="url(#shadow)" rx="4"/>
                <text x="175" y="150" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">7.1%</text>
                
                <rect x="245" y="120" width="70" height="50" fill="url(#mediumRisk)" filter="url(#shadow)" rx="4"/>
                <text x="280" y="150" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">11.3%</text>
                
                <rect x="350" y="120" width="70" height="50" fill="url(#highRisk)" filter="url(#shadow)" rx="4"/>
                <text x="385" y="150" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">19.6%</text>
                
                <rect x="455" y="120" width="70" height="50" fill="url(#criticalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="490" y="150" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">31.4%</text>
                
                {/* Row 3: DTI 36% */}
                <rect x="35" y="190" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="70" y="220" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">3.2%</text>
                
                <rect x="140" y="190" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="175" y="220" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">4.9%</text>
                
                <rect x="245" y="190" width="70" height="50" fill="url(#lowRisk)" filter="url(#shadow)" rx="4"/>
                <text x="280" y="220" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">8.1%</text>
                
                <rect x="350" y="190" width="70" height="50" fill="url(#mediumRisk)" filter="url(#shadow)" rx="4"/>
                <text x="385" y="220" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">14.2%</text>
                
                <rect x="455" y="190" width="70" height="50" fill="url(#highRisk)" filter="url(#shadow)" rx="4"/>
                <text x="490" y="220" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">22.8%</text>
                
                {/* Row 4: DTI 28% */}
                <rect x="35" y="260" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="70" y="290" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">2.1%</text>
                
                <rect x="140" y="260" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="175" y="290" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">3.4%</text>
                
                <rect x="245" y="260" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="280" y="290" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">5.7%</text>
                
                <rect x="350" y="260" width="70" height="50" fill="url(#lowRisk)" filter="url(#shadow)" rx="4"/>
                <text x="385" y="290" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">9.8%</text>
                
                <rect x="455" y="260" width="70" height="50" fill="url(#mediumRisk)" filter="url(#shadow)" rx="4"/>
                <text x="490" y="290" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">16.3%</text>
                
                {/* Row 5: DTI ≤20% */}
                <rect x="35" y="330" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="70" y="360" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">1.2%</text>
                
                <rect x="140" y="330" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="175" y="360" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">1.8%</text>
                
                <rect x="245" y="330" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="280" y="360" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">2.9%</text>
                
                <rect x="350" y="330" width="70" height="50" fill="url(#minimalRisk)" filter="url(#shadow)" rx="4"/>
                <text x="385" y="360" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">4.6%</text>
                
                <rect x="455" y="330" width="70" height="50" fill="url(#lowRisk)" filter="url(#shadow)" rx="4"/>
                <text x="490" y="360" fontSize="11" fill="white" fontWeight="bold" textAnchor="middle">7.4%</text>
              </g>
              
              {/* Legend */}
              <g transform="translate(650,120)">
                <rect x="0" y="0" width="250" height="220" fill="white" stroke="#d1d5db" strokeWidth="1" rx="8"/>
                <text x="125" y="25" textAnchor="middle" fontSize="16" fill="#1f2937" fontWeight="bold">Risk Assessment Scale</text>
                
                <rect x="20" y="45" width="25" height="15" fill="url(#criticalRisk)" rx="2"/>
                <text x="55" y="57" fontSize="12" fill="#1f2937">Critical Risk (25%+)</text>
                <text x="55" y="70" fontSize="10" fill="#6b7280">Manual review required</text>
                
                <rect x="20" y="85" width="25" height="15" fill="url(#highRisk)" rx="2"/>
                <text x="55" y="97" fontSize="12" fill="#1f2937">High Risk (15-25%)</text>
                <text x="55" y="110" fontSize="10" fill="#6b7280">Enhanced documentation</text>
                
                <rect x="20" y="125" width="25" height="15" fill="url(#mediumRisk)" rx="2"/>
                <text x="55" y="137" fontSize="12" fill="#1f2937">Medium Risk (8-15%)</text>
                <text x="55" y="150" fontSize="10" fill="#6b7280">Standard processing</text>
                
                <rect x="20" y="165" width="25" height="15" fill="url(#lowRisk)" rx="2"/>
                <text x="55" y="177" fontSize="12" fill="#1f2937">Low Risk (4-8%)</text>
                <text x="55" y="190" fontSize="10" fill="#6b7280">Preferred rates eligible</text>
                
                <rect x="20" y="205" width="25" height="15" fill="url(#minimalRisk)" rx="2"/>
                <text x="55" y="217" fontSize="12" fill="#1f2937">Minimal Risk (≤4%)</text>
                <text x="55" y="230" fontSize="10" fill="#6b7280">Premium rates eligible</text>
              </g>
              
              {/* Key Metrics Dashboard */}
              <g transform="translate(650,360)">
                <rect x="0" y="0" width="250" height="170" fill="white" stroke="#d1d5db" strokeWidth="1" rx="8"/>
                <text x="125" y="25" textAnchor="middle" fontSize="16" fill="#1f2937" fontWeight="bold">Portfolio Metrics</text>
                
                <text x="20" y="50" fontSize="12" fill="#1f2937" fontWeight="medium">HELOC Volume (2024):</text>
                <text x="20" y="68" fontSize="11" fill="#6b7280">• $359.9B nationwide (+8% YoY)</text>
                <text x="20" y="83" fontSize="11" fill="#6b7280">• Avg balance: $45,157</text>
                
                <text x="20" y="110" fontSize="12" fill="#1f2937" fontWeight="medium">Model Performance:</text>
                <text x="20" y="128" fontSize="11" fill="#6b7280">• Accuracy: 92.4%</text>
                <text x="20" y="143" fontSize="11" fill="#6b7280">• AUC Score: 0.89</text>
                <text x="20" y="158" fontSize="11" fill="#6b7280">• Model stability: High</text>
              </g>
              
              {/* Risk Factors */}
              <g transform="translate(60,560)">
                <text x="0" y="25" fontSize="14" fill="#1f2937" fontWeight="bold">Critical Risk Factors:</text>
                <text x="0" y="48" fontSize="12" fill="#374151">• LTV &gt; 85% combined with DTI &gt; 43% shows 28-35% default probability</text>
                <text x="0" y="66" fontSize="12" fill="#374151">• Credit scores &lt; 620 increase risk by 40-60% across all LTV/DTI combinations</text>
                <text x="0" y="84" fontSize="12" fill="#374151">• Geographic concentration in declining markets adds 15-25% risk premium</text>
                <text x="0" y="102" fontSize="12" fill="#374151">• Property type: Condos and rural properties carry 20% higher risk than SFR</text>
                
                <text x="480" y="25" fontSize="14" fill="#1f2937" fontWeight="bold">Approval Guidelines (2024):</text>
                <text x="480" y="48" fontSize="12" fill="#374151">• Maximum LTV: 85% (primary residence)</text>
                <text x="480" y="66" fontSize="12" fill="#374151">• Maximum DTI: 43% (preferred ≤36%)</text>
                <text x="480" y="84" fontSize="12" fill="#374151">• Minimum credit score: 620 (optimal 700+)</text>
                <text x="480" y="102" fontSize="12" fill="#374151">• Property occupancy: Owner-occupied required</text>
              </g>
            </svg>
          </div>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowCreditRiskGraph(null)} variant="contained" sx={{ backgroundColor: '#545454' }}>
            Close
          </Button>
        </DialogActions>
      </Dialog>
    </Flex>
  );
};

export default TaskTable;
