import axiosInstance from "./axios.config";
import { UploadForm } from "../models/upload.model";

export const deleteTasks = async (taskIds: string[]): Promise<void> => {
  try {
    // Make parallel delete requests for all selected tasks
    const deletePromises = taskIds.map((taskId) =>
      axiosInstance.delete(`/task/${taskId}`)
    );

    await Promise.all(deletePromises);
  } catch (error) {
    console.error("Error deleting tasks:", error);
    throw error;
  }
};

export const cancelTasks = async (taskIds: string[]): Promise<void> => {
  try {
    // Make parallel cancel requests for all selected tasks
    const cancelPromises = taskIds.map((taskId) =>
      axiosInstance.get(`/task/${taskId}/cancel`)
    );
    await Promise.all(cancelPromises);
  } catch (error) {
    console.error("Error cancelling tasks:", error);
    throw error;
  }
};

export const updateTask = async (
  taskId: string,
  task: Partial<UploadForm>
): Promise<void> => {
  try {
    await axiosInstance.patch(`/task/${taskId}/parse`, task);
  } catch (error) {
    console.error("Error updating task:", error);
    throw error;
  }
};

export const cancelTask = async (taskId: string): Promise<void> => {
  try {
    await axiosInstance.get(`/task/${taskId}/cancel`);
  } catch (error) {
    console.error("Error cancelling task:", error);
    throw error;
  }
};
