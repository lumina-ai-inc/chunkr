import axiosInstance from "./axios.config";

export const deleteTasks = async (taskIds: string[]): Promise<void> => {
  try {
    // Make parallel delete requests for all selected tasks
    const deletePromises = taskIds.map((taskId) =>
      axiosInstance.delete(`/api/v1/task/${taskId}`)
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
      axiosInstance.get(`/api/v1/task/${taskId}/cancel`)
    );
    await Promise.all(cancelPromises);
  } catch (error) {
    console.error("Error cancelling tasks:", error);
    throw error;
  }
};
