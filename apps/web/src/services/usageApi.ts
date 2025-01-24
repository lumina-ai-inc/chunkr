import axiosInstance from "./axios.config";
import { MonthlyUsageData } from "../models/usage.model";

export async function getMonthlyUsage(): Promise<MonthlyUsageData> {
  try {
    const response = await axiosInstance.get("api/v1/usage/monthly");
    return response.data;
  } catch (error) {
    console.error("Error fetching monthly usage:", error);
    throw error;
  }
}

interface TaskDetailsParams {
  start: string;
  end: string;
}

export async function getTaskDetails({ start, end }: TaskDetailsParams) {
  try {
    const response = await axiosInstance.get(`api/v1/tasks/details`, {
      params: {
        start,
        end,
      },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching task details:", error);
    throw error;
  }
}
