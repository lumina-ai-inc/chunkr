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
