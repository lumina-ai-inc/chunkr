import axiosInstance from "./axios.config";
import { MonthlyUsageData } from "../models/usage.model";

export async function getMonthlyUsage(): Promise<MonthlyUsageData> {
  try {
    const response = await axiosInstance.get("api/usage/monthly");
    console.log(response.data);
    return response.data;
  } catch (error) {
    console.error("Error fetching monthly usage:", error);
    throw error;
  }
}
