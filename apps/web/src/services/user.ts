import axiosInstance from "./axios.config";

export async function getUser(): Promise<void> {
    const { data } = await axiosInstance.get("/api/user");
    console.log("user:", data);
  }