import axiosInstance from "./axios.config";
import { User } from "../models/user.model";

export async function getUser(): Promise<User> {
  const { data } = await axiosInstance.get<User>("/api/v1/user");
  return data;
}
