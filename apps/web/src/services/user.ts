import axiosInstance from "./axios.config";
import { AppDispatch } from "../store/store";
import { setUserData, setUserLoading, setUserError } from "../store/userSlice";
import { User } from "../models/user.model";

export async function getUser(dispatch: AppDispatch): Promise<void> {
  dispatch(setUserLoading(true));
  try {
    const { data } = await axiosInstance.get<User>("/api/user");
    console.log("user:", data);
    dispatch(setUserData(data));
  } catch (error) {
    console.error("Error fetching user data:", error);
    dispatch(setUserError("Failed to fetch user data"));
  }
}

export async function getTasks(page: number, limit: number): Promise<void> {
  try {
    const { data } = await axiosInstance.get<User>("/api/tasks?page=" + page + "&limit=" + limit);
    console.log("tasks:", data);
  } catch (error) {
    console.error("Error fetching user data:", error);
  }
}
