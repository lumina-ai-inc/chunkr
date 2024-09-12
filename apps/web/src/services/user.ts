import axiosInstance from "./axios.config";
import { AppDispatch } from "../store/store";
import { setUserData, setUserLoading, setUserError } from "../store/userSlice";
import { User } from "../models/user.model";

export async function getUser(dispatch: AppDispatch): Promise<void> {
  dispatch(setUserLoading(true));
  try {
    const { data } = await axiosInstance.get<User>("/api/user");
    dispatch(setUserData(data));
  } catch (error) {
    console.error("Error fetching user data:", error);
    dispatch(setUserError("Failed to fetch user data"));
  }
}
