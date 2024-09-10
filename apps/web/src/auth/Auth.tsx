import { ReactNode, useEffect } from "react";
import { useAuth } from "react-oidc-context";
import { useDispatch } from "react-redux";
import axiosInstance from "../services/axios.config";
import { getUser, getTasks } from "../services/user";
import { AppDispatch } from "../store/store";
import { setAccessToken } from "../store/tokenSlice";

export default function Auth({ children }: { children: ReactNode }) {
  const auth = useAuth();
  const dispatch = useDispatch<AppDispatch>();

  useEffect(() => {
    if (auth.isAuthenticated && auth.user?.access_token) {
      console.log(auth.user.access_token);
      dispatch(setAccessToken(auth.user.access_token));
      axiosInstance.defaults.headers.common["Authorization"] =
        `Bearer ${auth.user.access_token}`;
      getUser(dispatch)
        .then()
        .catch((error) => {
          console.log("error:", error);
        });
      getTasks(1, 10)
        .then()
        .catch((error) => {
          console.log("error:", error);
        });
    } else {
      dispatch(setAccessToken(null));
    }
  }, [auth.isAuthenticated, auth.user, dispatch]);

  return <>{children}</>;
}
