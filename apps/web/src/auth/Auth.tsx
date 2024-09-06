import { ReactNode, useEffect } from "react";
import { useAuth } from "react-oidc-context";
import { useDispatch } from "react-redux";
import axiosInstance from "../services/axios.config";
import { getUser } from "../services/user";
import { AppDispatch } from "../store/store";

export default function Auth({ children }: { children: ReactNode }) {
  const auth = useAuth();
  const dispatch = useDispatch<AppDispatch>();

  useEffect(() => {
    console.log(auth.user);
    if (auth.isAuthenticated) {
      axiosInstance.defaults.headers.common["Authorization"] =
        `Bearer ${auth.user?.access_token}`;
      getUser(dispatch)
        .then()
        .catch((error) => {
          console.log("error:", error);
        });
    }
  }, [auth.isAuthenticated, auth.user, dispatch]);

  return <>{children}</>;
}
