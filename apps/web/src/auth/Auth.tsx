import { ReactNode, useEffect } from "react";
import { useDispatch } from "react-redux";
import { useAuth } from "react-oidc-context";
import axiosInstance from "../services/axios.config";
import { setUserData, setUserLoading, setUserError } from "../store/userSlice";
import { setAccessToken } from "../store/tokenSlice";
import useUser from "../hooks/useUser";

export default function Auth({ children }: { children: ReactNode }) {
  const auth = useAuth();
  const dispatch = useDispatch();
  const { data, isLoading, error } = useUser();

  useEffect(() => {
    if (auth.isAuthenticated && auth.user?.access_token) {
      dispatch(setAccessToken(auth.user.access_token));
      axiosInstance.defaults.headers.common["Authorization"] =
        `Bearer ${auth.user.access_token}`;
    } else {
      dispatch(setAccessToken(null));
    }
  }, [auth.isAuthenticated, auth.user, dispatch]);

  useEffect(() => {
    auth.signinSilent();
    dispatch(setUserData(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    dispatch(setUserData(data ? data : null));
    dispatch(setUserLoading(false));
    dispatch(setUserError("An unknown error occurred"));
  }, [data, isLoading, error, dispatch]);

  return <>{children}</>;
}
