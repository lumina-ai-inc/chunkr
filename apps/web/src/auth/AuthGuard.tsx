import { ReactNode, useEffect } from "react";
import toast from "react-hot-toast";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import Loader from "../pages/Loader/Loader";
import axiosInstance from "../services/axios.config";
import { setAccessToken } from "../store/tokenSlice";
import { setUserData, setUserLoading, setUserError } from "../store/userSlice";
import useUser from "../hooks/useUser";

interface AuthGuardProps {
  children: ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const auth = useAuth();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { data, isLoading, error } = useUser();

  // Handle access token and axios setup
  useEffect(() => {
    if (auth.isAuthenticated && auth.user?.access_token) {
      dispatch(setAccessToken(auth.user.access_token));
      axiosInstance.defaults.headers.common[
        "Authorization"
      ] = `Bearer ${auth.user.access_token}`;
    } else {
      dispatch(setAccessToken(null));
    }
  }, [auth.isAuthenticated, auth.user, dispatch]);

  // Handle user data
  useEffect(() => {
    dispatch(setUserData(data ? data : null));
    dispatch(setUserLoading(false));
    if (error) dispatch(setUserError("An unknown error occurred"));
  }, [data, isLoading, error, dispatch]);

  // Handle auth state and redirects
  useEffect(() => {
    if (auth.error) {
      toast.error("Error signing in");
      navigate("/");
    } else if (!auth.isAuthenticated && !auth.isLoading) {
      auth.signinRedirect();
    }
  }, [auth.error, auth.isAuthenticated, auth.isLoading, navigate, auth]);

  if (auth.isLoading || isLoading) {
    return (
      <div style={{ width: "100vw", height: "100vh" }}>
        <Loader />
      </div>
    );
  }

  if (auth.isAuthenticated) {
    return <>{children}</>;
  }

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Loader />
    </div>
  );
}
