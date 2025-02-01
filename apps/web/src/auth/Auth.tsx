import { ReactNode, useEffect } from "react";
import toast from "react-hot-toast";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";
import useUser from "../hooks/useUser";
import axiosInstance from "../services/axios.config";
import Loader from "../pages/Loader/Loader";

interface AuthProps {
  children: ReactNode;
}

export default function Auth({ children }: AuthProps) {
  const auth = useAuth();
  const navigate = useNavigate();
  const { isLoading } = useUser();


  // Handle access token and axios setup
  useEffect(() => {
    if (auth.isAuthenticated && auth.user?.access_token) {
      axiosInstance.defaults.headers.common[
        "Authorization"
      ] = `Bearer ${auth.user.access_token}`;
    } else {
      axiosInstance.defaults.headers.common["Authorization"] = null;
    }
  }, [auth.isAuthenticated, auth.user]);

  // Handle auth state and redirects
  useEffect(() => {
    if (auth.error) {
      console.log(auth.error);
      toast.error("Error signing in");
      navigate("/");
    }
  }, [auth.error, auth.isAuthenticated, auth.isLoading, navigate, auth]);

  if (auth.isLoading || isLoading) {
    return (
      <div style={{ width: "100vw", height: "100vh" }}>
        <Loader />
      </div>
    );
  }

  return <>{children}</>;
}