import { ReactNode, useEffect } from "react";
import toast from "react-hot-toast";
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";
import Loader from "../pages/Loader/Loader";

interface AuthGuardProps {
  children: ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const auth = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (auth.error) {
      toast.error("Error signing in");
      navigate("/");
    } else if (!auth.isAuthenticated && !auth.isLoading) {
      auth.signinRedirect();
    }
  }, [auth.error, auth.isAuthenticated, auth.isLoading, navigate, auth]);

  if (auth.isLoading) {
    return <Loader />;
  }

  if (auth.isAuthenticated) {
    return <>{children}</>;
  }

  return <Loader />;
}
