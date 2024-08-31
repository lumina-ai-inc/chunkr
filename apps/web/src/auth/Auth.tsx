import { ReactNode, useEffect } from "react";
import { useAuth } from "react-oidc-context";
import axiosInstance from "../services/axios.config";

export default function Auth({children}: {children: ReactNode}  ) {
  const auth = useAuth();

  useEffect(() => {
    console.log(auth.user)
    if (auth.isAuthenticated) {
      axiosInstance.defaults.headers.common["Authorization"] = `Bearer ${auth.user?.access_token}`;
    }
  }, [auth.isAuthenticated, auth.user?.access_token]);

  return <>{children}</>;
}