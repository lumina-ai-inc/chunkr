import { ReactNode, useEffect } from "react";
import { useAuth } from "react-oidc-context";
import axiosInstance from "../services/axios.config";
import { getUser } from "../services/user";

export default function Auth({children}: {children: ReactNode}  ) {
  const auth = useAuth();

  useEffect(() => {
    console.log(auth.user);
    if (auth.isAuthenticated) {
      axiosInstance.defaults.headers.common["Authorization"] = `Bearer ${auth.user?.access_token}`;
      getUser().then().catch((error) => {
        console.log("error:", error);
      });
    }
  }, [auth.isAuthenticated, auth.user]);

  return <>{children}</>;
}