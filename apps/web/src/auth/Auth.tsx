import { ReactNode, useEffect } from "react";
import { useAuth } from "react-oidc-context";
import useUser from "../hooks/useUser";
import axiosInstance from "../services/axios.config";

interface AuthProps {
  children: ReactNode;
}

export default function Auth({ children }: AuthProps) {
  const auth = useAuth();
  const { error } = useUser();

  // Handle access token and axios setup
  useEffect(() => {
    if (auth.isAuthenticated && auth.user?.access_token) {
      axiosInstance.defaults.headers.common[
        "Authorization"
      ] = `Bearer ${auth.user.access_token}`;
    } else {
      delete axiosInstance.defaults.headers.common["Authorization"]; // cleaner thannull
    }
  }, [auth.isAuthenticated, auth.user]);

  // Handle auth state and errors
  useEffect(() => {
    if (auth.error && !auth.isAuthenticated) {
      const ignoredErrors = ["login_required", "IFrame timed out"];

      if (!ignoredErrors.some((msg) => auth.error?.message?.includes(msg))) {
        console.log("Auth error:", auth.error);
        console.log("Unable to sign in. Please try again.");
      }
    }
  }, [auth.error, auth.isAuthenticated]);

  // Handle user fetch errors
  useEffect(() => {
    if (error && !auth.isAuthenticated) {
      console.log("Error getting user information", error);
    }
  }, [error, auth.isAuthenticated]);

  // Silent sign-in logic
  useEffect(() => {
    const attemptedSilentSignIn = sessionStorage.getItem(
      "attemptedSilentSignIn"
    );

    if (
      !auth.isAuthenticated &&
      !auth.isLoading &&
      !auth.activeNavigator &&
      !attemptedSilentSignIn
    ) {
      sessionStorage.setItem("attemptedSilentSignIn", "true");
      auth.signinSilent().catch((err) => {
        console.log("Silent sign-in failed:", err);
      });
    }
  }, [auth.isAuthenticated, auth.isLoading, auth.activeNavigator, auth]);

  return <>{children}</>;
}
