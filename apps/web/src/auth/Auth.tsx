import { ReactNode, useEffect, useRef } from "react";
import toast from "react-hot-toast";
import { useAuth } from "react-oidc-context";
import useUser from "../hooks/useUser";
import axiosInstance from "../services/axios.config";

interface AuthProps {
  children: ReactNode;
}

export default function Auth({ children }: AuthProps) {
  const auth = useAuth();
  const { error } = useUser();
  const toastShown = useRef(false); // track if toast was already shown

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
    if (auth.error && !auth.isAuthenticated && !toastShown.current) {
      const ignoredErrors = ["login_required", "IFrame timed out"];

      if (!ignoredErrors.some((msg) => auth.error?.message?.includes(msg))) {
        console.log("Auth error:", auth.error);
        toast.error("Unable to sign in. Please try again.");
        toastShown.current = true; // prevent duplicate toasts
      }
    }
    // Reset toast flag on successful login
    if (auth.isAuthenticated) {
      toastShown.current = false;
    }
  }, [auth.error, auth.isAuthenticated]);

  // Handle user fetch errors
  useEffect(() => {
    if (error && !auth.isAuthenticated && !toastShown.current) {
      console.log("Error getting user information", error);
      toast.error("Error getting user information");
      toastShown.current = true;
    }
  }, [error, auth.isAuthenticated]);

  // Silent sign-in logic
  useEffect(() => {
    const attemptedSilentSignIn = sessionStorage.getItem("attemptedSilentSignIn");

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
  }, [auth.isAuthenticated, auth.isLoading, auth.activeNavigator]);

  return <>{children}</>;
}