import { ReactNode, useEffect } from "react";
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
      const ignoredErrors = [
        "login_required",
        "IFrame timed out"
      ];

      if (!ignoredErrors.some(msg => auth.error?.message?.includes(msg))) {
        console.log("Auth error:", auth.error);
        toast.error("Unable to sign in. Please try again.");
      }
    }
  }, [auth.error]);

  useEffect(() => {
    if (error) {
      console.log("Error getting user information", error);
      toast.error("Error getting user information");
    }
  }, [error]);

  useEffect(() => {
    // Add a flag to track if we've attempted silent sign-in
    const attemptedSilentSignIn = sessionStorage.getItem('attemptedSilentSignIn');

    // Only try silent sign-in if we haven't attempted it yet
    if (!auth.isAuthenticated && !auth.isLoading && !auth.activeNavigator && !attemptedSilentSignIn) {
      sessionStorage.setItem('attemptedSilentSignIn', 'true');
      auth.signinSilent().catch((err) => {
        console.log("Silent sign-in failed:", err);
      });
    }
  }, [auth.isAuthenticated, auth.isLoading, auth.activeNavigator]);

  return <>{children}</>;
}