import { ReactNode, useEffect } from "react";
import { useAuth } from "react-oidc-context";
import { useNavigate, useLocation } from "react-router-dom";
import Loader from "../pages/Loader/Loader";
import useUser from "../hooks/useUser";
import { OnboardingStatus } from "../models/user.model";

interface AuthGuardProps {
  children: ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const auth = useAuth();
  const { isLoading, data } = useUser();
  const navigate = useNavigate();
  const location = useLocation();

  const onboardingStatus = data?.onboarding_record?.status;

  useEffect(() => {
    if (!auth.isAuthenticated || isLoading) {
      return;
    }

    // If the user doesn't have an onboarding record, redirect to dashboard - for old users
    if (!onboardingStatus) {
      navigate("/dashboard", { replace: true });
    }

    const currentPath = location.pathname;

    if (
      onboardingStatus === OnboardingStatus.Pending &&
      currentPath !== "/onboarding"
    ) {
      navigate("/onboarding", { replace: true });
    } else if (
      onboardingStatus !== OnboardingStatus.Pending &&
      currentPath === "/onboarding"
    ) {
      navigate("/dashboard", { replace: true });
    }
  }, [
    auth.isAuthenticated,
    isLoading,
    onboardingStatus,
    navigate,
    location.pathname,
  ]);

  if (auth.isLoading || isLoading) {
    return (
      <div style={{ width: "100vw", height: "100vh" }}>
        <Loader />
      </div>
    );
  }

  if (auth.isAuthenticated) {
    return <>{children}</>;
  } else if (!auth.isAuthenticated && !auth.isLoading) {
    auth.signinRedirect();
  }

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Loader />
    </div>
  );
}
