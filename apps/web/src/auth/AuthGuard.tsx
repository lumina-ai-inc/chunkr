import { ReactNode } from "react";
import { useAuth } from "react-oidc-context";
import Loader from "../pages/Loader/Loader";
import useUser from "../hooks/useUser";

interface AuthGuardProps {
  children: ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const auth = useAuth();
  const { isLoading } = useUser();

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
