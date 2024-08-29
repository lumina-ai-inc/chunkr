import { ReactNode } from 'react';
import toast from 'react-hot-toast';
import { useAuth } from "react-oidc-context";
import { useNavigate } from "react-router-dom";


interface AuthGuardProps {
  children: ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const auth = useAuth();
  const navigate = useNavigate();

  if (auth.isLoading) {
    // TODO: Add loader
    return <div>Loading...</div>;
  }

  if (auth.error) {
    console.log(auth.error);
    toast.error("Error signing in");
    navigate("/");
    return null;
  }

  if (!auth.isAuthenticated) {
    auth.signinRedirect();
    return null;
  }

  return <>{children}</>;
}