import { ReactNode } from 'react';
import { keycloak } from './KeycloakProvider';


interface AuthGuardProps {
  children: ReactNode;
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const isAuthenticated = keycloak.authenticated;

  if (!isAuthenticated) {
    keycloak.login();
    return null;
  }

  return <>{children}</>;
}