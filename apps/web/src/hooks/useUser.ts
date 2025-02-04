import { useQuery } from "react-query";
import { getUser } from "../services/user";
import { useAuth } from "react-oidc-context";

export default function useUser() {
  const auth = useAuth();
  const { data, isLoading, error } = useQuery("user", getUser, {
    enabled: auth.isAuthenticated && !auth.isLoading,
    refetchInterval: 300000,
  });
  return { data, isLoading, error };
}
