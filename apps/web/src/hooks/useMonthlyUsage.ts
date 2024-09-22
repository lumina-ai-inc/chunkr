import { useQuery } from "react-query";
import { getMonthlyUsage } from "../services/usageApi";
import { useAuth } from "react-oidc-context";

export default function useMonthlyUsage() {
  const auth = useAuth();
  const { data, isLoading, error } = useQuery("monthlyUsage", getMonthlyUsage, {
    enabled: auth.isAuthenticated,
    refetchInterval: 3000,
  });
  return { data, isLoading, error };
}
