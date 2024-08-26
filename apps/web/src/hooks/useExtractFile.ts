import { useQuery } from "@tanstack/react-query";
import { extractFile } from "../services/extractFileApi";
import { ExtractPayload } from "../models/extract.model";

export function useExtractFile(payload: ExtractPayload | null) {
  return useQuery({
    queryKey: ["extractFile", payload],
    queryFn: () => (payload ? extractFile(payload) : null),
    enabled: !!payload,
  });
}
