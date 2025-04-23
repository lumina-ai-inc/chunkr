// apps/web/src/services/llmModels.service.ts
import axiosInstance from "./axios.config";

export interface LLMModel {
  id: string;
  default: boolean;
  fallback: boolean;
}

/**
 * Fetches the list of available LLM models from Chunkr’s public API.
 */
export async function fetchLLMModels(): Promise<LLMModel[]> {
  const res = await axiosInstance
    .get("/llm/models")
    .then((res) => res.data)
    .catch((e) => {
      console.error(e);
      throw e;
    });
  return res as LLMModel[];
}
