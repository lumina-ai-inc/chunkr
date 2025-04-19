// apps/web/src/services/llmModels.service.ts

export interface LLMModel {
  id: string;
  default: boolean;
  fallback: boolean;
}

/**
 * Fetches the list of available LLM models from Chunkr’s public API.
 */
export async function fetchLLMModels(): Promise<LLMModel[]> {
  const res = await fetch("https://api.chunkr.ai/llm/models");
  if (!res.ok) {
    throw new Error(`Failed to fetch LLM models: ${res.status}`);
  }
  return (await res.json()) as LLMModel[];
}
