import { useEffect, useState } from "react";

/* shared, single worker instance */
const katexWorker = new Worker(
  new URL("../workers/katexWorker.ts", import.meta.url),
  { type: "module" }
);

/* id → resolver map */
const cbMap = new Map<string, (html: string) => void>();

katexWorker.addEventListener(
  "message",
  (e: MessageEvent<{ id: string; html: string }>) => {
    const { id, html } = e.data;
    const cb = cbMap.get(id);
    if (cb) {
      cb(html);
      cbMap.delete(id);
    }
  }
);

const htmlCache = new Map<string, string>();

export function useKatexRenderer(
  input: string | undefined | null,
  displayMode: boolean
) {
  const cacheKey = `${displayMode ? "block" : "inline"}-${input ?? ""}`;

  const [html, setHtml] = useState<string | null>(() =>
    input ? htmlCache.get(cacheKey) ?? null : null
  );

  useEffect(() => {
    if (!input) return;

    /* already have it? -> done */
    if (htmlCache.has(cacheKey)) {
      setHtml(htmlCache.get(cacheKey)!);
      return;
    }

    /* otherwise ask the worker */
    const id = crypto.randomUUID();
    cbMap.set(id, (workerHtml) => {
      htmlCache.set(cacheKey, workerHtml);
      setHtml(workerHtml);
    });

    katexWorker.postMessage({ id, input, displayMode });

    /* tidy if unmounted early */
    return () => {
      cbMap.delete(id);
    };
  }, [cacheKey, input, displayMode]);

  return html;
}
