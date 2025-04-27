/*  ────────────────────────────────────────────────────────────────
    Dedicated worker that turns LaTeX → KaTeX HTML.                *
    (Anything done here no longer blocks the main/UI thread.)      *
   ──────────────────────────────────────────────────────────────── */
import katex from "katex";

interface RenderMsg {
  id: string;
  input: string;
  displayMode: boolean;
}

self.addEventListener("message", (e: MessageEvent<RenderMsg>) => {
  const { id, input, displayMode } = e.data;
  let html: string;

  try {
    html = katex.renderToString(input, {
      displayMode,
      throwOnError: false,
      output: "html",
      strict: false,
    });
  } catch {
    html = displayMode
      ? `<div class="katex-error">$$${input}$$</div>`
      : `<span class="katex-error">\\(${input}\\)</span>`;
  }

  (self as DedicatedWorkerGlobalScope).postMessage({ id, html });
});
