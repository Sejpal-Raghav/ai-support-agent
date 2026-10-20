export class ApiError extends Error {
  constructor(code, message, status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function requestJson(url, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const upstreamSignal = options.signal;
  const abortFromUpstream = () => controller.abort();
  upstreamSignal?.addEventListener("abort", abortFromUpstream, { once: true });
  if (upstreamSignal?.aborted) controller.abort();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = body.detail || {};
      throw new ApiError(
        detail.code || `http_${response.status}`,
        detail.message || "The request could not be completed.",
        response.status,
      );
    }
    return body;
  } catch (error) {
    if (error.name === "AbortError") {
      const cancelled = Boolean(upstreamSignal?.aborted);
      throw new ApiError(
        cancelled ? "request_cancelled" : "request_timeout",
        cancelled ? "The request was cancelled." : "The request took too long.",
      );
    }
    if (error instanceof ApiError) throw error;
    throw new ApiError("network_error", "The local service could not be reached.");
  } finally {
    window.clearTimeout(timeout);
    upstreamSignal?.removeEventListener("abort", abortFromUpstream);
  }
}

export function getHealth(options = {}) {
  return requestJson("/api/health", { method: "GET", ...options }, 10000);
}

export function transcribeAudio(blob, options = {}) {
  const form = new FormData();
  const extension = blob.type.includes("ogg") ? "ogg" : "webm";
  form.append("audio", blob, `support-request.${extension}`);
  return requestJson("/api/transcribe", { method: "POST", body: form, ...options }, 180000);
}

export function requestSupport(text, options = {}) {
  return requestJson(
    "/api/respond",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, session_id: options.sessionId || null }),
      signal: options.signal,
    },
    30000,
  );
}
