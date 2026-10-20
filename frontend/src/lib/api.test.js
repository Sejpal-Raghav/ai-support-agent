import { afterEach, expect, it, vi } from "vitest";
import { transcribeAudio } from "./api";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("allows slower speech recognition beyond the old 45-second timeout", async () => {
  vi.useFakeTimers();
  let requestSignal;
  vi.stubGlobal("fetch", vi.fn((url, { signal }) => {
    requestSignal = signal;
    return new Promise((resolve) => {
      setTimeout(() => resolve({ ok: true, json: async () => ({ transcript: "hello" }) }), 50000);
    });
  }));
  const pending = transcribeAudio(new Blob(["audio"], { type: "audio/webm" }));
  await vi.advanceTimersByTimeAsync(45001);
  expect(requestSignal.aborted).toBe(false);
  await vi.advanceTimersByTimeAsync(5000);
  await expect(pending).resolves.toEqual({ transcript: "hello" });
  expect(vi.getTimerCount()).toBe(0);
});

it("still aborts a speech request at the bounded three-minute deadline", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("fetch", vi.fn((url, { signal }) => new Promise((resolve, reject) => {
    signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
  })));
  const pending = transcribeAudio(new Blob(["audio"], { type: "audio/webm" }));
  const rejected = expect(pending).rejects.toMatchObject({ code: "request_timeout" });
  await vi.advanceTimersByTimeAsync(180001);
  await rejected;
});
