import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { useVoiceRecorder } from "./useVoiceRecorder";

beforeEach(() => {
  window.MediaRecorder = class {
    static isTypeSupported() { return false; }
  };
  Object.defineProperty(navigator, "mediaDevices", {
    configurable: true, value: { getUserMedia: vi.fn() },
  });
});

it("releases the microphone if the recorder cannot be constructed", async () => {
  const stop = vi.fn();
  navigator.mediaDevices.getUserMedia.mockResolvedValue({ getTracks: () => [{ stop }] });
  const { result } = renderHook(() => useVoiceRecorder(vi.fn()));
  await act(() => result.current.start());
  expect(stop).toHaveBeenCalledOnce();
  expect(result.current.status).toBe("idle");
});

it("releases a microphone permission grant that arrives after unmount", async () => {
  let grant;
  const stop = vi.fn();
  navigator.mediaDevices.getUserMedia.mockReturnValue(new Promise((resolve) => { grant = resolve; }));
  const { result, unmount } = renderHook(() => useVoiceRecorder(vi.fn()));
  act(() => { result.current.start(); });
  unmount();
  await act(async () => { grant({ getTracks: () => [{ stop }] }); });
  await waitFor(() => expect(stop).toHaveBeenCalledOnce());
});
