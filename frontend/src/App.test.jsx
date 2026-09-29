import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

const responsePayload = {
  session_id: "test-session",
  text: "Where is my delayed package?",
  intent: "delivery_tracking",
  confidence: 0.85,
  score_type: "semantic_similarity",
  retrieval_similarity: 0.71,
  response: "Please check the latest carrier update for your order.",
  action: "respond",
  reasons: [],
  evidence_case_ids: ["AmazonHelp:test-case"],
  classifier_version: "minilm-test",
  retrieval_index_version: "test-index",
  latency_ms: 12,
};

function jsonResponse(body, status = 200) {
  return Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(body) });
}

describe("Support Listening Desk", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.MediaRecorder = class {
      static isTypeSupported() { return true; }
    };
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: { getUserMedia: vi.fn() },
    });
  });

  it("displays typed text, grounded response, and truthful score metadata", async () => {
    vi.stubGlobal("fetch", vi.fn((url) => {
      if (url === "/api/health") {
        return jsonResponse({ status: "ready", asr_ready: true, classifier_ready: true, retrieval_ready: true });
      }
      return jsonResponse(responsePayload);
    }));
    const user = userEvent.setup();
    render(<App />);

    await screen.findByText("Ready");
    const composer = screen.getByLabelText("Or type your question");
    await user.type(composer, "Where is my delayed package?");
    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(await screen.findByText(responsePayload.response)).toBeVisible();
    expect(screen.getByText("Where is my delayed package?")).toBeVisible();
    expect(screen.getByText("Intent similarity")).toBeInTheDocument();
    expect(screen.getByText("AmazonHelp:test-case")).toBeInTheDocument();
  });

  it("turns microphone denial into a recoverable instruction", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({
      status: "ready",
      asr_ready: true,
      classifier_ready: true,
      retrieval_ready: true,
    })));
    navigator.mediaDevices.getUserMedia.mockRejectedValue(
      Object.assign(new Error("denied"), { name: "NotAllowedError" }),
    );
    const user = userEvent.setup();
    render(<App />);

    const recordButton = await screen.findByRole("button", { name: "Start recording" });
    await waitFor(() => expect(recordButton).toBeEnabled());
    await user.click(recordButton);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Microphone access is blocked. Allow it in your browser, then try again.",
    );
  });

  it("keeps the transcript available when response generation fails", async () => {
    vi.stubGlobal("fetch", vi.fn((url) => {
      if (url === "/api/health") {
        return jsonResponse({ status: "ready", asr_ready: true, classifier_ready: true, retrieval_ready: true });
      }
      return jsonResponse({ detail: { code: "response_failed", message: "failed" } }, 503);
    }));
    const user = userEvent.setup();
    render(<App />);
    const composer = await screen.findByLabelText("Or type your question");
    await user.type(composer, "My refund has not arrived");
    await waitFor(() => expect(screen.getByRole("button", { name: "Send question" })).toBeEnabled());
    await user.click(screen.getByRole("button", { name: "Send question" }));

    expect(await screen.findByText("My refund has not arrived")).toBeVisible();
    expect(screen.getByRole("button", { name: "Retry response" })).toBeVisible();
  });
});
