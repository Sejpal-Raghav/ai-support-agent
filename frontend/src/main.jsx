import { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const initialState = { status: "idle", transcript: "", result: null, error: "" };

function App() {
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const [state, setState] = useState(initialState);
  const [typedText, setTypedText] = useState("");

  const respond = async (text) => {
    const reply = await fetch("/api/respond", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const body = await reply.json();
    if (!reply.ok) throw new Error(body.detail?.message || "The support response could not be created.");
    return body;
  };

  const transcribe = async (blob) => {
    const form = new FormData();
    form.append("audio", blob, "support-request.webm");
    const reply = await fetch("/api/transcribe", { method: "POST", body: form });
    const body = await reply.json();
    if (!reply.ok) throw new Error(body.detail?.message || "The recording could not be transcribed.");
    return body.transcript;
  };

  const stopRecording = () => recorderRef.current?.stop();

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
      streamRef.current = stream;
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => chunks.push(event.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop());
        setState((current) => ({ ...current, status: "transcribing" }));
        try {
          const transcript = await transcribe(new Blob(chunks, { type: "audio/webm" }));
          setState((current) => ({ ...current, transcript, status: "thinking" }));
          const result = await respond(transcript);
          setState({ status: "complete", transcript, result, error: "" });
        } catch (error) {
          setState((current) => ({ ...current, status: "idle", error: error.message }));
        }
      };
      recorder.start();
      setState({ ...initialState, status: "recording" });
    } catch (error) {
      setState({ ...initialState, error: "Microphone access is needed to record a support request." });
    }
  };

  const submitTyped = async (event) => {
    event.preventDefault();
    const text = typedText.trim();
    if (!text) return;
    setState({ ...initialState, status: "thinking", transcript: text });
    try {
      const result = await respond(text);
      setState({ status: "complete", transcript: text, result, error: "" });
    } catch (error) {
      setState((current) => ({ ...current, status: "idle", error: error.message }));
    }
  };

  const recording = state.status === "recording";
  const busy = ["recording", "transcribing", "thinking"].includes(state.status);

  return <main>
    <header>
      <p className="kicker">Voice support, with receipts</p>
      <h1>Say what went wrong.<br />See what happens next.</h1>
      <p className="lede">Record a short support request. The desk transcribes it, identifies the issue, and only uses a historically supported response when it has evidence.</p>
    </header>
    <section className="desk" aria-live="polite">
      <div className="recording-panel">
        <p className="status">{recording ? "Listening now" : state.status === "transcribing" ? "Writing it down" : state.status === "thinking" ? "Checking the support record" : "Ready when you are"}</p>
        <button className={`mic ${recording ? "active" : ""}`} onClick={recording ? stopRecording : startRecording} disabled={busy && !recording} aria-label={recording ? "Stop recording" : "Start recording"}>
          <span>{recording ? "Stop" : "Talk"}</span>
        </button>
        <p className="hint">{recording ? "Tap to finish. Keep it under 30 seconds." : "Your recording is processed for this request only."}</p>
      </div>
      <div className="conversation">
        <article className="transcript">
          <p className="label">Recognized speech</p>
          <p>{state.transcript || "Your words will appear here after you finish speaking."}</p>
        </article>
        {state.result && <article className="answer">
          <div className="answer-heading"><p className="label">Support response</p><span>{state.result.action}</span></div>
          <p>{state.result.response}</p>
          <dl><div><dt>Intent</dt><dd>{state.result.intent.replaceAll("_", " ")}</dd></div><div><dt>Confidence</dt><dd>{Math.round(state.result.confidence * 100)}%</dd></div></dl>
          {state.result.evidence_case_ids.length > 0 && <details><summary>Why this response?</summary><p>Historical support cases: {state.result.evidence_case_ids.join(", ")}</p></details>}
          {state.result.reasons.length > 0 && <p className="fallback">{state.result.reasons.join(", ").replaceAll("_", " ")}</p>}
        </article>}
        {state.error && <p className="error">{state.error}</p>}
      </div>
    </section>
    <form onSubmit={submitTyped} className="typed"><label htmlFor="typed-request">No microphone? Type your request.</label><div><input id="typed-request" value={typedText} onChange={(event) => setTypedText(event.target.value)} maxLength="1000" placeholder="For example: I need help with my delivery" /><button disabled={busy}>Send</button></div></form>
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
