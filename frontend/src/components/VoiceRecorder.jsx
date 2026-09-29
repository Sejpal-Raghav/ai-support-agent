function MicrophoneIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 15.25a4 4 0 0 0 4-4v-5a4 4 0 1 0-8 0v5a4 4 0 0 0 4 4Z" />
      <path d="M5.75 10.75v.5a6.25 6.25 0 0 0 12.5 0v-.5M12 17.5v3M9 20.5h6" />
    </svg>
  );
}

export function VoiceRecorder({ recorder, disabled, onStart }) {
  const recording = recorder.status === "recording";
  const requesting = recorder.status === "requesting_permission";
  const timer = `0:${String(recorder.elapsed).padStart(2, "0")} / 0:${recorder.maxSeconds}`;
  return (
    <section className="voice-panel" aria-labelledby="voice-heading">
      <div>
        <h2 id="voice-heading">Voice input</h2>
        <p className="section-copy">
          {requesting
            ? "Allow microphone access in your browser."
            : recording
              ? "Speak naturally, then stop when you finish."
              : "Record one short support question."}
        </p>
      </div>

      <div className="recorder-control">
        <button
          className={`record-button ${recording ? "record-button--active" : ""}`}
          type="button"
          onClick={recording ? recorder.stop : onStart}
          disabled={disabled || requesting}
          aria-label={recording ? "Stop recording" : "Start recording"}
        >
          {recording ? <span className="stop-icon" aria-hidden="true" /> : <MicrophoneIcon />}
        </button>
        <strong>{recording ? "Stop recording" : requesting ? "Waiting for access" : "Start recording"}</strong>
      </div>

      <div className="recording-meter" aria-hidden={!recording}>
        <div className="meter-track">
          <span style={{ transform: `scaleX(${recording ? Math.max(0.03, recorder.level) : 0})` }} />
        </div>
        <span className="recording-time">{recording ? timer : "Up to 30 seconds"}</span>
      </div>

      {recording && (
        <button className="text-button" type="button" onClick={recorder.cancel}>
          Cancel recording
        </button>
      )}
      <p className="privacy-copy">Audio is processed by this app and is not saved.</p>
    </section>
  );
}
