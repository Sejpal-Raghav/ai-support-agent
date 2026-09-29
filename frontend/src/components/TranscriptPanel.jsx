export function TranscriptPanel({ transcript, source, status, onEdit }) {
  const waiting = status === "transcribing";
  return (
    <section className="transcript-block" aria-labelledby="request-heading">
      <div className="section-heading-row">
        <div>
          <h2 id="request-heading">Your request</h2>
          {transcript && <p className="source-label">{source === "voice" ? "Recognized speech" : "Typed message"}</p>}
        </div>
        {transcript && status !== "responding" && (
          <button className="text-button" type="button" onClick={() => onEdit(transcript)}>
            Edit text
          </button>
        )}
      </div>
      <p className={transcript ? "transcript-text" : "empty-copy"}>
        {waiting
          ? "Transcribing your recording…"
          : transcript || "Your words will appear here after you finish speaking."}
      </p>
    </section>
  );
}
