export function ResponsePanel({ result, status }) {
  if (status === "responding") {
    return (
      <section className="response-block" aria-labelledby="response-heading">
        <h2 id="response-heading">Support response</h2>
        <p className="empty-copy" role="status">Finding a supported answer…</p>
      </section>
    );
  }
  if (!result) return null;
  const intent = result.intent.replaceAll("_", " ");
  const intentLabel = intent.charAt(0).toUpperCase() + intent.slice(1);
  return (
    <section className="response-block" aria-labelledby="response-heading">
      <h2 id="response-heading">Support response</h2>
      <dl className="response-intent">
        <dt>Intent</dt>
        <dd>{intentLabel}</dd>
      </dl>
      <p className="response-text">{result.response}</p>
    </section>
  );
}
