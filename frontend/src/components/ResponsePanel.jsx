import { ACTION_LABELS, reasonLabel } from "../lib/responseLabels";
import { EvidenceDetails } from "./EvidenceDetails";

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
  const needsHuman = result.action === "escalate";
  return (
    <section className="response-block" aria-labelledby="response-heading">
      <div className="response-label-row">
        <h2 id="response-heading">Support response</h2>
        <span className={`action-label action-label--${result.action}`}>
          {ACTION_LABELS[result.action] || "Support response"}
        </span>
      </div>
      <p className="response-text">{result.response}</p>
      {result.action !== "respond" && result.reasons.length > 0 && (
        <p className="decision-copy">{reasonLabel(result.reasons[0])}</p>
      )}
      {needsHuman && (
        <p className="human-note">
          This demo cannot contact support or access your account. Contact the brand’s support team for further help.
        </p>
      )}
      <EvidenceDetails result={result} />
    </section>
  );
}
