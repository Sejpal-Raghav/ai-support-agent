import { reasonLabel } from "../lib/responseLabels";

export function EvidenceDetails({ result }) {
  const scoreLabel = result.score_type === "calibrated_probability" ? "Calibrated confidence" : "Intent similarity";
  return (
    <details className="evidence-details">
      <summary>Response details</summary>
      <dl className="detail-list">
        <div><dt>Issue category</dt><dd>{result.intent.replaceAll("_", " ")}</dd></div>
        <div><dt>{scoreLabel}</dt><dd>{Math.round(result.confidence * 100)}%</dd></div>
        {result.retrieval_similarity !== null && result.retrieval_similarity !== undefined && (
          <div><dt>Retrieval similarity</dt><dd>{result.retrieval_similarity.toFixed(2)}</dd></div>
        )}
        {result.evidence_case_ids.length > 0 && (
          <div><dt>Historical evidence</dt><dd>{result.evidence_case_ids.join(", ")}</dd></div>
        )}
      </dl>
      {result.reasons.length > 0 && (
        <ul className="reason-list">
          {result.reasons.map((reason) => <li key={reason}>{reasonLabel(reason)}</li>)}
        </ul>
      )}
    </details>
  );
}
