export const ACTION_LABELS = {
  respond: "Suggested answer",
  fallback: "More information needed",
  escalate: "Human support recommended",
};

export const REASON_LABELS = {
  low_intent_confidence: "The issue category was not clear enough for an automatic answer.",
  insufficient_retrieval_evidence: "Similar historical cases did not provide enough safe guidance.",
  classifier_unavailable: "The intent model is unavailable.",
  authentication_required: "This request requires identity verification.",
  account_action_required: "This request requires access to a customer account.",
  money_or_billing_risk: "This request needs billing review.",
  privacy_or_credential_risk: "This request may involve private information.",
};

export function reasonLabel(reason) {
  return REASON_LABELS[reason] || "The request needs review before a reliable answer can be given.";
}
