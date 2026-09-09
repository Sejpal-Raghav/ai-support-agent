# Annotation guideline v1.0

Each row represents one incoming customer message. Read only `context_messages_json`; its final
message is the current request. Do not infer facts from later messages, from external brand
knowledge, or from a desired reply.

## Required fields

1. `intent_label`: choose exactly one final taxonomy ID. Use the narrowest matching substantive
   intent. Use `other` when no substantive definition applies. Set `intent_ambiguous=true` only
   when two final IDs remain equally plausible after applying their inclusion and exclusion rules;
   still choose the more conservative ID (or `other` if tied).
2. `requires_escalation`: set `true` if resolving this request needs authentication, an account
   action, payment/billing review, credentials/private data, live operational status, legal or
   safety handling, unsupported language, a repeated unresolved exchange, or an unclear request.
   Select every applicable `escalation_reasons_json` enum value. Set `false` and `[]` only when a
   public, evidence-supported reply would be safe.
3. `required_reply_elements_json`: a JSON string list of concrete facts/actions a safe reply must
   include. `forbidden_reply_elements_json`: a JSON string list of claims or promises the reply
   must not make. Use `[]` when none apply.
4. `evidence_adequate`: leave blank in Phase 3. In Phase 4, mark `true` only when supplied
   retrieved train cases collectively support a safe public reply without adding a missing fact;
   otherwise mark `false`.
5. `annotation_notes`, `annotator_id`, `annotation_guideline_version` (`v1.0`), and an ISO-8601
   timestamp with timezone are mandatory.

## Quality-control procedure

The primary annotator completes all 400 validation and 200 golden rows. After finishing, the
annotator blindly re-rates the intent and escalation fields for a deterministic 10% hash-ranked
subset of each packet without seeing their original labels. Resolve disagreements by preserving
the final decision and recording the cause in `annotation_notes`. A separate independent rater is
reserved for Phase 5 reply ratings and must not see system identity.
