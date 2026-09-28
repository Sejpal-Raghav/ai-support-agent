# Voice-Enabled Support Chatbot — Brief Report

## Problem and scope

The application accepts an English support request by microphone or text, displays the recognized
speech, classifies its intent with a neural language model, and retrieves a response from a real
AmazonHelp exchange. It is a demonstration system: it does not authenticate users, access private
accounts, complete transactions, or post replies.

## Dataset

The source is Kaggle's Customer Support on Twitter dataset. A deterministic leading subset of
750,000 validated messages contains 411,965 inbound messages and 59,254 AmazonHelp outbound
messages. The demo builder links direct inbound customer messages to AmazonHelp replies using
`in_response_to_tweet_id`, assigns one unambiguous keyword-derived intent, removes praise-only,
social, and non-actionable exchanges, sanitizes URL-dependent text, and caps each intent at 150
cases. The current local corpus contains 900 historical cases across six intents.

## Architecture and methodology

The browser uses Chrome/Edge Speech Recognition when available. The backend also exposes a Vosk
audio endpoint for deployment with `vosk-model-en-us-0.22-lgraph`. Recognized text is encoded by
`sentence-transformers/all-MiniLM-L6-v2`. Intent is selected by cosine similarity to curated intent
descriptions, and the nearest same-intent historical customer request supplies the response.

Replies are stripped of handles, direct customer names, URLs, and agent signatures. Low intent
confidence, weak retrieval similarity, absent evidence, or credential-related text causes a safe
fallback. Every normal response includes its historical case ID.

The repository includes a BERTweet fine-tuning implementation for the final supervised model. It
is not presented as trained until the required human-reviewed labels exist.

## Current results

- 56 Python tests pass; lint passes.
- React production build succeeds and `npm audit --omit=dev` reports zero vulnerabilities.
- Local page, health endpoint, normal-response path, and unrelated-request fallback were smoke
  tested successfully.
- Example: “I received the wrong item in my order” was classified as
  `damaged_or_wrong_item`, answered with return guidance, and displayed evidence case
  `AmazonHelp:73459`. “Tell me a joke about space” produced a low-confidence fallback.
- Example: “my order is late and tracking has not changed” was classified as
  `delivery_tracking` with 0.878 semantic confidence and answered using case
  `AmazonHelp:128727` in 134 ms after model warm-up.

These are engineering checks, not a held-out accuracy claim. Macro-F1, WER, and end-to-end task
success remain unreported until the planned independent human labels and multi-speaker recordings
are collected.

## Limitations

Keyword-derived corpus labels can be noisy, semantic confidence is not calibrated probability,
and historical Twitter replies can be context-dependent. Browser speech recognition varies by
browser and may use an external browser service. The current demo supports English and six common
intents only. It must not be used for real customer account decisions.

## Deployment

The local service runs at `http://127.0.0.1:7860`. The Dockerfile builds the React client and
FastAPI service for a Hugging Face Docker Space. Public deployment additionally requires the
account-owned Space and inference artifacts; raw Twitter data and recordings must not be uploaded.
