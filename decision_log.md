# Decision Log

1. **Use AmazonHelp for the demo.** It has 59,254 outbound messages in the deterministic subset,
   providing substantially more historical evidence than the next candidate.
2. **Use a deterministic 750,000-row source subset.** The assignment permits subsampling, and this
   keeps local setup practical while retaining hundreds of thousands of customer messages.
3. **Write Parquet atomically.** Interrupted ingestion writes `.part` only, preventing corrupt
   artifacts from being mistaken for completed data.
4. **Support six high-volume intents.** A compact taxonomy is easier to explain and safer than
   claiming coverage for every rare support request.
5. **Require an unambiguous keyword match when building demo cases.** Multi-match messages are
   excluded rather than assigned a questionable training/retrieval label.
6. **Cap each intent at 150 historical cases.** The cap limits startup latency and prevents the
   largest topic from dominating the evidence corpus.
7. **Use MiniLM for the working vertical slice.** It is a real neural encoder that runs acceptably
   on CPU; BERTweet remains the supervised upgrade after labels are reviewed.
8. **Retrieve only within the predicted intent.** This reduces fluent but topically wrong replies.
9. **Preserve an evidence case ID for every normal response.** Reviewers can audit which historical
   exchange supported the output.
10. **Filter for problem-solving exchanges before retrieval.** Praise-only/social conversations
    and replies without a support action are excluded; URLs are sanitized and dangling punctuation
    is repaired so useful guidance is not discarded solely because the original contained a link.
11. **Sanitize identity residue.** Handles, customer names, URLs, and agent signatures are removed
    before historical text reaches the UI.
12. **Fail closed on weak confidence or unsafe content.** The chatbot displays a fallback instead
    of forcing an answer.
13. **Use browser recognition for localhost and Vosk for self-hosted deployment.** This provides an
    immediately testable demo while retaining a private, portable backend ASR path.
14. **Serve React and FastAPI from one process.** One origin avoids CORS and independent release
    drift during evaluation.
15. **Do not invent evaluation results.** Engineering smoke tests are reported separately from the
    held-out human and voice evaluation that still must be conducted.
