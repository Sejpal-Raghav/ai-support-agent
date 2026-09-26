# Voice Chatbot Requirement and Evidence Matrix

This matrix is the release gate for the voice-enabled chatbot submission. A requirement is not
complete until the listed evidence exists and its acceptance check passes.

| Requirement | Implementation evidence | Verification evidence | Acceptance gate |
|---|---|---|---|
| Accept voice input | Browser microphone controls and `POST /api/transcribe` | Frontend permission-denial test and API audio fixture | Chrome and Edge can submit a recording without an uncaught UI error |
| Convert speech to text | Vosk `vosk-model-en-us-0.22-lgraph` after FFmpeg 16-kHz mono conversion | Held-out 50-recording WER report | Overall WER is at most 30% |
| Display recognized speech | Transcript panel in the React interface | UI rendering test and deployment screenshot | Transcript is visible before response generation completes |
| Use deep learning | Fine-tuned BERTweet and DeBERTa sequence classifiers | Training manifests, validation comparison, and model-selection record | Selected model has validation macro-F1 at least 0.80 and no supported intent below 0.65 F1 |
| Generate an appropriate response | Train-only same-intent retrieval with sanitization and safe fallback | Provenance, safety, and held-out response checks | Every normal response cites train evidence; unsafe or unsupported requests fall back |
| Display chatbot response | Response, intent, confidence, action, and reasons panels | UI component and end-to-end tests | A successful request visibly renders all required fields |
| Be publicly accessible | Single Hugging Face Docker Space serving API and built React assets | Ten external health/inference checks over 30 minutes | All ten checks pass and the URL works from another device/network |
| Submit source code | GitHub branch and reviewed merge to `main` | Clean-clone verification | No raw data, audio, secrets, caches, or oversized model weights are committed |
| Submit a brief report | Voice-project report linked from `README.md` | Documentation link check | Dataset, architecture, methodology, results, limitations, source URL, and live URL are present |
| Reproduce results | Frozen manifests and one evaluation command | Clean-checkout cached reproduction | Reported tables are regenerated without training or test-set tuning |

## Fixed boundaries

- The application supports English voice input and typed fallback input.
- Text-to-speech is not required by the supplied assignment and is excluded.
- The service drafts responses but never authenticates users, accesses private account data,
  performs account actions, or posts to Twitter.
- Raw Twitter data and participant recordings remain local. The deployed image contains only the
  selected model, reviewed taxonomy, and train-only retrieval artifacts required for inference.
- Test labels and held-out voice results are used for final reporting only. Model selection,
  calibration, retrieval thresholds, and fallback thresholds use validation data exclusively.

## Release evidence checklist

- [ ] Selected brand and reviewed intent taxonomy
- [ ] Thread-level leakage and near-duplicate audit
- [ ] BERTweet vertical-slice model and manifest
- [ ] Public microphone-to-response deployment
- [ ] BERTweet versus DeBERTa validation comparison
- [ ] Fifty-recording voice evaluation
- [ ] Frozen text, voice, safety, coverage, and latency metrics
- [ ] Public Space URL and GitHub URL
- [ ] Final report and five-minute reproduction instructions
