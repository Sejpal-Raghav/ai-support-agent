---
title: Voice Support Chatbot
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
---

# Voice Support Chatbot Space

This Docker Space serves the React microphone interface and FastAPI backend from one container.

Before publishing, place the reviewed classifier, taxonomy, safe retrieval index, and Vosk model
in the configured artifact volume. Do not upload raw Twitter data, recordings, secrets, or model
selection/test artifacts that are not required for inference.

The source repository's `Dockerfile` is the deployment build definition. Create a new Docker Space,
copy the tracked deployment files, then set the Space hardware according to the selected model's
measured memory and latency profile.
