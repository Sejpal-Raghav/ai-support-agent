import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage, microphoneErrorCode } from "../lib/errorMessages";

const MAX_SECONDS = 30;

function preferredMimeType() {
  const candidates = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"];
  return candidates.find((type) => window.MediaRecorder?.isTypeSupported(type)) || "";
}

export function useVoiceRecorder(onAudioReady) {
  const [state, setState] = useState({ status: "idle", elapsed: 0, level: 0, error: "" });
  const refs = useRef({
    recorder: null,
    stream: null,
    context: null,
    levelTimer: 0,
    elapsedTimer: 0,
    discard: false,
  });
  const callbackRef = useRef(onAudioReady);
  callbackRef.current = onAudioReady;

  const cleanup = useCallback(() => {
    window.clearTimeout(refs.current.levelTimer);
    window.clearInterval(refs.current.elapsedTimer);
    refs.current.stream?.getTracks().forEach((track) => track.stop());
    refs.current.context?.close().catch(() => {});
    refs.current = {
      recorder: null,
      stream: null,
      context: null,
      levelTimer: 0,
      elapsedTimer: 0,
      discard: false,
    };
  }, []);

  const stop = useCallback((discard = false) => {
    refs.current.discard = discard;
    const recorder = refs.current.recorder;
    if (recorder?.state === "recording") recorder.stop();
  }, []);

  const start = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setState({
        status: "idle",
        elapsed: 0,
        level: 0,
        error: errorMessage("unsupported_recorder"),
      });
      return;
    }
    setState({ status: "requesting_permission", elapsed: 0, level: 0, error: "" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = preferredMimeType();
      if (!mimeType) throw Object.assign(new Error("unsupported"), { name: "NotSupportedError" });
      const recorder = new MediaRecorder(stream, { mimeType });
      const chunks = [];
      let failed = false;
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = AudioContextClass ? new AudioContextClass() : null;
      const analyser = context?.createAnalyser();
      if (analyser) {
        analyser.fftSize = 256;
        context.createMediaStreamSource(stream).connect(analyser);
      }
      const levels = new Uint8Array(analyser?.frequencyBinCount || 1);
      refs.current = {
        recorder,
        stream,
        context,
        levelTimer: 0,
        elapsedTimer: 0,
        discard: false,
      };

      const updateLevel = () => {
        if (!analyser) return;
        analyser.getByteTimeDomainData(levels);
        const rms = Math.sqrt(
          levels.reduce((sum, value) => sum + ((value - 128) / 128) ** 2, 0) /
            levels.length,
        );
        setState((current) => ({ ...current, level: Math.min(1, rms * 4) }));
        refs.current.levelTimer = window.setTimeout(updateLevel, 50);
      };
      recorder.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      recorder.onerror = () => {
        failed = true;
        cleanup();
        setState({
          status: "idle",
          elapsed: 0,
          level: 0,
          error: errorMessage("invalid_audio"),
        });
      };
      recorder.onstop = () => {
        const discard = failed || refs.current.discard;
        cleanup();
        setState({ status: "idle", elapsed: 0, level: 0, error: "" });
        if (!discard && chunks.length) callbackRef.current(new Blob(chunks, { type: mimeType }));
      };
      recorder.start(250);
      const startedAt = Date.now();
      refs.current.elapsedTimer = window.setInterval(() => {
        const elapsed = Math.min(MAX_SECONDS, Math.floor((Date.now() - startedAt) / 1000));
        setState((current) => ({ ...current, elapsed }));
        if (elapsed >= MAX_SECONDS) stop(false);
      }, 250);
      updateLevel();
      setState({ status: "recording", elapsed: 0, level: 0, error: "" });
    } catch (error) {
      cleanup();
      const code = microphoneErrorCode(error);
      setState({ status: "idle", elapsed: 0, level: 0, error: errorMessage(code) });
    }
  }, [cleanup, stop]);

  useEffect(() => cleanup, [cleanup]);

  return {
    ...state,
    start,
    stop: () => stop(false),
    cancel: () => stop(true),
    maxSeconds: MAX_SECONDS,
  };
}
