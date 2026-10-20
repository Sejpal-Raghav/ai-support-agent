import { useCallback, useEffect, useRef, useState } from "react";
import { getHealth } from "../lib/api";

const MAX_ATTEMPTS = 60;

export function useServiceHealth() {
  const [state, setState] = useState({ status: "checking", health: null, error: "" });
  const timerRef = useRef(null);
  const attemptRef = useRef(0);
  const mountedRef = useRef(true);

  const check = useCallback(async () => {
    window.clearTimeout(timerRef.current);
    setState((current) => ({ ...current, status: "checking", error: "" }));
    try {
      const health = await getHealth();
      if (!mountedRef.current) return;
      if (health.classifier_ready && health.retrieval_ready) {
        attemptRef.current = 0;
        setState({ status: health.asr_ready ? "ready" : "partial", health, error: "" });
        return;
      }
      if (["degraded", "unconfigured"].includes(health.status)) {
        setState({
          status: "unavailable", health,
          error: health.message || "The server's response model and evidence corpus are not configured.",
        });
        return;
      }
      throw new Error("services_not_ready");
    } catch {
      if (!mountedRef.current) return;
      attemptRef.current += 1;
      if (attemptRef.current >= MAX_ATTEMPTS) {
        setState({ status: "unavailable", health: null, error: "Service startup timed out." });
        return;
      }
      timerRef.current = window.setTimeout(check, 2000);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    check();
    return () => {
      mountedRef.current = false;
      window.clearTimeout(timerRef.current);
    };
  }, [check]);

  const checkAgain = useCallback(() => {
    attemptRef.current = 0;
    check();
  }, [check]);

  return { ...state, checkAgain };
}
