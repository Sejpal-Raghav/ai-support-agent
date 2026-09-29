import { useCallback, useEffect, useReducer, useRef } from "react";
import { requestSupport, transcribeAudio } from "../lib/api";
import { errorMessage } from "../lib/errorMessages";

const initialState = {
  status: "idle",
  transcript: "",
  inputSource: "voice",
  result: null,
  errorCode: "",
  error: "",
};

function reducer(state, action) {
  switch (action.type) {
    case "transcribing":
      return { ...initialState, status: "transcribing", inputSource: "voice" };
    case "responding":
      return {
        ...initialState,
        status: "responding",
        transcript: action.text,
        inputSource: action.source,
      };
    case "complete":
      return { ...state, status: "complete", result: action.result, errorCode: "", error: "" };
    case "error":
      return {
        ...state,
        status: "error",
        errorCode: action.code,
        error: errorMessage(action.code),
      };
    case "reset":
      return initialState;
    default:
      return state;
  }
}

export function useSupportConversation() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const requestRef = useRef({ id: 0, controller: null });

  const beginRequest = useCallback(() => {
    requestRef.current.controller?.abort();
    const controller = new AbortController();
    const id = requestRef.current.id + 1;
    requestRef.current = { id, controller };
    return { id, controller };
  }, []);

  const finishResponse = useCallback(async (text, source, request) => {
    dispatch({ type: "responding", text, source });
    try {
      const result = await requestSupport(text, { signal: request.controller.signal });
      if (requestRef.current.id === request.id) dispatch({ type: "complete", result });
    } catch (error) {
      if (requestRef.current.id !== request.id || error.code === "request_cancelled") return;
      dispatch({ type: "error", code: error.code || "response_failed" });
    }
  }, []);

  const submitText = useCallback(
    async (text, source = "typed") => {
      const normalized = text.trim();
      if (!normalized) return;
      const request = beginRequest();
      await finishResponse(normalized, source, request);
    },
    [beginRequest, finishResponse],
  );

  const submitAudio = useCallback(
    async (blob) => {
      const request = beginRequest();
      dispatch({ type: "transcribing" });
      try {
        const result = await transcribeAudio(blob, { signal: request.controller.signal });
        if (requestRef.current.id !== request.id) return;
        await finishResponse(result.transcript, "voice", request);
      } catch (error) {
        if (requestRef.current.id !== request.id || error.code === "request_cancelled") return;
        dispatch({ type: "error", code: error.code || "invalid_audio" });
      }
    },
    [beginRequest, finishResponse],
  );

  const retryResponse = useCallback(() => {
    if (state.transcript) submitText(state.transcript, state.inputSource);
  }, [state.inputSource, state.transcript, submitText]);

  const cancel = useCallback(() => {
    requestRef.current.controller?.abort();
    requestRef.current = { id: requestRef.current.id + 1, controller: null };
    dispatch({ type: "reset" });
  }, []);

  useEffect(() => () => requestRef.current.controller?.abort(), []);

  return { ...state, submitText, submitAudio, retryResponse, cancel, reset: cancel };
}
