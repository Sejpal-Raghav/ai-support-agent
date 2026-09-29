const ERROR_MESSAGES = {
  microphone_permission_denied: "Microphone access is blocked. Allow it in your browser, then try again.",
  microphone_missing: "No microphone was found. Connect one or type your question.",
  unsupported_recorder: "This browser cannot record a supported audio format. Use Chrome or Edge, or type your question.",
  empty_audio: "No audio was received. Move closer to the microphone and try again.",
  empty_transcript: "No speech was detected. Move closer to the microphone and try again.",
  audio_too_long: "The recording exceeded 30 seconds. Record a shorter question and try again.",
  audio_too_large: "The recording is too large. Record a shorter question and try again.",
  unsupported_audio_type: "This browser cannot record a supported audio format. Use Chrome or Edge, or type your question.",
  invalid_audio: "The recording could not be read. Try recording again.",
  invalid_pcm_audio: "The recording could not be read. Try recording again.",
  asr_unavailable: "Speech recognition is unavailable. You can still type your question.",
  response_unavailable: "The response could not be loaded. Your text is still available.",
  request_timeout: "This request took too long. Try again.",
  network_error: "The local service could not be reached. Check that it is running, then try again.",
  response_failed: "The response could not be loaded. Your text is still available.",
};

export function errorMessage(code) {
  return ERROR_MESSAGES[code] || "Something went wrong. Try again or type your question.";
}

export function microphoneErrorCode(error) {
  if (["NotAllowedError", "SecurityError"].includes(error?.name)) {
    return "microphone_permission_denied";
  }
  if (["NotFoundError", "DevicesNotFoundError"].includes(error?.name)) {
    return "microphone_missing";
  }
  return "unsupported_recorder";
}
