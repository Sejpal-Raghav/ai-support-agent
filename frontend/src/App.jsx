import { useState } from "react";
import { AppHeader } from "./components/AppHeader";
import { ResponsePanel } from "./components/ResponsePanel";
import { StatusMessage } from "./components/StatusMessage";
import { TextComposer } from "./components/TextComposer";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { VoiceRecorder } from "./components/VoiceRecorder";
import { useServiceHealth } from "./hooks/useServiceHealth";
import { useSupportConversation } from "./hooks/useSupportConversation";
import { useVoiceRecorder } from "./hooks/useVoiceRecorder";

export default function App() {
  const [draft, setDraft] = useState("");
  const service = useServiceHealth();
  const conversation = useSupportConversation();
  const recorder = useVoiceRecorder(conversation.submitAudio);
  const busy = ["transcribing", "responding"].includes(conversation.status);
  const textReady = Boolean(service.health?.classifier_ready && service.health?.retrieval_ready);
  const voiceReady = Boolean(textReady && service.health?.asr_ready);
  const serviceError = service.status === "unavailable" ? "The local service did not become ready." : "";

  const sendText = async (text) => {
    setDraft("");
    await conversation.submitText(text, "typed");
  };
  const startRecording = () => {
    conversation.reset();
    recorder.start();
  };
  const editTranscript = (text) => {
    setDraft(text);
    document.getElementById("typed-request")?.focus();
  };

  return (
    <main className="app-shell">
      <AppHeader serviceStatus={service.status} />
      <section className="workspace" aria-label="Voice support workspace">
        <VoiceRecorder
          recorder={recorder}
          disabled={!voiceReady || busy}
          onStart={startRecording}
        />
        <div className="conversation-panel">
          <TranscriptPanel
            transcript={conversation.transcript}
            source={conversation.inputSource}
            status={conversation.status}
            onEdit={editTranscript}
          />
          <ResponsePanel result={conversation.result} status={conversation.status} />
          <StatusMessage
            error={recorder.error || conversation.error || serviceError}
            canRetry={Boolean(conversation.error && conversation.transcript)}
            onRetry={conversation.retryResponse}
            onCheckAgain={service.status === "unavailable" ? service.checkAgain : null}
          />
          {service.status === "checking" && (
            <p className="processing-status" role="status">Checking microphone service…</p>
          )}
        </div>
        <TextComposer
          value={draft}
          onChange={setDraft}
          onSend={sendText}
          disabled={!textReady || busy || recorder.status !== "idle"}
        />
      </section>
      <footer>
        Independent student demo. Not affiliated with Amazon. It cannot access accounts or complete transactions.
      </footer>
    </main>
  );
}
