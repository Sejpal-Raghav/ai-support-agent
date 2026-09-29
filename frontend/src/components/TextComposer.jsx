export function TextComposer({ value, onChange, onSend, disabled }) {
  const submit = (event) => {
    event.preventDefault();
    if (value.trim() && !disabled) onSend(value);
  };
  const keyDown = (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      if (value.trim() && !disabled) onSend(value);
    }
  };
  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="typed-request">Or type your question</label>
      <div className="composer-row">
        <textarea
          id="typed-request"
          rows="2"
          maxLength="1000"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={keyDown}
          placeholder="For example: Where is my delayed package?"
        />
        <button className="send-button" type="submit" disabled={disabled || !value.trim()}>
          Send question
        </button>
      </div>
      <p className="keyboard-hint">Ctrl+Enter or ⌘+Enter to send</p>
    </form>
  );
}
