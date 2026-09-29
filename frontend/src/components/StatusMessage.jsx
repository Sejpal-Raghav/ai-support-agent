export function StatusMessage({ error, canRetry, onRetry, onCheckAgain }) {
  if (!error) return null;
  return (
    <div className="status-message" role="alert">
      <p>{error}</p>
      <div className="status-actions">
        {canRetry && <button type="button" onClick={onRetry}>Retry response</button>}
        {onCheckAgain && <button type="button" onClick={onCheckAgain}>Check again</button>}
      </div>
    </div>
  );
}
