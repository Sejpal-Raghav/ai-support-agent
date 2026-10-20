export function AppHeader({ serviceStatus }) {
  const ready = serviceStatus === "ready";
  const label = ready ? "Ready" : serviceStatus === "partial" ? "Text only" : serviceStatus === "unavailable" ? "Unavailable" : "Starting";
  return (
    <header className="app-header">
      <div className="brand-row">
        <span className="brand-mark" aria-hidden="true">S</span>
        <p className="product-name">Support Listening Desk</p>
        <span className={`service-badge service-badge--${serviceStatus}`}>{label}</span>
      </div>
      <h1>How can we help with your order?</h1>
      <p className="intro">Speak or type an Amazon-related support question.</p>
    </header>
  );
}
