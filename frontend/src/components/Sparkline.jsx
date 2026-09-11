export default function Sparkline({ values = [] }) {
  if (!values.length) {
    return <p className="empty">Loss history is not available yet.</p>;
  }
  const width = 640;
  const height = 120;
  const pad = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values.map((value, index) => {
    const x = pad + (index / Math.max(values.length - 1, 1)) * (width - pad * 2);
    const y = height - pad - ((value - min) / span) * (height - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });

  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training loss">
      <polyline
        fill="none"
        stroke="var(--copper)"
        strokeWidth="2.2"
        points={points.join(" ")}
      />
      <text x={pad} y="14" fill="var(--muted)" fontSize="11" fontFamily="Outfit, sans-serif">
        {max.toFixed(3)}
      </text>
      <text x={pad} y={height - 2} fill="var(--muted)" fontSize="11" fontFamily="Outfit, sans-serif">
        {min.toFixed(3)}
      </text>
    </svg>
  );
}
